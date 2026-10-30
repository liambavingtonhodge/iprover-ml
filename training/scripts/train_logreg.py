#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "num_literals",
    "num_symbols",
    "max_depth",
    "num_variables",
    "age",
    "shared_conj_symbols",
    "is_ground",
    "has_equality",
    "clause_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a logistic regression model for clause selection with either "
            "row-level or problem-level train/dev split."
        )
    )
    parser.add_argument("--train-csv", required=True, help="Path to clause-level training CSV.")
    parser.add_argument("--label-col", default="selected", help="Name of label column (default: selected).")
    parser.add_argument("--problem-col", default="problem", help="Name of problem column in the clause CSV (default: problem).")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns to use.")
    parser.add_argument("--model-out", required=True, help="Path to save trained model (.joblib).")
    parser.add_argument("--metrics-out", default="", help="Optional path to save metrics as JSON.")
    parser.add_argument("--coeffs-out", default="", help="Optional path to save learned coefficients as CSV.")
    parser.add_argument("--dev-fraction", type=float, default=0.1, help="Fraction to hold out for dev split (default: 0.1).")
    parser.add_argument(
        "--split-mode",
        choices=["row", "problem"],
        default="row",
        help="Split mode: row = stratified row split, problem = stratified problem split (default: row).",
    )
    parser.add_argument(
        "--split-metadata-csv",
        default="",
        help="For split-mode=problem: metadata CSV containing problem/domain/runtime or problem/domain/difficulty_bin.",
    )
    parser.add_argument(
        "--split-metadata-problem-col",
        default="problem",
        help="Problem column name in metadata CSV (default: problem).",
    )
    parser.add_argument(
        "--problem-stratify-cols",
        nargs="+",
        default=["domain", "difficulty_bin"],
        help="For split-mode=problem: metadata columns used for stratification (default: domain difficulty_bin).",
    )
    parser.add_argument(
        "--runtime-col",
        default="runtime_seconds",
        help="Runtime column in metadata CSV used to derive difficulty_bin if needed (default: runtime_seconds).",
    )
    parser.add_argument(
        "--num-difficulty-bins",
        type=int,
        default=3,
        help="Number of runtime-based difficulty bins to create if difficulty_bin is missing (default: 3).",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for LogisticRegression (default: 1000).")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularisation strength C (default: 1.0).")
    parser.add_argument(
        "--solver",
        default="liblinear",
        choices=["liblinear", "lbfgs", "saga", "newton-cg", "sag"],
        help="LogisticRegression solver (default: liblinear).",
    )
    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["balanced", "none"],
        help="Class weighting strategy (default: balanced).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def load_clause_dataset(
    path: Path,
    features: List[str],
    label_col: str,
    split_mode: str,
    problem_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = list(features) + [label_col]
    if split_mode == "problem":
        required.append(problem_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[label_col])
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing label values.")

    unique_labels = set(df[label_col].dropna().unique())
    bad_labels = sorted(unique_labels - {0, 1})
    if bad_labels:
        raise ValueError(f"Label column '{label_col}' must contain only 0/1 values. Found: {bad_labels}")

    df[label_col] = df[label_col].astype(int)

    if split_mode == "problem":
        df[problem_col] = df[problem_col].astype(str).str.strip()

    return df


def make_difficulty_bins(runtime_series: pd.Series, num_bins: int) -> pd.Series:
    if num_bins < 2:
        raise ValueError("--num-difficulty-bins must be at least 2")

    runtime_numeric = pd.to_numeric(runtime_series, errors="coerce")
    if runtime_numeric.isna().any():
        raise ValueError("Runtime column contains missing or non-numeric values; cannot derive difficulty bins.")

    labels = [f"bin_{i}" for i in range(num_bins)]

    try:
        binned = pd.qcut(runtime_numeric, q=num_bins, labels=labels, duplicates="drop")
        if binned.isna().any() or binned.nunique(dropna=True) < 2:
            raise ValueError("qcut produced insufficient distinct bins")
        return binned.astype(str)
    except Exception:
        ranked = runtime_numeric.rank(method="average")
        binned = pd.qcut(ranked, q=num_bins, labels=labels, duplicates="drop")
        if binned.isna().any() or binned.nunique(dropna=True) < 2:
            raise ValueError("Could not derive usable difficulty bins from runtime column")
        return binned.astype(str)


def load_split_metadata(
    metadata_csv: Path,
    metadata_problem_col: str,
    problem_stratify_cols: List[str],
    runtime_col: str,
    num_difficulty_bins: int,
) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    meta.columns = [c.strip() for c in meta.columns]

    base_required = [metadata_problem_col]
    missing_base = [col for col in base_required if col not in meta.columns]
    if missing_base:
        raise ValueError(f"Missing required columns in metadata CSV {metadata_csv}: {missing_base}")

    needs_difficulty = "difficulty_bin" in problem_stratify_cols and "difficulty_bin" not in meta.columns
    if needs_difficulty:
        if runtime_col not in meta.columns:
            raise ValueError(
                f"Metadata CSV {metadata_csv} needs either 'difficulty_bin' or runtime column '{runtime_col}' "
                f"to derive difficulty bins."
            )
        meta["difficulty_bin"] = make_difficulty_bins(meta[runtime_col], num_difficulty_bins)

    required = [metadata_problem_col] + problem_stratify_cols
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise ValueError(f"Missing required columns in metadata CSV {metadata_csv}: {missing}")

    meta = meta[required].copy()
    meta[metadata_problem_col] = meta[metadata_problem_col].astype(str).str.strip()

    before = len(meta)
    meta = meta.drop_duplicates(subset=[metadata_problem_col], keep="first")
    after = len(meta)
    if after < before:
        print(f"Dropped {before - after} duplicate problem rows from metadata CSV.")

    for col in problem_stratify_cols:
        meta[col] = meta[col].astype(str).str.strip()

    return meta


def build_stratify_labels(meta: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(["all"] * len(meta), index=meta.index)
    return meta[cols].astype(str).agg("||".join, axis=1)


def split_dataset(
    df: pd.DataFrame,
    label_col: str,
    problem_col: str,
    split_mode: str,
    dev_fraction: float,
    random_state: int,
    metadata_csv: Path | None,
    metadata_problem_col: str,
    problem_stratify_cols: List[str],
    runtime_col: str,
    num_difficulty_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    if split_mode == "row":
        train_df, dev_df = train_test_split(
            df,
            test_size=dev_fraction,
            random_state=random_state,
            stratify=df[label_col],
        )
        return train_df.copy(), dev_df.copy(), {
            "row_stratified_by_label": True,
            "problem_split_stratified": None,
            "num_unique_problems_total": None,
            "num_unique_problems_train": None,
            "num_unique_problems_dev": None,
            "split_metadata_csv": None,
            "split_metadata_problem_col": None,
            "problem_stratify_cols": None,
            "runtime_col": None,
            "num_difficulty_bins": None,
        }

    if metadata_csv is None:
        raise ValueError("--split-metadata-csv is required when --split-mode problem")

    meta = load_split_metadata(
        metadata_csv=metadata_csv,
        metadata_problem_col=metadata_problem_col,
        problem_stratify_cols=problem_stratify_cols,
        runtime_col=runtime_col,
        num_difficulty_bins=num_difficulty_bins,
    )

    clause_problem_set = set(df[problem_col].astype(str).str.strip().unique())
    meta_problem_set = set(meta[metadata_problem_col].unique())

    missing_in_meta = sorted(clause_problem_set - meta_problem_set)
    if missing_in_meta:
        preview = missing_in_meta[:10]
        raise ValueError(
            "Some problems in the clause CSV are missing from the metadata CSV. "
            f"Examples: {preview}"
        )

    meta = meta[meta[metadata_problem_col].isin(clause_problem_set)].copy()

    stratify_labels = build_stratify_labels(meta, problem_stratify_cols)
    stratify_counts = stratify_labels.value_counts()
    can_stratify = bool((stratify_counts >= 2).all() and stratify_labels.nunique() > 1)

    if can_stratify:
        train_probs, dev_probs = train_test_split(
            meta[metadata_problem_col],
            test_size=dev_fraction,
            random_state=random_state,
            stratify=stratify_labels,
        )
        problem_split_stratified = True
    else:
        print(
            "Warning: could not fully stratify problem split by the requested metadata "
            "columns because at least one stratum had fewer than 2 problems. "
            "Falling back to unstratified problem split."
        )
        train_probs, dev_probs = train_test_split(
            meta[metadata_problem_col],
            test_size=dev_fraction,
            random_state=random_state,
            stratify=None,
        )
        problem_split_stratified = False

    train_problem_set = set(train_probs.astype(str).tolist())
    dev_problem_set = set(dev_probs.astype(str).tolist())

    train_df = df[df[problem_col].isin(train_problem_set)].copy()
    dev_df = df[df[problem_col].isin(dev_problem_set)].copy()

    split_info = {
        "row_stratified_by_label": False,
        "problem_split_stratified": problem_split_stratified,
        "num_unique_problems_total": int(len(clause_problem_set)),
        "num_unique_problems_train": int(train_df[problem_col].nunique()),
        "num_unique_problems_dev": int(dev_df[problem_col].nunique()),
        "split_metadata_csv": str(metadata_csv),
        "split_metadata_problem_col": metadata_problem_col,
        "problem_stratify_cols": problem_stratify_cols,
        "runtime_col": runtime_col if "difficulty_bin" in problem_stratify_cols else None,
        "num_difficulty_bins": num_difficulty_bins if "difficulty_bin" in problem_stratify_cols else None,
    }
    return train_df, dev_df, split_info


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=6, zero_division=0, output_dict=True
        ),
    }


def print_metrics(split_name: str, metrics: dict) -> None:
    print(f"\n{split_name} metrics")
    print("-" * (len(split_name) + 8))
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1       : {metrics['f1']:.6f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC  : {metrics['roc_auc']:.6f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")


def main() -> int:
    args = parse_args()

    if not (0.0 < args.dev_fraction < 1.0):
        raise ValueError("--dev-fraction must be strictly between 0 and 1")

    train_csv = Path(args.train_csv).resolve()
    model_out = Path(args.model_out).resolve()
    metrics_out = Path(args.metrics_out).resolve() if args.metrics_out else None
    coeffs_out = Path(args.coeffs_out).resolve() if args.coeffs_out else None
    metadata_csv = Path(args.split_metadata_csv).resolve() if args.split_metadata_csv else None

    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if args.split_mode == "problem" and metadata_csv is not None and not metadata_csv.is_file():
        raise FileNotFoundError(f"Split metadata CSV not found: {metadata_csv}")

    df = load_clause_dataset(
        path=train_csv,
        features=args.features,
        label_col=args.label_col,
        split_mode=args.split_mode,
        problem_col=args.problem_col,
    )

    print(f"Loaded {len(df)} rows from {train_csv}")
    print(f"Overall class distribution: {df[args.label_col].value_counts().to_dict()}")

    train_df, dev_df, split_info = split_dataset(
        df=df,
        label_col=args.label_col,
        problem_col=args.problem_col,
        split_mode=args.split_mode,
        dev_fraction=args.dev_fraction,
        random_state=args.random_state,
        metadata_csv=metadata_csv,
        metadata_problem_col=args.split_metadata_problem_col,
        problem_stratify_cols=args.problem_stratify_cols,
        runtime_col=args.runtime_col,
        num_difficulty_bins=args.num_difficulty_bins,
    )

    X_train = train_df[args.features]
    y_train = train_df[args.label_col]
    X_dev = dev_df[args.features]
    y_dev = dev_df[args.label_col]

    print(f"Train rows: {len(X_train)}")
    print(f"Dev rows  : {len(X_dev)}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Dev class distribution  : {y_dev.value_counts().to_dict()}")

    if args.split_mode == "problem":
        print(f"Unique problems total: {split_info['num_unique_problems_total']}")
        print(f"Unique problems train: {split_info['num_unique_problems_train']}")
        print(f"Unique problems dev  : {split_info['num_unique_problems_dev']}")
        print(f"Problem split stratified: {split_info['problem_split_stratified']}")
        print(f"Problem stratify cols   : {split_info['problem_stratify_cols']}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, args.features),
        ]
    )

    class_weight = None if args.class_weight == "none" else args.class_weight

    clf = LogisticRegression(
        max_iter=args.max_iter,
        C=args.c,
        solver=args.solver,
        class_weight=class_weight,
        random_state=args.random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_dev_pred = model.predict(X_dev)
    y_dev_prob = model.predict_proba(X_dev)[:, 1]

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
    dev_metrics = compute_metrics(y_dev, y_dev_pred, y_dev_prob)

    classifier = model.named_steps["classifier"]

    metrics = {
        "split_mode": args.split_mode,
        **split_info,
        "problem_col": args.problem_col if args.split_mode == "problem" else None,
        "num_rows_total": int(len(df)),
        "num_rows_train": int(len(X_train)),
        "num_rows_dev": int(len(X_dev)),
        "num_positive_total": int((df[args.label_col] == 1).sum()),
        "num_negative_total": int((df[args.label_col] == 0).sum()),
        "num_positive_train": int((y_train == 1).sum()),
        "num_negative_train": int((y_train == 0).sum()),
        "num_positive_dev": int((y_dev == 1).sum()),
        "num_negative_dev": int((y_dev == 0).sum()),
        "features": args.features,
        "label_col": args.label_col,
        "dev_fraction": args.dev_fraction,
        "max_iter": args.max_iter,
        "C": args.c,
        "solver": args.solver,
        "class_weight": args.class_weight,
        "random_state": args.random_state,
        "train_time_seconds": round(train_time, 6),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "intercept": float(classifier.intercept_[0]),
    }

    print_metrics("Train", train_metrics)
    print_metrics("Dev", dev_metrics)

    print("\nDev classification report")
    print("-------------------------")
    print(classification_report(y_dev, y_dev_pred, digits=6, zero_division=0))

    print(f"Training time: {train_time:.2f} seconds")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Saved model to {model_out}")

    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_out}")

    if coeffs_out is not None:
        coeffs_out.parent.mkdir(parents=True, exist_ok=True)
        coefs = classifier.coef_[0]
        coeff_df = pd.DataFrame({
            "feature": args.features,
            "coefficient": coefs,
            "abs_coefficient": [abs(x) for x in coefs],
        }).sort_values("abs_coefficient", ascending=False)
        coeff_df.to_csv(coeffs_out, index=False)
        print(f"Saved coefficients to {coeffs_out}")
        print(f"Intercept: {metrics['intercept']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())