#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a pairwise logistic regression ranking model from delta features, "
            "with row-level, problem-level, or no train/dev split."
        )
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--dev-fraction", type=float, default=0.1)

    parser.add_argument(
        "--split-mode",
        choices=["row", "problem", "none"],
        default="row",
        help="row = random row split, problem = split by problem, none = train on all rows",
    )
    parser.add_argument("--problem-col", default="problem")
    parser.add_argument("--split-metadata-csv", default="")
    parser.add_argument("--split-metadata-problem-col", default="problem")
    parser.add_argument("--problem-stratify-cols", nargs="+", default=["domain", "difficulty_bin"])
    parser.add_argument("--runtime-col", default="runtime_seconds")
    parser.add_argument("--num-difficulty-bins", type=int, default=3)

    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        default="liblinear",
        choices=["liblinear", "lbfgs", "saga", "newton-cg", "sag"],
    )
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def load_dataset(path: Path, split_mode: str, problem_col: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    feature_cols = [c for c in df.columns if c.startswith("delta_")]

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in pairwise dataset.")
    if not feature_cols:
        raise ValueError("No delta_* feature columns found.")

    required = feature_cols + ["label"]
    if split_mode == "problem":
        required.append(problem_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    bad_labels = sorted(set(df["label"].dropna().astype(int).unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"Label column must contain only 0/1 values. Found: {bad_labels}")

    df["label"] = df["label"].astype(int)

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    if split_mode == "problem":
        df[problem_col] = df[problem_col].astype(str).str.strip()

    return df, feature_cols


def make_difficulty_bins(runtime_series: pd.Series, num_bins: int) -> pd.Series:
    runtime_numeric = pd.to_numeric(runtime_series, errors="coerce")
    if runtime_numeric.isna().any():
        raise ValueError("Runtime column contains missing/non-numeric values.")

    labels = [f"bin_{i}" for i in range(num_bins)]
    try:
        return pd.qcut(runtime_numeric, q=num_bins, labels=labels, duplicates="drop").astype(str)
    except Exception:
        ranked = runtime_numeric.rank(method="average")
        return pd.qcut(ranked, q=num_bins, labels=labels, duplicates="drop").astype(str)


def load_split_metadata(
    metadata_csv: Path,
    metadata_problem_col: str,
    problem_stratify_cols: List[str],
    runtime_col: str,
    num_difficulty_bins: int,
) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    meta.columns = [c.strip() for c in meta.columns]

    if metadata_problem_col not in meta.columns:
        raise ValueError(f"Missing problem column in metadata CSV: {metadata_problem_col}")

    if "difficulty_bin" in problem_stratify_cols and "difficulty_bin" not in meta.columns:
        if runtime_col not in meta.columns:
            raise ValueError(
                f"Need either difficulty_bin or runtime column '{runtime_col}' in metadata CSV."
            )
        meta["difficulty_bin"] = make_difficulty_bins(meta[runtime_col], num_difficulty_bins)

    required = [metadata_problem_col] + problem_stratify_cols
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    meta = meta[required].copy()
    meta[metadata_problem_col] = meta[metadata_problem_col].astype(str).str.strip()
    meta = meta.drop_duplicates(subset=[metadata_problem_col], keep="first")

    for col in problem_stratify_cols:
        meta[col] = meta[col].astype(str).str.strip()

    return meta


def build_stratify_labels(meta: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(["all"] * len(meta), index=meta.index)
    return meta[cols].astype(str).agg("||".join, axis=1)


def split_dataset(
    df: pd.DataFrame,
    split_mode: str,
    dev_fraction: float,
    random_state: int,
    problem_col: str,
    metadata_csv: Optional[Path],
    metadata_problem_col: str,
    problem_stratify_cols: List[str],
    runtime_col: str,
    num_difficulty_bins: int,
):
    if split_mode == "none":
        split_info = {
            "split_mode": "none",
            "row_stratified_by_label": False,
            "problem_split_stratified": None,
            "num_unique_problems_total": int(df[problem_col].nunique()) if problem_col in df.columns else None,
            "num_unique_problems_train": int(df[problem_col].nunique()) if problem_col in df.columns else None,
            "num_unique_problems_dev": 0,
            "split_metadata_csv": None,
            "problem_stratify_cols": None,
        }
        return df.copy(), None, split_info

    if not (0.0 < dev_fraction < 1.0):
        raise ValueError("--dev-fraction must be strictly between 0 and 1 unless --split-mode none")

    if split_mode == "row":
        train_df, dev_df = train_test_split(
            df,
            test_size=dev_fraction,
            random_state=random_state,
            stratify=df["label"],
        )
        split_info = {
            "split_mode": "row",
            "row_stratified_by_label": True,
            "problem_split_stratified": None,
            "num_unique_problems_total": None,
            "num_unique_problems_train": None,
            "num_unique_problems_dev": None,
            "split_metadata_csv": None,
            "problem_stratify_cols": None,
        }
        return train_df.copy(), dev_df.copy(), split_info

    unique_problem_set = set(df[problem_col].astype(str).str.strip().unique())

    if metadata_csv is not None:
        meta = load_split_metadata(
            metadata_csv=metadata_csv,
            metadata_problem_col=metadata_problem_col,
            problem_stratify_cols=problem_stratify_cols,
            runtime_col=runtime_col,
            num_difficulty_bins=num_difficulty_bins,
        )

        meta_problem_set = set(meta[metadata_problem_col].astype(str).str.strip().unique())
        missing = sorted(unique_problem_set - meta_problem_set)
        if missing:
            raise ValueError(
                "Some problems in the pairwise CSV are missing from metadata CSV. "
                f"Examples: {missing[:10]}"
            )

        meta = meta[meta[metadata_problem_col].isin(unique_problem_set)].copy()
        split_problem_series = meta[metadata_problem_col]
        stratify_labels = build_stratify_labels(meta, problem_stratify_cols)

        counts = stratify_labels.value_counts()
        can_stratify = bool((counts >= 2).all() and stratify_labels.nunique() > 1)

        train_probs, dev_probs = train_test_split(
            split_problem_series,
            test_size=dev_fraction,
            random_state=random_state,
            stratify=stratify_labels if can_stratify else None,
        )
        problem_split_stratified = can_stratify
    else:
        problem_df = pd.DataFrame({"problem": sorted(unique_problem_set)})
        train_probs, dev_probs = train_test_split(
            problem_df["problem"],
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
        "split_mode": "problem",
        "row_stratified_by_label": False,
        "problem_split_stratified": problem_split_stratified,
        "num_unique_problems_total": int(len(unique_problem_set)),
        "num_unique_problems_train": int(train_df[problem_col].nunique()),
        "num_unique_problems_dev": int(dev_df[problem_col].nunique()),
        "split_metadata_csv": str(metadata_csv) if metadata_csv else None,
        "problem_stratify_cols": problem_stratify_cols if metadata_csv else None,
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


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name} metrics")
    print("-" * 20)
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1       : {metrics['f1']:.6f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC  : {metrics['roc_auc']:.6f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")


def main() -> int:
    args = parse_args()

    train_csv = Path(args.train_csv).resolve()
    model_out = Path(args.model_out).resolve()
    metrics_out = Path(args.metrics_out).resolve() if args.metrics_out else None
    metadata_csv = Path(args.split_metadata_csv).resolve() if args.split_metadata_csv else None

    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df, feature_cols = load_dataset(
        path=train_csv,
        split_mode=args.split_mode,
        problem_col=args.problem_col,
    )

    print(f"Loaded {len(df)} rows from {train_csv}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"Feature columns: {feature_cols}")

    train_df, dev_df, split_info = split_dataset(
        df=df,
        split_mode=args.split_mode,
        dev_fraction=args.dev_fraction,
        random_state=args.random_state,
        problem_col=args.problem_col,
        metadata_csv=metadata_csv,
        metadata_problem_col=args.split_metadata_problem_col,
        problem_stratify_cols=args.problem_stratify_cols,
        runtime_col=args.runtime_col,
        num_difficulty_bins=args.num_difficulty_bins,
    )

    X_train = train_df[feature_cols]
    y_train = train_df["label"]

    print(f"Train rows: {len(X_train)}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")

    if dev_df is not None:
        X_dev = dev_df[feature_cols]
        y_dev = dev_df["label"]
        print(f"Dev rows  : {len(X_dev)}")
        print(f"Dev class distribution  : {y_dev.value_counts().to_dict()}")
    else:
        X_dev = None
        y_dev = None
        print("Dev rows  : 0 (no split)")

    if args.split_mode == "problem":
        print(f"Unique problems total: {split_info['num_unique_problems_total']}")
        print(f"Unique problems train: {split_info['num_unique_problems_train']}")
        print(f"Unique problems dev  : {split_info['num_unique_problems_dev']}")
        print(f"Problem split stratified: {split_info['problem_split_stratified']}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
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

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)

    print_metrics("Train", train_metrics)

    dev_metrics = None
    if dev_df is not None:
        y_dev_pred = model.predict(X_dev)
        y_dev_prob = model.predict_proba(X_dev)[:, 1]
        dev_metrics = compute_metrics(y_dev, y_dev_pred, y_dev_prob)

        print_metrics("Dev", dev_metrics)
        print("\nDev classification report")
        print("-------------------------")
        print(classification_report(y_dev, y_dev_pred, digits=6, zero_division=0))

    print(f"\nTraining time: {train_time:.2f}s")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Saved model to {model_out}")

    if metrics_out is not None:
        metrics = {
            **split_info,
            "num_rows_total": int(len(df)),
            "num_rows_train": int(len(train_df)),
            "num_rows_dev": int(len(dev_df)) if dev_df is not None else 0,
            "num_positive_total": int((df["label"] == 1).sum()),
            "num_negative_total": int((df["label"] == 0).sum()),
            "num_positive_train": int((y_train == 1).sum()),
            "num_negative_train": int((y_train == 0).sum()),
            "num_positive_dev": int((y_dev == 1).sum()) if y_dev is not None else 0,
            "num_negative_dev": int((y_dev == 0).sum()) if y_dev is not None else 0,
            "features": feature_cols,
            "label_col": "label",
            "dev_fraction": args.dev_fraction if args.split_mode != "none" else None,
            "max_iter": args.max_iter,
            "C": args.c,
            "solver": args.solver,
            "class_weight": args.class_weight,
            "random_state": args.random_state,
            "train_time_seconds": round(train_time, 6),
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
        }

        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {metrics_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())