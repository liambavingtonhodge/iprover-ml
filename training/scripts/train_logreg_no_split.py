#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List

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
        description="Train a logistic regression model for clause selection on the full dataset (no external train/dev split)."
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Path to clause-level training CSV.",
    )
    parser.add_argument(
        "--label-col",
        default="selected",
        help="Name of label column (default: selected).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to use.",
    )
    parser.add_argument(
        "--model-out",
        required=True,
        help="Path to save trained model (.joblib).",
    )
    parser.add_argument(
        "--metrics-out",
        default="",
        help="Optional path to save training-fit metrics as JSON.",
    )
    parser.add_argument(
        "--coeffs-out",
        default="",
        help="Optional path to save learned coefficients as CSV.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for LogisticRegression (default: 1000).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength C (default: 1.0).",
    )
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
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def load_dataset(path: Path, features: List[str], label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = list(features) + [label_col]
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

    valid_labels = {0, 1}
    bad_labels = sorted(set(df[label_col].dropna().astype(int).unique()) - valid_labels)
    if bad_labels:
        raise ValueError(
            f"Label column '{label_col}' must contain only 0/1 values. Found: {bad_labels}"
        )

    df[label_col] = df[label_col].astype(int)
    return df


def main() -> int:
    args = parse_args()

    train_csv = Path(args.train_csv).resolve()
    model_out = Path(args.model_out).resolve()
    metrics_out = Path(args.metrics_out).resolve() if args.metrics_out else None
    coeffs_out = Path(args.coeffs_out).resolve() if args.coeffs_out else None

    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df = load_dataset(train_csv, args.features, args.label_col)

    X = df[args.features]
    y = df[args.label_col]

    class_counts = y.value_counts().to_dict()
    print(f"Loaded {len(df)} training rows from {train_csv}")
    print(f"Class distribution: {class_counts}")

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

    model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "num_rows": int(len(df)),
        "num_positive": int((y == 1).sum()),
        "num_negative": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)) if len(set(y)) > 1 else None,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "features": args.features,
        "label_col": args.label_col,
        "max_iter": args.max_iter,
        "C": args.c,
        "solver": args.solver,
        "class_weight": args.class_weight,
        "random_state": args.random_state,
        "metric_scope": "training_fit",
    }

    print("\nTraining-fit metrics")
    print("--------------------")
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1       : {metrics['f1']:.6f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC  : {metrics['roc_auc']:.6f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    print("\nClassification report")
    print("---------------------")
    print(classification_report(y, y_pred, digits=6, zero_division=0))

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

        classifier = model.named_steps["classifier"]
        coefs = classifier.coef_[0]
        intercept = float(classifier.intercept_[0])

        coeff_df = pd.DataFrame({
            "feature": args.features,
            "coefficient": coefs,
            "abs_coefficient": [abs(x) for x in coefs],
        }).sort_values("abs_coefficient", ascending=False)

        coeff_df.to_csv(coeffs_out, index=False)
        print(f"Saved coefficients to {coeffs_out}")
        print(f"Intercept: {intercept:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())