#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
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


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "hidden-layer-sizes must be a comma-separated list of positive integers, e.g. 32,16"
        )

    try:
        sizes = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden-layer-sizes must contain only integers"
        ) from exc

    if any(s <= 0 for s in sizes):
        raise argparse.ArgumentTypeError(
            "hidden-layer-sizes must contain only positive integers"
        )

    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP model for clause selection on the full dataset (no external train/dev split)."
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
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations for MLPClassifier (default: 100).",
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        type=parse_hidden_layers,
        default=(32, 16),
        help="Comma-separated hidden layer sizes, e.g. 32,16 (default: 32,16).",
    )
    parser.add_argument(
        "--activation",
        default="relu",
        choices=["identity", "logistic", "tanh", "relu"],
        help="Activation function for the hidden layers (default: relu).",
    )
    parser.add_argument(
        "--solver",
        default="adam",
        choices=["lbfgs", "sgd", "adam"],
        help="MLP solver (default: adam).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="L2 regularisation term alpha (default: 0.0001).",
    )
    parser.add_argument(
        "--learning-rate",
        default="constant",
        choices=["constant", "invscaling", "adaptive"],
        help="Learning rate schedule for SGD/adam (default: constant).",
    )
    parser.add_argument(
        "--learning-rate-init",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping using sklearn's internal validation split.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Validation fraction for early stopping (default: 0.1).",
    )
    parser.add_argument(
        "--n-iter-no-change",
        type=int,
        default=10,
        help="Number of epochs with no improvement before stopping (default: 10).",
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

    clf = MLPClassifier(
        hidden_layer_sizes=args.hidden_layer_sizes,
        activation=args.activation,
        solver=args.solver,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=args.early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
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

    classifier = model.named_steps["classifier"]

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
        "hidden_layer_sizes": list(args.hidden_layer_sizes),
        "activation": args.activation,
        "solver": args.solver,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "learning_rate_init": args.learning_rate_init,
        "max_iter": args.max_iter,
        "early_stopping": args.early_stopping,
        "validation_fraction": args.validation_fraction,
        "n_iter_no_change": args.n_iter_no_change,
        "random_state": args.random_state,
        "n_iter_": int(classifier.n_iter_),
        "loss_": float(classifier.loss_),
        "best_validation_score_": (
            float(classifier.best_validation_score_)
            if args.early_stopping and hasattr(classifier, "best_validation_score_")
            and classifier.best_validation_score_ is not None
            else None
        ),
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
    print(f"Iterations used : {metrics['n_iter_']}")
    print(f"Final loss      : {metrics['loss_']:.6f}")
    if metrics["best_validation_score_"] is not None:
        print(f"Best val score  : {metrics['best_validation_score_']:.6f}")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())