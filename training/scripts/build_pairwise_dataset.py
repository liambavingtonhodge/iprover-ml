#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd


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
            "Build a pairwise ranking dataset from clause-level data. "
            "Works for both given-clause decision data and proof-membership data."
        )
    )

    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)

    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to use.",
    )

    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=["problem", "decision_index"],
        help=(
            "Columns defining a comparison group. "
            "Use 'problem decision_index' for given-clause data; "
            "use 'problem' for proof-based data."
        ),
    )

    parser.add_argument(
        "--label-col",
        default="selected",
        help="Binary label column in the clause-level CSV.",
    )

    parser.add_argument(
        "--max-pairs-per-positive",
        type=int,
        default=10,
        help=(
            "Maximum number of negative clauses to pair with each positive clause. "
            "Use -1 to pair each positive with all negatives in the group."
        ),
    )

    parser.add_argument(
        "--include-reverse-pairs",
        action="store_true",
        help="Also include reverse pairs labelled 0.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_dataset(
    path: Path,
    features: List[str],
    label_col: str,
    group_cols: List[str],
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = list(group_cols) + list(features) + [label_col]
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

    for col in features:
        median = df[col].median()
        if pd.isna(median):
            median = 0.0
        df[col] = df[col].fillna(median)

    df[label_col] = df[label_col].astype(int)

    bad_labels = sorted(set(df[label_col].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(
            f"Label column '{label_col}' must contain only 0/1 values. "
            f"Found: {bad_labels}"
        )

    return df


def normalise_group_key(group_key) -> Tuple:
    if isinstance(group_key, tuple):
        return group_key
    return (group_key,)


def write_pair(
    writer: csv.DictWriter,
    group_cols: List[str],
    group_key: Tuple,
    features: List[str],
    left_row: pd.Series,
    right_row: pd.Series,
    label: int,
) -> None:
    row = {}

    for col_name, col_value in zip(group_cols, group_key):
        row[col_name] = col_value

    for feat in features:
        row[f"delta_{feat}"] = float(left_row[feat]) - float(right_row[feat])

    row["label"] = label
    writer.writerow(row)


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rng = random.Random(args.random_seed)

    df = load_dataset(
        path=input_csv,
        features=args.features,
        label_col=args.label_col,
        group_cols=args.group_cols,
    )

    out_fieldnames = (
        list(args.group_cols)
        + [f"delta_{feat}" for feat in args.features]
        + ["label"]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    num_groups = 0
    num_skipped_groups = 0
    num_positive_rows = 0
    num_negative_rows = 0
    num_pairs_written = 0

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        grouped = df.groupby(args.group_cols, sort=False, dropna=False)

        for group_key, group_df in grouped:
            num_groups += 1
            group_key = normalise_group_key(group_key)

            positives = group_df[group_df[args.label_col] == 1]
            negatives = group_df[group_df[args.label_col] == 0]

            num_positive_rows += len(positives)
            num_negative_rows += len(negatives)

            if len(positives) == 0 or len(negatives) == 0:
                num_skipped_groups += 1
                continue

            negative_indices_all = list(negatives.index)

            for _, pos_row in positives.iterrows():
                negative_indices = negative_indices_all.copy()

                if (
                    args.max_pairs_per_positive >= 0
                    and len(negative_indices) > args.max_pairs_per_positive
                ):
                    negative_indices = rng.sample(
                        negative_indices,
                        args.max_pairs_per_positive,
                    )

                for neg_idx in negative_indices:
                    neg_row = negatives.loc[neg_idx]

                    # Positive clause preferred over negative clause.
                    write_pair(
                        writer=writer,
                        group_cols=args.group_cols,
                        group_key=group_key,
                        features=args.features,
                        left_row=pos_row,
                        right_row=neg_row,
                        label=1,
                    )
                    num_pairs_written += 1

                    if args.include_reverse_pairs:
                        # Negative clause compared against positive clause.
                        write_pair(
                            writer=writer,
                            group_cols=args.group_cols,
                            group_key=group_key,
                            features=args.features,
                            left_row=neg_row,
                            right_row=pos_row,
                            label=0,
                        )
                        num_pairs_written += 1

    print(f"Loaded clause-level rows: {len(df)}")
    print(f"Groups seen: {num_groups}")
    print(f"Groups skipped: {num_skipped_groups}")
    print(f"Positive rows seen: {num_positive_rows}")
    print(f"Negative rows seen: {num_negative_rows}")
    print(f"Pairwise rows written: {num_pairs_written}")
    print(f"Saved pairwise dataset to: {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())