#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def proportional_group_counts(group_sizes: pd.Series, target_total: int) -> pd.Series:
    """
    Allocate exactly target_total samples across groups proportionally to group_sizes.
    Ensures:
      - no group gets more than its size
      - total allocated = target_total
    """
    if target_total <= 0:
        raise ValueError("target_total must be positive")

    total = int(group_sizes.sum())
    if target_total > total:
        raise ValueError(
            f"Requested sample size {target_total} exceeds dataset size {total}"
        )

    exact = group_sizes / total * target_total
    base = exact.astype(int)

    remainder = target_total - int(base.sum())
    fractional = (exact - base).sort_values(ascending=False)

    alloc = base.copy()
    for idx in fractional.index:
        if remainder == 0:
            break
        if alloc.loc[idx] < group_sizes.loc[idx]:
            alloc.loc[idx] += 1
            remainder -= 1

    overflow = alloc - group_sizes
    if (overflow > 0).any():
        for idx in overflow[overflow > 0].index:
            extra = int(overflow.loc[idx])
            alloc.loc[idx] -= extra
            remainder += extra

        spare = (group_sizes - alloc).sort_values(ascending=False)
        for idx in spare.index:
            if remainder == 0:
                break
            if alloc.loc[idx] < group_sizes.loc[idx]:
                alloc.loc[idx] += 1
                remainder -= 1

    if int(alloc.sum()) != target_total:
        raise RuntimeError(
            f"Internal error: allocated {int(alloc.sum())} instead of {target_total}"
        )

    return alloc


def maybe_add_difficulty_bin(
    df: pd.DataFrame,
    difficulty_col: str,
    runtime_col: str,
    difficulty_bins: List[float],
    difficulty_labels: List[str],
) -> pd.DataFrame:
    if difficulty_col in df.columns:
        return df

    if runtime_col not in df.columns:
        raise ValueError(
            f"'{difficulty_col}' not found and runtime column '{runtime_col}' not found."
        )

    df = df.copy()
    df[runtime_col] = pd.to_numeric(df[runtime_col], errors="coerce")
    if df[runtime_col].isna().any():
        bad = int(df[runtime_col].isna().sum())
        raise ValueError(
            f"Column '{runtime_col}' contains {bad} non-numeric or missing values."
        )

    if len(difficulty_labels) != len(difficulty_bins) + 1:
        raise ValueError(
            "--difficulty-labels must have exactly len(--difficulty-bins)+1 entries"
        )

    cut_bins = [-float("inf")] + difficulty_bins + [float("inf")]
    df[difficulty_col] = pd.cut(
        df[runtime_col],
        bins=cut_bins,
        labels=difficulty_labels,
        include_lowest=True,
        right=True,
    ).astype(str)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Take a stratified subsample of a CSV in dataset/results/ and save the "
            "sampled CSV back into dataset/results/."
        )
    )
    parser.add_argument(
        "csv_name",
        help="Input CSV inside dataset/results/, e.g. dataset_ALG_COL_GEO_LAT_NUM_train.csv",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of rows/problems to sample (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--stratify-cols",
        type=parse_csv_list,
        default=["domain", "solved"],
        help="Comma-separated columns to stratify on, e.g. domain,solved or domain,difficulty_bin",
    )
    parser.add_argument(
        "--difficulty-col",
        default="difficulty_bin",
        help="Difficulty column name to use/create if needed (default: difficulty_bin)",
    )
    parser.add_argument(
        "--runtime-col",
        default="runtime_seconds",
        help="Runtime column used to create difficulty bins if needed (default: runtime_seconds)",
    )
    parser.add_argument(
        "--difficulty-bins",
        type=parse_csv_list,
        default=["1.0", "5.0"],
        help="Comma-separated internal bin edges, e.g. 1.0,5.0",
    )
    parser.add_argument(
        "--difficulty-labels",
        type=parse_csv_list,
        default=["easy", "medium", "hard"],
        help="Comma-separated labels for difficulty bins",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV filename inside dataset/results/. If omitted, a name is generated.",
    )
    args = parser.parse_args()

    results_dir = Path("dataset/results").resolve()
    input_path = results_dir / args.csv_name
    if not input_path.is_file():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    stratify_cols = args.stratify_cols
    difficulty_bins = [float(x) for x in args.difficulty_bins]

    if args.difficulty_col in stratify_cols and args.difficulty_col not in df.columns:
        df = maybe_add_difficulty_bin(
            df=df,
            difficulty_col=args.difficulty_col,
            runtime_col=args.runtime_col,
            difficulty_bins=difficulty_bins,
            difficulty_labels=args.difficulty_labels,
        )

    missing = [c for c in stratify_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required stratification columns: {missing}")

    if args.sample_size > len(df):
        raise ValueError(
            f"Requested sample size {args.sample_size} exceeds number of rows {len(df)}"
        )

    grouped = df.groupby(stratify_cols, dropna=False)
    group_sizes = grouped.size()

    alloc = proportional_group_counts(group_sizes, args.sample_size)

    sampled_parts = []

    print("Per-group allocation:")
    for key, group_df in grouped:
        n_take = int(alloc.loc[key])
        sampled_group = (
            group_df.sample(n=n_take, random_state=args.seed)
            if n_take > 0
            else group_df.iloc[0:0]
        )
        sampled_parts.append(sampled_group)

        if not isinstance(key, tuple):
            key = (key,)
        key_desc = " ".join(f"{col}={val}" for col, val in zip(stratify_cols, key))
        print(f"{key_desc}: available={len(group_df)} sampled={n_take}")

    sampled_df = (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )

    if args.output_csv:
        output_name = args.output_csv
    else:
        stem = Path(args.csv_name).stem
        output_name = f"{stem}_sample_{args.sample_size}.csv"

    output_path = results_dir / output_name
    sampled_df.to_csv(output_path, index=False)

    print("\nOverall:")
    print(f"  input rows:   {len(df)}")
    print(f"  sampled rows: {len(sampled_df)}")

    print("\nSaved:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()