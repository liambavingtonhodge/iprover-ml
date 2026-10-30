#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def split_group(df: pd.DataFrame, train_frac: float, dev_frac: float):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    n_train = int(train_frac * n)
    n_dev = int(dev_frac * n)

    train = df.iloc[:n_train]
    dev = df.iloc[n_train:n_train + n_dev]
    test = df.iloc[n_train + n_dev:]

    return train, dev, test


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/dev/test, stratified by both domain and solved status."
    )
    parser.add_argument(
        "csv_name",
        help="CSV file inside dataset/results/, e.g. dataset_ALG_COL_GEO_LAT_NUM.csv",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="Training fraction (default: 0.7)",
    )
    parser.add_argument(
        "--dev_frac",
        type=float,
        default=0.15,
        help="Development fraction (default: 0.15)",
    )
    args = parser.parse_args()

    if args.train_frac <= 0 or args.dev_frac <= 0 or args.train_frac + args.dev_frac >= 1:
        raise ValueError("Require 0 < train_frac, 0 < dev_frac, and train_frac + dev_frac < 1.")

    input_path = Path("dataset/results") / args.csv_name
    if not input_path.is_file():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"domain", "solved"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    train_parts = []
    dev_parts = []
    test_parts = []

    grouped = df.groupby(["domain", "solved"], dropna=False)

    for (domain, solved), group_df in grouped:
        train_df, dev_df, test_df = split_group(
            group_df,
            train_frac=args.train_frac,
            dev_frac=args.dev_frac,
        )

        train_parts.append(train_df)
        dev_parts.append(dev_df)
        test_parts.append(test_df)

        print(
            f"domain={domain} solved={solved}: "
            f"total={len(group_df)} train={len(train_df)} dev={len(dev_df)} test={len(test_df)}"
        )

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    dev_df = pd.concat(dev_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    base_name = args.csv_name.replace(".csv", "")

    train_path = Path("dataset/results") / f"{base_name}_train.csv"
    dev_path = Path("dataset/results") / f"{base_name}_dev.csv"
    test_path = Path("dataset/results") / f"{base_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nOverall split sizes:")
    print(f"  train: {len(train_df)}")
    print(f"  dev:   {len(dev_df)}")
    print(f"  test:  {len(test_df)}")

    print("\nSaved:")
    print(f"  {train_path}")
    print(f"  {dev_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()