#!/usr/bin/env python3

import argparse
import os
import pandas as pd

BASE_DIR = "dataset/results"


def extract_domain(filename: str) -> str:
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]

    prefixes = [
        "dataset_pool_",
        "dataset_",
    ]

    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]

    return name


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple dataset CSV files from dataset/results/"
    )
    parser.add_argument(
        "csvs",
        nargs="+",
        help="CSV filenames (located in dataset/results/)"
    )
    args = parser.parse_args()

    dfs = []
    domains = []

    for csv_name in args.csvs:
        full_path = os.path.join(BASE_DIR, csv_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found")

        print(f"Loading {full_path}...")
        df = pd.read_csv(full_path)
        dfs.append(df)

        domain_part = extract_domain(csv_name)
        domains.extend(domain_part.split("_"))

    merged_df = pd.concat(dfs, ignore_index=True)

    domains_sorted = sorted(set(domains))
    domain_str = "_".join(domains_sorted)
    output_file = os.path.join(BASE_DIR, f"dataset_{domain_str}.csv")

    merged_df.to_csv(output_file, index=False)

    print(f"\nSaved merged dataset to: {output_file}")
    print(f"Total rows: {len(merged_df)}")


if __name__ == "__main__":
    main()