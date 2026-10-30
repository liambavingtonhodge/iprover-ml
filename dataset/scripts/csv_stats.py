#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def runtime_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return {
            "mean": None,
            "min": None,
            "25%": None,
            "50%": None,
            "75%": None,
            "max": None,
        }

    q = s.quantile([0.25, 0.50, 0.75])

    return {
        "mean": s.mean(),
        "min": s.min(),
        "25%": q.loc[0.25],
        "50%": q.loc[0.50],
        "75%": q.loc[0.75],
        "max": s.max(),
    }


def print_runtime_stats(df: pd.DataFrame) -> None:
    solved_df = df[df["solved"] == 1]
    stats = runtime_summary(solved_df["runtime_seconds"])

    print("Runtime (solved only):")
    if stats["mean"] is None:
        print("  No solved problems.")
        return

    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Min:  {stats['min']:.6f}")
    print(f"  25%:  {stats['25%']:.6f}")
    print(f"  50%:  {stats['50%']:.6f}")
    print(f"  75%:  {stats['75%']:.6f}")
    print(f"  Max:  {stats['max']:.6f}")


def print_status_counts(df: pd.DataFrame) -> None:
    print("No. of problems with each status:")
    counts = df["status"].fillna("NO_STATUS").value_counts(dropna=False).sort_index()
    for status, count in counts.items():
        print(f"  {status}: {count}")


def print_block(title: str, df: pd.DataFrame) -> None:
    total = len(df)
    solved = int((df["solved"] == 1).sum())
    unsolved = total - solved
    timeouts = int((df["timed_out"] == 1).sum())

    solve_rate = solved / total if total > 0 else 0

    print(title)
    print(f"Total no. problems: {total}")
    print(f"Total no. problems solved: {solved}")
    print(f"Total no. problems unsolved: {unsolved}")
    print(f"Solved/Total: {solved}/{total} ({solve_rate*100:.2f}%)")
    print(f"No. of timeouts: {timeouts}")
    print_status_counts(df)
    print_runtime_stats(df)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a results CSV from dataset/results and print overall and per-domain stats."
        )
    )
    parser.add_argument(
        "csv_name",
        help="Name of the CSV file inside dataset/results, e.g. database_pool_ALG_SET_LCL.csv",
    )
    args = parser.parse_args()

    csv_path = Path("dataset/results") / args.csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"domain", "status", "solved", "runtime_seconds", "timed_out"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    print_block("OVERALL", df)

    for domain in sorted(df["domain"].dropna().unique()):
        domain_df = df[df["domain"] == domain]
        print_block(f"DOMAIN: {domain}", domain_df)


if __name__ == "__main__":
    main()