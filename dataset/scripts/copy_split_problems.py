#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import pandas as pd


def infer_tptp_root(input_path: Path) -> Path:
    input_path = input_path.resolve()

    if (input_path / "Problems").is_dir() and (input_path / "Axioms").is_dir():
        return input_path

    if input_path.name == "Problems" and (input_path.parent / "Axioms").is_dir():
        return input_path.parent

    raise FileNotFoundError(
        f"Could not infer TPTP root from {input_path}. "
        f"Pass the TPTP root (containing Problems/ and Axioms/) "
        f"or the Problems/ directory."
    )


def resolve_problem_source_path(problem_entry: str, tptp_root: Path) -> Path:
    rel_path = Path(problem_entry)
    parts = rel_path.parts

    # CSV path like: TPTPLibrary/TPTP-v9.2.1/Problems/GEO/GEO027-2.p
    if len(parts) >= 3 and parts[0] == "TPTPLibrary":
        trimmed = Path(*parts[2:])  # remove TPTPLibrary / TPTP-v9.2.1
        candidate = tptp_root / trimmed
        if candidate.is_file():
            return candidate

    # CSV path like: Problems/GEO/GEO027-2.p
    if len(parts) >= 1 and parts[0] == "Problems":
        candidate = tptp_root / rel_path
        if candidate.is_file():
            return candidate

    # CSV path like: GEO/GEO027-2.p
    candidate = tptp_root / "Problems" / rel_path
    if candidate.is_file():
        return candidate

    # Last fallback
    candidate = tptp_root / rel_path
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Problem file not found for CSV entry: {problem_entry}")


def infer_split_name(csv_name: str) -> str:
    stem = Path(csv_name).stem.lower()

    if stem.endswith("_train"):
        return "train"
    if stem.endswith("_dev"):
        return "dev"
    if stem.endswith("_test"):
        return "test"

    raise ValueError(
        f"CSV filename must end with _train.csv, _dev.csv, or _test.csv: {csv_name}"
    )


def copy_split(csv_name: str, tptp_root: Path, results_dir: Path, problems_dir: Path) -> None:
    csv_path = results_dir / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "problem" not in df.columns:
        raise ValueError(f"'problem' column missing from {csv_path}")

    split_name = infer_split_name(csv_name)
    split_dir = problems_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for problem_entry in df["problem"]:
        if pd.isna(problem_entry):
            continue

        source_path = resolve_problem_source_path(str(problem_entry), tptp_root)
        dest_path = split_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        copied += 1

    print(f"{split_name}: copied {copied} problems to {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read split CSV files from dataset/results and copy the listed TPTP problems "
            "into dataset/problems/train, dataset/problems/dev, and dataset/problems/test."
        )
    )
    parser.add_argument(
        "--tptp_root",
        required=True,
        help="Path to TPTP root (containing Problems/ and Axioms/) or directly to Problems/",
    )
    parser.add_argument(
        "csvs",
        nargs="+",
        help="CSV filenames inside dataset/results/, e.g. dataset_ALG_COL_train.csv",
    )
    args = parser.parse_args()

    tptp_root = infer_tptp_root(Path(args.tptp_root))
    results_dir = Path("dataset/results").resolve()
    problems_dir = Path("dataset/problems").resolve()

    for csv_name in args.csvs:
        copy_split(csv_name, tptp_root, results_dir, problems_dir)


if __name__ == "__main__":
    main()