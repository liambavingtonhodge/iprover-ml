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
    if len(parts) >= 4 and parts[0] == "TPTPLibrary":
        trimmed = Path(*parts[2:])  # remove TPTPLibrary / TPTP-v9.2.1
        candidate = tptp_root / trimmed
        if candidate.is_file():
            return candidate

    # CSV path like: Problems/GEO/GEO027-2.p
    if len(parts) >= 2 and parts[0] == "Problems":
        candidate = tptp_root / rel_path
        if candidate.is_file():
            return candidate

    # CSV path like: GEO/GEO027-2.p
    candidate = tptp_root / "Problems" / rel_path
    if candidate.is_file():
        return candidate

    # CSV contains only basename like GEO027-2.p
    candidate = tptp_root / "Problems" / rel_path.name[:3].upper() / rel_path.name
    if candidate.is_file():
        return candidate

    # Last fallback
    candidate = tptp_root / rel_path
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Problem file not found for CSV entry: {problem_entry}")


def copy_problem_subset(
    csv_name: str,
    tptp_root: Path,
    results_dir: Path,
    problems_dir: Path,
    output_dir_name: str,
    problem_col: str,
    clear_output_dir: bool,
) -> None:
    csv_path = results_dir / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if problem_col not in df.columns:
        raise ValueError(f"'{problem_col}' column missing from {csv_path}")

    out_dir = problems_dir / output_dir_name

    if clear_output_dir and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for problem_entry in df[problem_col]:
        if pd.isna(problem_entry):
            skipped += 1
            continue

        source_path = resolve_problem_source_path(str(problem_entry), tptp_root)
        dest_path = out_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        copied += 1

    print(f"{output_dir_name}: copied {copied} problems to {out_dir}")
    if skipped:
        print(f"{output_dir_name}: skipped {skipped} empty/missing entries")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a sampled CSV from dataset/results/ and copy the listed TPTP "
            "problems into a named folder under dataset/problems/."
        )
    )
    parser.add_argument(
        "--tptp-root",
        required=True,
        help="Path to TPTP root (containing Problems/ and Axioms/) or directly to Problems/",
    )
    parser.add_argument(
        "csv_name",
        help="CSV filename inside dataset/results/, e.g. rl_train_20.csv",
    )
    parser.add_argument(
        "--output-dir-name",
        default="train_200",
        help="Output directory name inside dataset/problems/ (default: train_200)",
    )
    parser.add_argument(
        "--problem-col",
        default="problem",
        help="Problem column in CSV (default: problem)",
    )
    parser.add_argument(
        "--clear-output-dir",
        action="store_true",
        help="If set, delete the output directory before copying",
    )
    args = parser.parse_args()

    tptp_root = infer_tptp_root(Path(args.tptp_root))
    results_dir = Path("dataset/results").resolve()
    problems_dir = Path("dataset/problems").resolve()

    copy_problem_subset(
        csv_name=args.csv_name,
        tptp_root=tptp_root,
        results_dir=results_dir,
        problems_dir=problems_dir,
        output_dir_name=args.output_dir_name,
        problem_col=args.problem_col,
        clear_output_dir=args.clear_output_dir,
    )


if __name__ == "__main__":
    main()