#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List


RATING_RE = re.compile(r"%\s*Rating\s*:\s*([0-9.]+)")
STATUS_RE = re.compile(r"%\s*Status\s*:\s*([A-Za-z_]+)")


def parse_metadata(path: Path) -> Optional[Tuple[Optional[float], Optional[str]]]:
    rating = None
    status = None

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if rating is None:
                    m = RATING_RE.search(line)
                    if m:
                        rating = float(m.group(1))

                if status is None:
                    m = STATUS_RE.search(line)
                    if m:
                        status = m.group(1)

                if rating is not None and status is not None:
                    break
    except Exception:
        return None

    return rating, status


def infer_problems_root(input_path: Path) -> Path:
    input_path = input_path.resolve()

    if (input_path / "Problems").is_dir():
        return input_path / "Problems"

    if input_path.name == "Problems":
        return input_path

    raise FileNotFoundError(
        f"Could not find Problems/ from {input_path}. "
        f"Pass either the TPTP root or the Problems/ directory."
    )


def matches_restrictions(path: Path) -> bool:
    name = path.name

    if "+" not in name and "-" not in name:
        return False

    meta = parse_metadata(path)
    if meta is None:
        return False

    rating, status = meta
    if rating is None or status is None:
        return False

    if not (0.05 <= rating <= 0.9):
        return False

    if "+" in name and status != "Theorem":
        return False

    if "-" in name and status != "Unsatisfiable":
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count problem files matching restrictions for given domains."
    )
    parser.add_argument(
        "--tptp_root",
        required=True,
        help="Path to the TPTP root or directly to the Problems/ directory",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        required=False,  # 👈 now optional
        help="List of domains (space-separated), e.g. ALG SET LCL",
    )
    args = parser.parse_args()

    problems_root = infer_problems_root(Path(args.tptp_root))

    # 👇 NEW: if no domains given, use all folders in Problems/
    if args.domains:
        domains: List[str] = args.domains
    else:
        domains = sorted(
            [d.name for d in problems_root.iterdir() if d.is_dir()]
        )
        print(f"No domains specified. Using ALL domains: {' '.join(domains)}")

    total = 0
    total_ratings: List[float] = []

    for domain in domains:
        domain_dir = problems_root / domain
        count = 0
        ratings: List[float] = []

        if not domain_dir.exists():
            print(f"{domain}: 0 (missing domain folder)")
            continue

        for path in sorted(domain_dir.rglob("*.p")):
            if matches_restrictions(path):
                count += 1
                meta = parse_metadata(path)
                if meta is not None:
                    rating, _ = meta
                    if rating is not None:
                        ratings.append(rating)

        total += count
        total_ratings.extend(ratings)

        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            print(f"{domain}: {count}, average rating: {avg_rating:.4f}")
        else:
            print(f"{domain}: {count}, average rating: N/A")

    print(f"TOTAL: {total}")
    if total_ratings:
        overall_avg = sum(total_ratings) / len(total_ratings)
        print(f"OVERALL average rating: {overall_avg:.4f}")
    else:
        print("OVERALL average rating: N/A")


if __name__ == "__main__":
    main()