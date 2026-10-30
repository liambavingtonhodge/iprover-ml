#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RATING_RE = re.compile(r"%\s*Rating\s*:\s*([0-9.]+)")
STATUS_RE = re.compile(r"%\s*Status\s*:\s*([A-Za-z_]+)")
SZS_RE = re.compile(r"%\s*SZS status\s+([A-Za-z_]+)", re.IGNORECASE)

SOLVED_STATUSES = {"Theorem", "Unsatisfiable"}
VALID_PLUS_STATUS = "Theorem"
VALID_MINUS_STATUS = "Unsatisfiable"


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


def is_valid_problem(path: Path, min_rating: float, max_rating: float) -> bool:
    name = path.name

    if "+" not in name and "-" not in name:
        return False

    meta = parse_metadata(path)
    if meta is None:
        return False

    rating, status = meta
    if rating is None or status is None:
        return False

    if not (min_rating <= rating <= max_rating):
        return False

    if "+" in name and status != VALID_PLUS_STATUS:
        return False

    if "-" in name and status != VALID_MINUS_STATUS:
        return False

    return True


def collect_candidates(
    tptp_problems_root: Path,
    domains: List[str],
    min_rating: float,
    max_rating: float,
) -> List[Path]:
    candidates: List[Path] = []

    for domain in domains:
        domain_dir = tptp_problems_root / domain
        if not domain_dir.exists():
            print(f"Warning: missing domain folder: {domain_dir}")
            continue

        for path in sorted(domain_dir.rglob("*.p")):
            if is_valid_problem(path, min_rating, max_rating):
                candidates.append(path)

    return candidates


def infer_tptp_root(input_path: Path) -> Tuple[Path, Path]:
    input_path = input_path.resolve()

    if (input_path / "Problems").is_dir() and (input_path / "Axioms").is_dir():
        return input_path, input_path / "Problems"

    if input_path.name == "Problems" and (input_path.parent / "Axioms").is_dir():
        return input_path.parent, input_path

    raise FileNotFoundError(
        f"Could not infer TPTP root from {input_path}. "
        f"Pass either the TPTP root (containing Problems/ and Axioms/) "
        f"or directly the Problems/ directory."
    )


def make_csv_problem_name(problem_file: Path) -> str:
    parts = problem_file.parts

    if "TPTPLibrary" in parts:
        idx = parts.index("TPTPLibrary")
        return str(Path(*parts[idx:]))

    return str(problem_file)


def parse_optional_int_stat(output: str, stat_name: str) -> Optional[int]:
    pattern = rf"^{re.escape(stat_name)}:\s*(\d+|undef)\s*$"
    m = re.search(pattern, output, flags=re.MULTILINE | re.IGNORECASE)
    if not m:
        return None

    value = m.group(1).strip().lower()
    if value == "undef":
        return None

    return int(value)


def parse_runtime_seconds(output: str, measured_wallclock: float) -> float:
    patterns = [
        r"total_time:\s*([0-9.]+)",
        r"^\s*real\s+([0-9.]+)\s*$",
        r"^\s*Real time\s*:\s*([0-9.]+)\s*$",
        r"^\s*CPU time\s*:\s*([0-9.]+)\s*$",
        r"^\s*Total time\s*:\s*([0-9.]+)\s*$",
    ]

    for pat in patterns:
        m = re.search(pat, output, flags=re.MULTILINE | re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    return measured_wallclock


def parse_szs_status(output: str) -> Optional[str]:
    m = SZS_RE.search(output)
    if m:
        return m.group(1)
    return None


def saw_real_timeout_message(output: str) -> bool:
    return "Time_out_real" in output or "Time Out Real" in output


def classify_result(
    szs_status: Optional[str],
    output: str,
    measured_wallclock: float,
    timeout: float,
) -> Tuple[int, int, int]:
    if szs_status in SOLVED_STATUSES:
        return 1, 1, 0

    timed_out = 0
    if measured_wallclock >= timeout - 0.2:
        timed_out = 1
    elif saw_real_timeout_message(output):
        timed_out = 1

    return 0, 0, timed_out


def parse_iprover_output(
    output: str,
    measured_wallclock: float,
    timeout: float,
) -> Dict[str, Any]:
    szs_status = parse_szs_status(output)
    solved, proof_found, timed_out = classify_result(
        szs_status=szs_status,
        output=output,
        measured_wallclock=measured_wallclock,
        timeout=timeout,
    )

    loop_iterations = None
    for loop_name in ["sup_num_of_loops", "inst_num_of_loops", "res_num_of_loops"]:
        loop_val = parse_optional_int_stat(output, loop_name)
        if loop_val is not None:
            loop_iterations = loop_val
            break

    return {
        "szs_status": szs_status,
        "solved": solved,
        "runtime_seconds": parse_runtime_seconds(output, measured_wallclock),
        "instantiations": parse_optional_int_stat(output, "inst_num_of_clauses"),
        "generated_clauses": parse_optional_int_stat(output, "sup_num_of_clauses"),
        "processed_clauses": parse_optional_int_stat(output, "sup_num_in_active"),
        "loop_iterations": loop_iterations,
        "proof_found": proof_found,
        "timed_out": timed_out,
    }


def run_one_problem(
    problem_file: Path,
    domain: str,
    iprover_bin: Path,
    timeout: int,
    tptp_root: Path,
) -> Dict[str, Any]:
    cmd = [
        str(iprover_bin),
        "--proof_out",
        "false",
        "--stats_out",
        "all",
        "--time_out_real",
        str(timeout),
        str(problem_file.resolve()),
    ]

    env = dict(os.environ)
    env["TPTP"] = str(tptp_root.resolve())

    start = time.time()

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
            env=env,
            cwd=str(tptp_root.resolve()),
        )

        wallclock = time.time() - start
        output = completed.stdout + "\n" + completed.stderr
        parsed = parse_iprover_output(
            output,
            measured_wallclock=wallclock,
            timeout=timeout,
        )

        final_status = parsed["szs_status"]
        if final_status is None:
            final_status = "TIMEOUT" if parsed["timed_out"] else "NO_STATUS"

        return {
            "problem": make_csv_problem_name(problem_file),
            "domain": domain,
            "status": final_status,
            "solved": parsed["solved"],
            "runtime_seconds": round(parsed["runtime_seconds"], 6),
            "instantiations": parsed["instantiations"],
            "generated_clauses": parsed["generated_clauses"],
            "processed_clauses": parsed["processed_clauses"],
            "loop_iterations": parsed["loop_iterations"],
            "proof_found": parsed["proof_found"],
            "timed_out": parsed["timed_out"],
            "return_code": completed.returncode,
            "agent_error": "",
        }

    except subprocess.TimeoutExpired:
        wallclock = time.time() - start
        return {
            "problem": make_csv_problem_name(problem_file),
            "domain": domain,
            "status": "TIMEOUT",
            "solved": 0,
            "runtime_seconds": round(wallclock, 6),
            "instantiations": None,
            "generated_clauses": None,
            "processed_clauses": None,
            "loop_iterations": None,
            "proof_found": 0,
            "timed_out": 1,
            "return_code": None,
            "agent_error": "",
        }

    except Exception as e:
        wallclock = time.time() - start
        return {
            "problem": make_csv_problem_name(problem_file),
            "domain": domain,
            "status": "ERROR",
            "solved": 0,
            "runtime_seconds": round(wallclock, 6),
            "instantiations": None,
            "generated_clauses": None,
            "processed_clauses": None,
            "loop_iterations": None,
            "proof_found": 0,
            "timed_out": 0,
            "return_code": None,
            "agent_error": str(e),
        }


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "problem",
        "domain",
        "status",
        "solved",
        "runtime_seconds",
        "instantiations",
        "generated_clauses",
        "processed_clauses",
        "loop_iterations",
        "proof_found",
        "timed_out",
        "return_code",
        "agent_error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan TPTP domains, filter by status/rating, "
            "run standard iProver, and write results to CSV."
        )
    )
    parser.add_argument(
        "--tptp_root",
        required=True,
        help="Path to TPTP root (with Problems/ and Axioms/) or directly to Problems/",
    )
    parser.add_argument(
        "--output_csv_dir",
        required=True,
        help="Directory where the output CSV will be written",
    )
    parser.add_argument(
        "--iprover_bin",
        required=True,
        help="Path to iproveropt",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        required=True,
        help="Domains to use, separated by spaces, e.g. ALG SET LCL",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--min_rating",
        type=float,
        default=0.05,
        help="Minimum TPTP rating (default: 0.05)",
    )
    parser.add_argument(
        "--max_rating",
        type=float,
        default=0.9,
        help="Maximum TPTP rating (default: 0.9)",
    )
    args = parser.parse_args()

    input_path = Path(args.tptp_root)
    output_csv_dir = Path(args.output_csv_dir)
    iprover_bin = Path(args.iprover_bin).resolve()

    if not iprover_bin.is_file():
        raise FileNotFoundError(f"iprover binary not found: {iprover_bin}")

    tptp_root, tptp_problems_root = infer_tptp_root(input_path)

    print(f"TPTP root: {tptp_root}")
    print(f"TPTP Problems root: {tptp_problems_root}")
    print(f"Using TPTP environment variable: {tptp_root.resolve()}")
    print(f"Domains: {' '.join(args.domains)}")

    candidates = collect_candidates(
        tptp_problems_root=tptp_problems_root,
        domains=args.domains,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
    )

    print(f"Found {len(candidates)} candidate problems.")

    rows: List[Dict[str, Any]] = []

    for i, problem in enumerate(candidates, start=1):
        domain = problem.relative_to(tptp_problems_root).parts[0]
        print(f"[{i}/{len(candidates)}] Running {problem}")

        row = run_one_problem(
            problem_file=problem,
            domain=domain,
            iprover_bin=iprover_bin,
            timeout=args.timeout,
            tptp_root=tptp_root,
        )
        rows.append(row)

        print(
            f"    status={row['status']} "
            f"solved={row['solved']} "
            f"runtime={row['runtime_seconds']} "
            f"timed_out={row['timed_out']}"
        )

    output_csv_path = output_csv_dir / f"dataset_{'_'.join(sorted(args.domains))}.csv"
    write_csv(rows, output_csv_path)

    print("\nFinished.")
    print(f"CSV written to: {output_csv_path}")


if __name__ == "__main__":
    main()