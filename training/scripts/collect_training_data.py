#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


SOLVED_STATUSES = {"Theorem", "Unsatisfiable"}

CLAUSE_FIELDNAMES = [
    "problem",
    "decision_index",
    "candidate_clause_id",
    "selected_clause_id",
    "selected",
    "num_literals",
    "num_symbols",
    "max_depth",
    "num_variables",
    "age",
    "shared_conj_symbols",
    "is_ground",
    "has_equality",
    "clause_weight",
    "clause_text",
]


def infer_tptp_root(input_path: Path) -> Path:
    input_path = input_path.resolve()

    if (input_path / "Problems").is_dir() and (input_path / "Axioms").is_dir():
        return input_path

    if input_path.name == "Problems" and (input_path.parent / "Axioms").is_dir():
        return input_path.parent

    raise FileNotFoundError(
        f"Could not infer TPTP root from {input_path}. "
        f"Pass either the TPTP root (containing Problems/ and Axioms/) "
        f"or directly the Problems/ directory."
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def safe_kill(proc: Optional[subprocess.Popen], grace: float = 3.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=grace)
        except Exception:
            pass
    except Exception:
        pass


def collect_process_output(proc: Optional[subprocess.Popen], grace: float = 3.0) -> Dict[str, str]:
    if proc is None:
        return {"stdout": "", "stderr": ""}

    try:
        if proc.poll() is None:
            proc.terminate()
        stdout, stderr = proc.communicate(timeout=grace)
        return {
            "stdout": stdout or "",
            "stderr": stderr or "",
        }
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=grace)
            return {
                "stdout": stdout or "",
                "stderr": stderr or "",
            }
        except Exception:
            return {"stdout": "", "stderr": ""}
    except Exception:
        return {"stdout": "", "stderr": ""}


def infer_domain(problem_file: Path, problems_root: Path) -> str:
    name = problem_file.name.upper()
    m = re.match(r"^([A-Z]+)\d", name)
    if m:
        return m.group(1)

    m = re.match(r"^([A-Z]+)", name)
    if m:
        return m.group(1)

    return "UNKNOWN"


def make_problem_name(problem_file: Path, domain: str) -> str:
    return str(Path("TPTPLibrary") / "TPTP-v9.2.1" / "Problems" / domain / problem_file.name)


def build_interactive_iprover_command(
    iprover_bin: Path,
    host: str,
    port: int,
    timeout: int,
    extra_args: List[str],
    problem_file: Path,
    proof_out: bool = False,
    collection_mode: str = "external_score",
) -> List[str]:
    cmd = [
        str(iprover_bin),
        "--interactive_mode", "true",
        "--external_ip_address", host,
        "--external_port", str(port),
    ]

    if collection_mode == "external_score":
        cmd.extend([
            "--schedule", "none",
            "--preprocessing_flag", "false",
            "--instantiation_flag", "true",
            "--superposition_flag", "true",
            "--resolution_flag", "false",
            "--inst_passive_queue_type", "priority_queues",
            "--inst_passive_queues_freq", "[1]",
            "--inst_passive_queues", "[[-external_score]]",
        ])

    elif collection_mode == "age":
        cmd.extend([
            "--schedule", "none",
            "--preprocessing_flag", "false",
            "--instantiation_flag", "true",
            "--superposition_flag", "true",
            "--resolution_flag", "false",
            "--inst_passive_queue_type", "priority_queues",
            "--inst_passive_queues_freq", "[1]",
            "--inst_passive_queues", "[[+age]]",
        ])

    elif collection_mode == "default":
        # leave iProver defaults
        pass

    else:
        raise ValueError(f"Unsupported collection_mode: {collection_mode}")

    cmd.extend([
        "--proof_out", "true" if proof_out else "false",
        "--stats_out", "all",
        "--time_out_real", str(timeout),
    ])

    cmd.extend(extra_args)
    cmd.append(str(problem_file.resolve()))
    return cmd
    

def parse_szs_status(output: str) -> Optional[str]:
    m = re.search(r"%\s*SZS status\s+([A-Za-z_]+)", output)
    if m:
        return m.group(1)
    return None


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


def parse_optional_int_stat(output: str, stat_name: str) -> Optional[int]:
    pattern = rf"^{re.escape(stat_name)}:\s*(\d+|undef)\s*$"
    m = re.search(pattern, output, flags=re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    value = m.group(1).strip().lower()
    if value == "undef":
        return None
    return int(value)


def saw_real_timeout_message(output: str) -> bool:
    return "Time_out_real" in output or "Time Out Real" in output


def finalise_status(szs_status: Optional[str], output: str, wallclock: float, timeout: float) -> str:
    if szs_status is not None:
        return szs_status
    if saw_real_timeout_message(output) or wallclock >= timeout - 0.2:
        return "TIMEOUT"
    return "NO_STATUS"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def parse_collector_rows(stdout_text: str) -> List[Dict[str, Any]]:
    stdout_text = (stdout_text or "").strip()
    if not stdout_text:
        return []

    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError:
        return []

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []

    clean_rows: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            clean_rows.append(row)
    return clean_rows


def append_rows_to_master(rows: List[Dict[str, Any]], master_csv: Path) -> int:
    if not rows:
        return 0

    master_csv.parent.mkdir(parents=True, exist_ok=True)
    need_header = not master_csv.exists() or master_csv.stat().st_size == 0

    with master_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLAUSE_FIELDNAMES)
        if need_header:
            writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def run_one_problem(
    idx: int,
    problem_file: Path,
    problems_root: Path,
    tptp_root: Path,
    iprover_bin: Path,
    collector_agent: Path,
    clause_csv: Path,
    timeout: int,
    iprover_extra_args: List[str],
    logs_dir: Optional[Path],
    max_negatives_per_positive: int,
    random_seed: int,
    successful_runs_only: bool,
    collector_debug_log: str,
    collection_mode: str,
) -> Dict[str, Any]:
    domain = infer_domain(problem_file, problems_root)
    problem_name = make_problem_name(problem_file, domain)

    host = "127.0.0.1"
    port = find_free_port()

    agent_cmd = [
        sys.executable,
        str(collector_agent),
        "--host", host,
        "--port", str(port),
        "--problem-name", problem_name,
        "--problem-file", str(problem_file.resolve()),
        "--max-negatives-per-positive", str(max_negatives_per_positive),
        "--random-seed", str(random_seed),
    ]

    if collector_debug_log:
        agent_cmd.extend([
            "--debug-log",
            str(Path(collector_debug_log).resolve()),
        ])

    env = dict(os.environ)
    env["TPTP"] = str(tptp_root.resolve())

    agent_proc = None
    start = time.time()

    try:
        agent_proc = subprocess.Popen(
            agent_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root()),
        )

        time.sleep(1.0)

        if agent_proc.poll() is not None:
            proc_output = collect_process_output(agent_proc)
            return {
                "problem": problem_name,
                "domain": domain,
                "status": "AGENT_START_FAILED",
                "runtime_seconds": 0.0,
                "instantiations": None,
                "generated_clauses": None,
                "processed_clauses": None,
                "return_code": None,
                "agent_error": (
                    "Collector agent exited immediately. "
                    + (proc_output["stderr"][:500] if proc_output["stderr"] else "")
                ).strip(),
                "rows_total": 0,
                "rows_kept": 0,
            }

        enable_proof_out = collector_agent.name == "collect_clause_data_proof.py"

        iprover_cmd = build_interactive_iprover_command(
            iprover_bin=iprover_bin,
            host=host,
            port=port,
            timeout=timeout,
            extra_args=iprover_extra_args,
            problem_file=problem_file,
            proof_out=enable_proof_out,
            collection_mode=collection_mode,
        )

        completed = subprocess.run(
            iprover_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 2,
            env=env,
            cwd=str(tptp_root.resolve()),
        )

        wallclock = time.time() - start
        output = completed.stdout + "\n" + completed.stderr
        szs_status = parse_szs_status(output)
        status = finalise_status(szs_status, output, wallclock, timeout)

        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / f"{problem_file.stem}.log").write_text(
                output, encoding="utf-8", errors="replace"
            )

        proc_output = collect_process_output(agent_proc)
        collector_stdout = proc_output["stdout"]
        collector_stderr = proc_output["stderr"]
        collected_rows = parse_collector_rows(collector_stdout)
        rows_total = len(collected_rows)

        rows_kept = 0
        keep_rows = (not successful_runs_only) or (status in SOLVED_STATUSES)
        if keep_rows:
            rows_kept = append_rows_to_master(collected_rows, clause_csv)

        return {
            "problem": problem_name,
            "domain": domain,
            "status": status,
            "runtime_seconds": round(parse_runtime_seconds(output, wallclock), 6),
            "instantiations": parse_optional_int_stat(output, "inst_num_of_clauses"),
            "generated_clauses": parse_optional_int_stat(output, "sup_num_of_clauses"),
            "processed_clauses": parse_optional_int_stat(output, "sup_num_in_active"),
            "return_code": completed.returncode,
            "agent_error": collector_stderr[:500].strip(),
            "rows_total": rows_total,
            "rows_kept": rows_kept,
        }

    except subprocess.TimeoutExpired:
        wallclock = time.time() - start

        proc_output = collect_process_output(agent_proc)
        collector_stdout = proc_output["stdout"]
        collector_stderr = proc_output["stderr"]
        collected_rows = parse_collector_rows(collector_stdout)
        rows_total = len(collected_rows)

        rows_kept = 0
        if not successful_runs_only:
            rows_kept = append_rows_to_master(collected_rows, clause_csv)

        return {
            "problem": problem_name,
            "domain": domain,
            "status": "TIMEOUT",
            "runtime_seconds": round(wallclock, 6),
            "instantiations": None,
            "generated_clauses": None,
            "processed_clauses": None,
            "return_code": None,
            "agent_error": collector_stderr[:500].strip(),
            "rows_total": rows_total,
            "rows_kept": rows_kept,
        }

    except Exception as e:
        wallclock = time.time() - start

        proc_output = collect_process_output(agent_proc)
        collector_stdout = proc_output["stdout"]
        collector_stderr = proc_output["stderr"]
        collected_rows = parse_collector_rows(collector_stdout)
        rows_total = len(collected_rows)

        return {
            "problem": problem_name,
            "domain": domain,
            "status": "ERROR",
            "runtime_seconds": round(wallclock, 6),
            "instantiations": None,
            "generated_clauses": None,
            "processed_clauses": None,
            "return_code": None,
            "agent_error": (str(e) + (" | " + collector_stderr[:500] if collector_stderr else "")).strip(),
            "rows_total": rows_total,
            "rows_kept": 0,
        }

    finally:
        safe_kill(agent_proc)


def write_summary_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "domain",
        "status",
        "runtime_seconds",
        "instantiations",
        "generated_clauses",
        "processed_clauses",
        "return_code",
        "agent_error",
        "rows_total",
        "rows_kept",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stats_json(stats: Dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect clause-level training data using interactive iProver runs."
    )
    parser.add_argument("--problems", required=True)
    parser.add_argument("--tptp_root", default="../TPTPLibrary/TPTP-v9.2.1")
    parser.add_argument("--iprover-bin", default="../iproveropt")
    parser.add_argument("--collector-agent", default="agents/collect_clause_data_given.py")
    parser.add_argument("--clause-csv", required=True)
    parser.add_argument("--summary-csv", default="training/results/collect_training_summary.csv")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--iprover-extra-args", default="")
    parser.add_argument("--logs-dir", default="")
    parser.add_argument("--collector-debug-log", default="")
    parser.add_argument(
        "--collection-mode",
        choices=["external_score", "age", "default"],
        default="external_score",
        help=(
            "Collection regime: "
            "external_score = interactive with [[-external_score]] and agent returning 0.0; "
            "age = interactive with [[+age]]; "
            "default = interactive with standard iProver defaults."
        ),
    )
    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=10,
        help="Keep at most this many negative rows per positive decision row (default: 10). Use -1 to keep all.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for negative sampling (default: 42).",
    )
    parser.add_argument(
        "--successful-runs-only",
        action="store_true",
        help="If set, only append rows from problems whose final SZS status is Theorem or Unsatisfiable.",
    )
    args = parser.parse_args()

    overall_start = time.time()

    problems_dir = Path(args.problems).resolve()
    if not problems_dir.is_dir():
        print(f"Problem folder does not exist or is not a directory: {problems_dir}", file=sys.stderr)
        return 1

    tptp_root = infer_tptp_root(Path(args.tptp_root))
    iprover_bin = Path(args.iprover_bin).resolve()
    if not iprover_bin.is_file():
        print(f"iProver binary not found: {iprover_bin}", file=sys.stderr)
        return 1

    collector_agent = Path(args.collector_agent)
    if not collector_agent.is_absolute():
        collector_agent = (project_root() / args.collector_agent).resolve()
    if not collector_agent.is_file():
        print(f"Collector agent not found: {collector_agent}", file=sys.stderr)
        return 1

    clause_csv = Path(args.clause_csv)
    if not clause_csv.is_absolute():
        clause_csv = (project_root() / args.clause_csv).resolve()

    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = (project_root() / args.summary_csv).resolve()

    if clause_csv.exists():
        clause_csv.unlink()
    if summary_csv.exists():
        summary_csv.unlink()

    logs_dir = None
    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
        if not logs_dir.is_absolute():
            logs_dir = (project_root() / args.logs_dir).resolve()

    iprover_extra_args = shlex.split(args.iprover_extra_args)

    problem_files = sorted(problems_dir.rglob("*.p"))
    if not problem_files:
        print(f"No .p files found under: {problems_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(problem_files)} problem files.")
    print(f"TPTP root: {tptp_root}")
    print(f"iProver binary: {iprover_bin}")
    print(f"Collector agent: {collector_agent}")
    print(f"Clause CSV: {clause_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Successful runs only: {args.successful_runs_only}")
    print(f"Collection mode: {args.collection_mode}")

    rows: List[Dict[str, Any]] = []

    for i, problem in enumerate(problem_files, start=1):
        print(f"[{i}/{len(problem_files)}] Collecting from {problem}")
        row = run_one_problem(
            idx=i,
            problem_file=problem,
            problems_root=problems_dir,
            tptp_root=tptp_root,
            iprover_bin=iprover_bin,
            collector_agent=collector_agent,
            clause_csv=clause_csv,
            timeout=args.timeout,
            iprover_extra_args=iprover_extra_args,
            logs_dir=logs_dir,
            max_negatives_per_positive=args.max_negatives_per_positive,
            random_seed=args.random_seed,
            successful_runs_only=args.successful_runs_only,
            collector_debug_log=args.collector_debug_log,
            collection_mode=args.collection_mode,
        )
        rows.append(row)

        print(
            f"    status={row['status']} "
            f"runtime={row['runtime_seconds']} "
            f"inst={row.get('instantiations')} "
            f"rows_total={row.get('rows_total', 0)} "
            f"rows_kept={row.get('rows_kept', 0)}"
        )

    write_summary_csv(rows, summary_csv)

    total_time = time.time() - overall_start
    num_problems = len(problem_files)
    total_rows = sum(int(r.get("rows_kept", 0) or 0) for r in rows)
    solved_count = sum(1 for r in rows if r.get("status") in SOLVED_STATUSES)
    timeout_count = sum(1 for r in rows if r.get("status") == "TIMEOUT")
    zero_row_solved_count = sum(
        1
        for r in rows
        if r.get("status") in SOLVED_STATUSES and int(r.get("rows_kept", 0) or 0) == 0
    )

    stats = {
        "num_problems": num_problems,
        "solved_problems": solved_count,
        "timeout_problems": timeout_count,
        "zero_row_solved_problems": zero_row_solved_count,
        "successful_runs_only": bool(args.successful_runs_only),
        "collection_mode": args.collection_mode,
        "total_time_seconds": round(total_time, 6),
        "avg_time_per_problem_seconds": round(total_time / num_problems, 6) if num_problems else 0.0,
        "total_rows_collected": total_rows,
        "rows_per_second": round(total_rows / total_time, 6) if total_time > 0 else 0.0,
        "clause_csv": str(clause_csv),
        "summary_csv": str(summary_csv),
    }

    stats_json = summary_csv.with_suffix(".stats.json")
    write_stats_json(stats, stats_json)

    print("\nFinished.")
    print(f"Clause-level CSV: {clause_csv}")
    print(f"Problem-level summary CSV: {summary_csv}")
    print(f"Stats JSON: {stats_json}")
    print(f"Total collection time: {total_time:.2f} seconds")
    print(f"Average per problem: {total_time / num_problems:.2f} seconds" if num_problems else "Average per problem: 0.00 seconds")
    print(f"Total rows collected: {total_rows}")
    print(f"Zero-row solved problems: {zero_row_solved_count}")
    print(f"Rows per second: {total_rows / total_time:.2f}" if total_time > 0 else "Rows per second: 0.00")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())