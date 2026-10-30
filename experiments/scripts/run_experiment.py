#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


SOLVED_STATUSES = {"Theorem", "Unsatisfiable"}
DOMAIN_RE = re.compile(r"^([A-Z]{3})[0-9].*\.p$", re.IGNORECASE)


def infer_tptp_root(input_path: Path) -> Path:
    input_path = input_path.resolve()

    if (input_path / "Problems").is_dir() and (input_path / "Axioms").is_dir():
        return input_path

    if input_path.name == "Problems" and (input_path.parent / "Axioms").is_dir():
        return input_path.parent

    raise FileNotFoundError(
        f"Could not infer TPTP root from {input_path}. "
        f"Pass either the TPTP root containing Problems/ and Axioms/, "
        f"or directly the Problems/ directory."
    )


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_agent_ready(
    proc: subprocess.Popen,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> tuple[bool, str]:
    deadline = time.time() + timeout
    stderr_chunks: List[str] = []

    while time.time() < deadline:
        if proc.poll() is not None:
            try:
                remaining = proc.stderr.read() if proc.stderr is not None else ""
            except Exception:
                remaining = ""
            if remaining:
                stderr_chunks.append(remaining)
            return False, "".join(stderr_chunks)

        line = ""
        try:
            if proc.stderr is not None:
                line = proc.stderr.readline()
        except Exception:
            line = ""

        if line:
            stderr_chunks.append(line)
            if "EA listening on" in line:
                return True, "".join(stderr_chunks)

        time.sleep(poll_interval)

    return False, "".join(stderr_chunks)


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


def collect_agent_output(proc: Optional[subprocess.Popen], grace: float = 3.0) -> tuple[str, str]:
    if proc is None:
        return "", ""

    try:
        if proc.poll() is None:
            proc.terminate()
        stdout_text, stderr_text = proc.communicate(timeout=grace)
        return stdout_text or "", stderr_text or ""
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            stdout_text, stderr_text = proc.communicate(timeout=grace)
            return stdout_text or "", stderr_text or ""
        except Exception:
            return "", ""
    except Exception:
        return "", ""


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
    m = re.search(r"%\s*SZS status\s+([A-Za-z_]+)", output)
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
) -> tuple[int, int, int]:
    if szs_status in SOLVED_STATUSES:
        return 1, 1, 0

    timed_out = 0
    if measured_wallclock >= timeout - 0.2:
        timed_out = 1
    elif saw_real_timeout_message(output):
        timed_out = 1

    return 0, 0, timed_out


def finalise_status(
    szs_status: Optional[str],
    output: str,
    measured_wallclock: float,
    timeout: float,
) -> str:
    if szs_status is not None:
        return szs_status
    if saw_real_timeout_message(output) or measured_wallclock >= timeout - 0.2:
        return "TIMEOUT"
    return "NO_STATUS"


def parse_iprover_output(output: str, measured_wallclock: float, timeout: float) -> Dict[str, Any]:
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
        "status": finalise_status(szs_status, output, measured_wallclock, timeout),
        "solved": solved,
        "runtime_seconds": parse_runtime_seconds(output, measured_wallclock),
        "instantiations": parse_optional_int_stat(output, "inst_num_of_clauses"),
        "generated_clauses": parse_optional_int_stat(output, "sup_num_of_clauses"),
        "processed_clauses": parse_optional_int_stat(output, "sup_num_in_active"),
        "loop_iterations": loop_iterations,
        "proof_found": proof_found,
        "timed_out": timed_out,
    }


def infer_domain(problem_file: Path, problems_root: Path) -> str:
    try:
        rel = problem_file.relative_to(problems_root)
        if len(rel.parts) >= 2:
            candidate = rel.parts[0].upper()
            if re.fullmatch(r"[A-Z]{3}", candidate):
                return candidate
    except Exception:
        pass

    m = DOMAIN_RE.match(problem_file.name)
    if m:
        return m.group(1).upper()

    return problem_file.parent.name.upper()


def build_standard_iprover_command(
    iprover_bin: str,
    timeout: int,
    extra_args: List[str],
    problem_file: Path,
) -> List[str]:
    cmd = [
        iprover_bin,
        "--proof_out", "false",
        "--stats_out", "all",
        "--time_out_real", str(timeout),
    ]
    cmd.extend(extra_args)
    cmd.append(str(problem_file.resolve()))
    return cmd


def build_interactive_iprover_command(
    iprover_bin: str,
    host: str,
    port: int,
    timeout: int,
    extra_args: List[str],
    problem_file: Path,
    inst_passive_queues: str,
    inst_passive_queues_freq: str,
) -> List[str]:
    cmd = [
        iprover_bin,
        "--interactive_mode", "true",
        "--external_ip_address", host,
        "--external_port", str(port),
        "--schedule", "none",
        "--preprocessing_flag", "false",
        "--instantiation_flag", "true",
        "--superposition_flag", "true",
        "--resolution_flag", "false",
        "--inst_learning_loop_flag", "false",
        "--inst_passive_queue_type", "priority_queues",
        "--inst_passive_queues_freq", inst_passive_queues_freq,
        "--inst_passive_queues", inst_passive_queues,
        "--proof_out", "false",
        "--stats_out", "all",
        "--time_out_real", str(timeout),
    ]
    cmd.extend(extra_args)
    cmd.append(str(problem_file.resolve()))
    return cmd


def make_csv_problem_name(problem_file: Path, domain: str) -> str:
    return str(Path("TPTPLibrary") / "TPTP-v9.2.1" / "Problems" / domain / problem_file.name)


def resolve_output_path(path_str: str, default_base: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (default_base / path).resolve()


def run_one_problem(
    problem_file: Path,
    problems_root: Path,
    tptp_root: Path,
    agent_cmd_template: Optional[str],
    iprover_bin: str,
    iprover_extra_args: List[str],
    timeout: int,
    logs_dir: Optional[Path],
    inst_passive_queues: str,
    inst_passive_queues_freq: str,
) -> Dict[str, Any]:
    domain = infer_domain(problem_file, problems_root)
    start = time.time()
    agent_proc = None

    env = dict(os.environ)
    env["TPTP"] = str(tptp_root.resolve())

    try:
        if agent_cmd_template:
            host = "127.0.0.1"
            port = find_free_port()

            agent_cmd = agent_cmd_template.format(host=host, port=port)
            agent_args = shlex.split(agent_cmd)

            agent_proc = subprocess.Popen(
                agent_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(project_root()),
            )

            ready, startup_stderr = wait_for_agent_ready(agent_proc, timeout=5.0)

            if not ready:
                agent_stdout, agent_stderr = collect_agent_output(agent_proc)
                return {
                    "problem": make_csv_problem_name(problem_file, domain),
                    "domain": domain,
                    "status": "AGENT_START_FAILED",
                    "solved": 0,
                    "runtime_seconds": 0.0,
                    "instantiations": None,
                    "generated_clauses": None,
                    "processed_clauses": None,
                    "loop_iterations": None,
                    "proof_found": 0,
                    "timed_out": 0,
                    "return_code": None,
                    "agent_error": (
                        "Agent did not report readiness within timeout.\n"
                        + startup_stderr + "\n" + agent_stdout + "\n" + agent_stderr
                    ).strip(),
                }

            iprover_cmd = build_interactive_iprover_command(
                iprover_bin=iprover_bin,
                host=host,
                port=port,
                timeout=timeout,
                extra_args=iprover_extra_args,
                problem_file=problem_file,
                inst_passive_queues=inst_passive_queues,
                inst_passive_queues_freq=inst_passive_queues_freq,
            )
        else:
            iprover_cmd = build_standard_iprover_command(
                iprover_bin=iprover_bin,
                timeout=timeout,
                extra_args=iprover_extra_args,
                problem_file=problem_file,
            )

        completed = subprocess.run(
            iprover_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 20,
            env=env,
            cwd=str(tptp_root.resolve()),
        )

        wallclock = time.time() - start
        output = completed.stdout + "\n" + completed.stderr
        parsed = parse_iprover_output(output, measured_wallclock=wallclock, timeout=timeout)

        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"{problem_file.stem}.log"
            log_path.write_text(output, encoding="utf-8", errors="replace")

        agent_stdout, agent_stderr = collect_agent_output(agent_proc)

        return {
            "problem": make_csv_problem_name(problem_file, domain),
            "domain": domain,
            "status": parsed["status"],
            "solved": parsed["solved"],
            "runtime_seconds": round(parsed["runtime_seconds"], 6),
            "instantiations": parsed["instantiations"],
            "generated_clauses": parsed["generated_clauses"],
            "processed_clauses": parsed["processed_clauses"],
            "loop_iterations": parsed["loop_iterations"],
            "proof_found": parsed["proof_found"],
            "timed_out": parsed["timed_out"],
            "return_code": completed.returncode,
            "agent_error": (agent_stdout + "\n" + agent_stderr).strip(),
        }

    except subprocess.TimeoutExpired:
        wallclock = time.time() - start
        agent_stdout, agent_stderr = collect_agent_output(agent_proc)
        return {
            "problem": make_csv_problem_name(problem_file, domain),
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
            "agent_error": (agent_stdout + "\n" + agent_stderr).strip(),
        }

    except Exception as e:
        wallclock = time.time() - start
        agent_stdout, agent_stderr = collect_agent_output(agent_proc)
        return {
            "problem": make_csv_problem_name(problem_file, domain),
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
            "agent_error": (str(e) + "\n" + agent_stdout + "\n" + agent_stderr).strip(),
        }

    finally:
        safe_kill(agent_proc)


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


def write_stats_json(stats: Dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run standard or interactive iProver experiments over a folder of problems."
    )
    parser.add_argument("--problems", required=True)
    parser.add_argument("--tptp_root", default="../TPTPLibrary/TPTP-v9.2.1")
    parser.add_argument("--agent-cmd", default="")
    parser.add_argument("--iprover-bin", default="../iproveropt")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--iprover-extra-args", default="")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--logs-dir", default="")

    parser.add_argument(
        "--inst-passive-queues",
        default="[[-external_score]]",
        help=(
            "Interactive instantiation passive queues. "
            "Default: [[-external_score]]. "
            "Example hybrid: [[-external_score];[+age],;+[weight]]"
        ),
    )
    parser.add_argument(
        "--inst-passive-queues-freq",
        default="[1]",
        help=(
            "Frequencies for --inst-passive-queues. "
            "Default: [1]. Example hybrid: [1,2,2]"
        ),
    )

    args = parser.parse_args()

    overall_start = time.time()

    problems_dir = Path(args.problems).resolve()
    if not problems_dir.exists() or not problems_dir.is_dir():
        print(f"Problem folder does not exist or is not a directory: {problems_dir}", file=sys.stderr)
        return 1

    tptp_root = infer_tptp_root(Path(args.tptp_root))

    iprover_bin = Path(args.iprover_bin).resolve()
    if not iprover_bin.is_file():
        print(f"iProver binary not found: {iprover_bin}", file=sys.stderr)
        return 1

    iprover_extra_args = shlex.split(args.iprover_extra_args)
    agent_cmd_template = args.agent_cmd.strip() or None

    default_results_base = (project_root() / "experiments" / "results").resolve()
    default_results_base.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_output_path(args.results_csv, default_results_base)

    logs_dir = None
    if args.logs_dir:
        logs_dir = resolve_output_path(args.logs_dir, project_root())

    problem_files = sorted(problems_dir.rglob("*.p"))
    if not problem_files:
        print(f"No .p files found under: {problems_dir}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []

    print(f"Found {len(problem_files)} problem files.")
    print(f"TPTP root: {tptp_root}")
    print(f"iProver binary: {iprover_bin}")
    print("Mode:", "interactive agent-guided" if agent_cmd_template else "standard iProver")
    print(f"Results CSV: {csv_path}")
    if agent_cmd_template:
        print(f"inst_passive_queues: {args.inst_passive_queues}")
        print(f"inst_passive_queues_freq: {args.inst_passive_queues_freq}")
    if logs_dir is not None:
        print(f"Logs dir: {logs_dir}")

    for i, problem in enumerate(problem_files, start=1):
        print(f"[{i}/{len(problem_files)}] Running {problem}")

        row = run_one_problem(
            problem_file=problem,
            problems_root=problems_dir,
            tptp_root=tptp_root,
            agent_cmd_template=agent_cmd_template,
            iprover_bin=str(iprover_bin),
            iprover_extra_args=iprover_extra_args,
            timeout=args.timeout,
            logs_dir=logs_dir,
            inst_passive_queues=args.inst_passive_queues,
            inst_passive_queues_freq=args.inst_passive_queues_freq,
        )
        rows.append(row)

        print(
            f"    domain={row['domain']} "
            f"status={row['status']} "
            f"solved={row['solved']} "
            f"runtime={row['runtime_seconds']} "
            f"inst={row['instantiations']} "
            f"loops={row['loop_iterations']}"
        )

    write_csv(rows, csv_path)

    total_runtime = time.time() - overall_start
    solved_count = sum(int(r.get("solved", 0)) for r in rows)
    timeout_count = sum(int(r.get("timed_out", 0)) for r in rows)

    stats = {
        "num_problems": len(problem_files),
        "solved_problems": solved_count,
        "timeout_problems": timeout_count,
        "total_runtime_seconds": round(total_runtime, 6),
        "avg_runtime_per_problem_seconds": round(total_runtime / len(problem_files), 6),
        "results_csv": str(csv_path),
        "mode": "interactive agent-guided" if agent_cmd_template else "standard iProver",
        "inst_passive_queues": args.inst_passive_queues if agent_cmd_template else None,
        "inst_passive_queues_freq": args.inst_passive_queues_freq if agent_cmd_template else None,
    }

    stats_json = csv_path.with_suffix(".stats.json")
    write_stats_json(stats, stats_json)

    print("\nFinished.")
    print(f"CSV results written to: {csv_path}")
    print(f"Stats JSON written to: {stats_json}")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Average per problem: {total_runtime / len(problem_files):.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())