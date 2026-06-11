#!/usr/bin/env python3
"""
Sweep runner for the FitzHugh-Nagumo RING network.

Runs Java simulations varying the ring half-width (ring_k) and, optionally,
several K values. Use this script for the TP5 ring-network experiments.

Examples:
    # TP point: K = 0.1, ring_k = 1..10, >10 realizations
    python Visualization/sweep_ring_k.py --compile-first --k-values 0.1 --ring-k-values 1 2 3 4 5 6 7 8 9 10 --repetitions 15

    # Grid for 2D maps sigma(K, ring_k) and t_sync(K, ring_k)
    python Visualization/sweep_ring_k.py --compile-first --k-values 0.01 0.03 0.05 0.1 0.2 0.5 1.0 --ring-k-values 1 2 3 4 5 6 7 8 9 10 --repetitions 12
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SIMULATION_DIR = _PROJECT_ROOT / "Simulation"
_FHN_DIR = _SIMULATION_DIR / "fhn"
_BIN_DIR = _SIMULATION_DIR / "bin"


def _fmt_num(value: float | int) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ring-network sweeps over ring_k and K.")
    parser.add_argument("--ring-k-values", type=int, nargs="+", default=list(range(1, 11)), help="Ring half-width values, default 1..10")
    parser.add_argument("--k-values", type=float, nargs="+", default=[0.1], help="K values to run, default 0.1")
    parser.add_argument("--repetitions", type=int, required=True, help="Independent realizations per (K, ring_k)")
    parser.add_argument("--n", type=int, default=512, help="Number of neurons (TP requires > 500)")
    parser.add_argument("--dt", type=float, default=0.005, help="Integrator step")
    parser.add_argument("--tmax", type=float, default=500.0, help="Final simulation time")
    parser.add_argument("--sample-every", type=int, default=10, help="Integrator steps between stored samples")
    parser.add_argument("--base-seed", type=int, default=42000, help="Initial seed; each run increments it")
    parser.add_argument("--threads", type=int, default=1, help="Parallel Java processes")
    parser.add_argument("--compile-first", action="store_true", help="Compile Java before running")
    parser.add_argument("--java-cmd", default="java", help="Java command")
    parser.add_argument("--javac-cmd", default="javac", help="Javac command")
    parser.add_argument("--output-prefix", default="ring_sweep", help="Output file prefix in Simulation/output")
    parser.add_argument("--retry-dts", type=float, nargs="*", default=[], help="Optional smaller dt values to retry if a run fails")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n <= 500:
        raise ValueError("--n must be > 500")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if any(v < 1 for v in args.ring_k_values):
        raise ValueError("All --ring-k-values must be >= 1")
    if any(dt <= 0 for dt in [args.dt, *args.retry_dts]):
        raise ValueError("dt values must be > 0")


def compile_java(javac_cmd: str) -> None:
    java_files = sorted(str(path) for path in _FHN_DIR.glob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No Java files found in {_FHN_DIR}")
    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [javac_cmd, "-d", str(_BIN_DIR), *java_files]
    print("Compiling Java:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)


def _build_jobs(args: argparse.Namespace) -> list[tuple[int, float, int, int, int, list[str]]]:
    jobs: list[tuple[int, float, int, int, int, list[str]]] = []
    run_no = 0
    for k in args.k_values:
        for ring_k in args.ring_k_values:
            for realization in range(1, args.repetitions + 1):
                run_no += 1
                seed = args.base_seed + run_no - 1
                out_name = (
                    f"{args.output_prefix}_K{_fmt_num(k)}_ring{ring_k}"
                    f"_real{realization}_seed{seed}.txt"
                )
                cmd = [
                    args.java_cmd,
                    "-cp",
                    str(_BIN_DIR),
                    "fhn.Main",
                    "--n",
                    _fmt_num(args.n),
                    "--k",
                    _fmt_num(k),
                    "--dt",
                    "{DT}",
                    "--tmax",
                    _fmt_num(args.tmax),
                    "--network",
                    "RING",
                    "--ring-k",
                    _fmt_num(ring_k),
                    "--seed",
                    _fmt_num(seed),
                    "--realization",
                    _fmt_num(realization),
                    "--sample-every",
                    _fmt_num(args.sample_every),
                    "--output",
                    str(Path("output") / out_name),
                ]
                jobs.append((run_no, k, ring_k, realization, seed, cmd))
    return jobs


def run_sweep(args: argparse.Namespace) -> None:
    jobs = _build_jobs(args)
    total = len(jobs)
    retry_dts = [args.dt, *args.retry_dts]
    print(
        f"Starting RING sweep: {len(args.k_values)} K values x "
        f"{len(args.ring_k_values)} ring_k values x {args.repetitions} reps = {total} runs"
    )

    def _run_job(job: tuple[int, float, int, int, int, list[str]]) -> dict[str, object]:
        run_no, k, ring_k, realization, seed, cmd_template = job
        last_error = ""
        for attempt, dt_try in enumerate(retry_dts, start=1):
            cmd = [(_fmt_num(dt_try) if token == "{DT}" else token) for token in cmd_template]
            result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return {"ok": True, "run_no": run_no, "k": k, "ring_k": ring_k, "realization": realization, "seed": seed, "dt": dt_try, "attempt": attempt}
            last_error = (result.stderr or result.stdout or "").strip()
        return {"ok": False, "run_no": run_no, "k": k, "ring_k": ring_k, "realization": realization, "seed": seed, "error": last_error}

    successes = 0
    failures: list[dict[str, object]] = []

    if args.threads == 1:
        iterator = (_run_job(job) for job in jobs)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.threads)
        iterator = concurrent.futures.as_completed([executor.submit(_run_job, job) for job in jobs])

    try:
        for item in iterator:
            result = item.result() if hasattr(item, "result") else item
            done = successes + len(failures) + 1
            if bool(result["ok"]):
                successes += 1
                print(
                    f"[{done}/{total}] done K={_fmt_num(float(result['k']))} "
                    f"ring_k={result['ring_k']} real={result['realization']} seed={result['seed']} dt={_fmt_num(float(result['dt']))}"
                )
            else:
                failures.append(result)
                print(
                    f"[{done}/{total}] FAILED K={_fmt_num(float(result['k']))} "
                    f"ring_k={result['ring_k']} real={result['realization']} seed={result['seed']}"
                )
    finally:
        if args.threads != 1:
            executor.shutdown(wait=True)

    print(f"Finished: {successes} succeeded, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for fail in failures[:20]:
            print(f"- K={fail['k']} ring_k={fail['ring_k']} real={fail['realization']}: {fail.get('error', '')[:300]}")
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.compile_first:
        compile_java(args.javac_cmd)
    run_sweep(args)


if __name__ == "__main__":
    main()
