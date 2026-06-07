#!/usr/bin/env python3
"""
Sweep runner for FitzHugh-Nagumo simulations.

Runs multiple Java simulations for a list of K values and N repetitions
per K. This script is intentionally separate from animate.py.

Examples:
    python Visualization/sweep.py --compile-first --k-values 0.6 0.8 1.0 --repetitions 5
    python Visualization/sweep.py --k-values 1.0 --repetitions 3 --network RING --ring-k 4
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
    parser = argparse.ArgumentParser(
        description="Run sweep over K with multiple repetitions per K.",
    )
    parser.add_argument("--k-values", type=float, nargs="+", required=True, help="List of K values to run")
    parser.add_argument("--repetitions", type=int, required=True, help="Runs per K value")
    parser.add_argument("--n", type=int, default=512, help="Number of neurons (must be > 500)")
    parser.add_argument("--dt", type=float, default=0.005, help="Integrator step")
    parser.add_argument("--tmax", type=float, default=500.0, help="Final simulation time")
    parser.add_argument("--network", choices=["FULL", "RANDOM", "RING"], default="RANDOM", help="Network topology")
    parser.add_argument("--p", type=float, default=0.2, help="Connection probability for RANDOM")
    parser.add_argument("--ring-k", type=int, default=3, help="Ring half-width for RING")
    parser.add_argument("--sample-every", type=int, default=10, help="Integrator steps between samples")
    parser.add_argument("--base-seed", type=int, default=42, help="Initial seed used to generate run seeds")
    parser.add_argument("--threads", type=int, default=1, help="Number of parallel runs to execute")
    parser.add_argument(
        "--retry-dts",
        type=float,
        nargs="*",
        default=[],
        help="Optional list of smaller dt values to retry on failed runs (e.g. --retry-dts 0.0025 0.001)",
    )
    parser.add_argument("--compile-first", action="store_true", help="Compile Java before running sweep")
    parser.add_argument("--java-cmd", default="java", help="Java command")
    parser.add_argument("--javac-cmd", default="javac", help="Javac command")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for output files (stored in Simulation/output)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    if args.n <= 500:
        raise ValueError("--n must be > 500 (TP5 requirement)")
    if not args.k_values:
        raise ValueError("--k-values cannot be empty")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if any(dt <= 0 for dt in args.retry_dts):
        raise ValueError("All --retry-dts values must be > 0")


def compile_java(javac_cmd: str) -> None:
    java_files = sorted(str(path) for path in _FHN_DIR.glob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No Java files found in {_FHN_DIR}")

    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [javac_cmd, "-d", str(_BIN_DIR), *java_files]
    print("Compiling Java:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)


def run_sweep(args: argparse.Namespace) -> None:
    total_runs = len(args.k_values) * args.repetitions
    print(
        f"Starting sweep: {len(args.k_values)} K values x "
        f"{args.repetitions} repetitions = {total_runs} runs"
    )

    jobs: list[tuple[int, float, int, int, list[str]]] = []
    run_no = 0
    for k in args.k_values:
        for realization in range(1, args.repetitions + 1):
            run_no += 1
            seed = args.base_seed + run_no - 1

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
                args.network,
                "--seed",
                _fmt_num(seed),
                "--realization",
                _fmt_num(realization),
                "--sample-every",
                _fmt_num(args.sample_every),
            ]

            if args.network == "RANDOM":
                cmd += ["--p", _fmt_num(args.p)]
            if args.network == "RING":
                cmd += ["--ring-k", _fmt_num(args.ring_k)]

            if args.output_prefix:
                file_name = (
                    f"{args.output_prefix}_K{_fmt_num(k)}"
                    f"_real{realization}_seed{seed}.txt"
                )
                cmd += ["--output", str(Path("output") / file_name)]

            jobs.append((run_no, k, realization, seed, cmd))

    retry_dts = [args.dt, *args.retry_dts]

    def _run_job(job: tuple[int, float, int, int, list[str]]) -> dict[str, object]:
        run_no, k, realization, seed, cmd_template = job
        last_error = ""
        for attempt, dt_try in enumerate(retry_dts, start=1):
            cmd = [(_fmt_num(dt_try) if token == "{DT}" else token) for token in cmd_template]
            result = subprocess.run(
                cmd,
                check=False,
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return {
                    "ok": True,
                    "run_no": run_no,
                    "k": k,
                    "realization": realization,
                    "seed": seed,
                    "dt": dt_try,
                    "attempt": attempt,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

            last_error = (result.stderr or result.stdout or "").strip()

        return {
            "ok": False,
            "run_no": run_no,
            "k": k,
            "realization": realization,
            "seed": seed,
            "dt": retry_dts[-1],
            "attempt": len(retry_dts),
            "error": last_error,
        }

    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    if args.threads == 1:
        for job in jobs:
            result = _run_job(job)
            if bool(result["ok"]):
                successes.append(result)
                print(
                    f"[{len(successes) + len(failures)}/{total_runs}] done "
                    f"run={result['run_no']} K={_fmt_num(float(result['k']))} "
                    f"real={result['realization']} seed={result['seed']} dt={_fmt_num(float(result['dt']))}"
                )
            else:
                failures.append(result)
                print(
                    f"[{len(successes) + len(failures)}/{total_runs}] FAILED "
                    f"run={result['run_no']} K={_fmt_num(float(result['k']))} "
                    f"real={result['realization']} seed={result['seed']}"
                )
    else:
        print(f"Running with parallel threads: {args.threads}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {executor.submit(_run_job, job): job for job in jobs}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                done += 1
                result = future.result()
                if bool(result["ok"]):
                    successes.append(result)
                    print(
                        f"[{done}/{total_runs}] done run={result['run_no']} "
                        f"K={_fmt_num(float(result['k']))} real={result['realization']} "
                        f"seed={result['seed']} dt={_fmt_num(float(result['dt']))}"
                    )
                else:
                    failures.append(result)
                    print(
                        f"[{done}/{total_runs}] FAILED run={result['run_no']} "
                        f"K={_fmt_num(float(result['k']))} real={result['realization']} "
                        f"seed={result['seed']}"
                    )

    print("Sweep finished.")
    print(f"Successful runs: {len(successes)}")
    print(f"Failed runs: {len(failures)}")
    print(f"Outputs are in: {_SIMULATION_DIR / 'output'}")

    if failures:
        print("\nFailure summary:")
        for failure in sorted(failures, key=lambda x: int(x["run_no"])):
            err_text = str(failure.get("error", "")).replace("\n", " ")
            if len(err_text) > 180:
                err_text = err_text[:177] + "..."
            print(
                f"- run={failure['run_no']} K={_fmt_num(float(failure['k']))} "
                f"real={failure['realization']} seed={failure['seed']} :: {err_text}"
            )
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.compile_first:
        compile_java(args.javac_cmd)

    run_sweep(args)


if __name__ == "__main__":
    main()
