#!/usr/bin/env python3
"""
Sweep runner for FitzHugh-Nagumo simulations over p.

Example:
    python Visualization/sweep_p.py \
        --p-values 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 \
        --k 0.1 \
        --repetitions 10 \
        --compile-first
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
        description="Run sweep over p with multiple repetitions per p.",
    )

    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        required=True,
        help="List of p values to run",
    )

    parser.add_argument(
        "--k",
        type=float,
        default=0.1,
        help="Fixed coupling K",
    )

    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--tmax", type=float, default=500.0)

    parser.add_argument(
        "--network",
        choices=["RANDOM"],
        default="RANDOM",
        help="Random network only",
    )

    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)

    parser.add_argument(
        "--retry-dts",
        type=float,
        nargs="*",
        default=[],
    )

    parser.add_argument("--compile-first", action="store_true")
    parser.add_argument("--java-cmd", default="java")
    parser.add_argument("--javac-cmd", default="javac")

    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for output files",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")

    if args.n <= 500:
        raise ValueError("--n must be > 500")

    if not args.p_values:
        raise ValueError("--p-values cannot be empty")

    if args.threads < 1:
        raise ValueError("--threads must be >= 1")


def compile_java(javac_cmd: str) -> None:
    java_files = sorted(str(path) for path in _FHN_DIR.glob("*.java"))

    _BIN_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [javac_cmd, "-d", str(_BIN_DIR), *java_files]

    print("Compiling Java:", " ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
        cwd=_PROJECT_ROOT,
    )


def run_sweep(args: argparse.Namespace) -> None:

    total_runs = len(args.p_values) * args.repetitions

    print(
        f"Starting sweep: {len(args.p_values)} p values x "
        f"{args.repetitions} repetitions = {total_runs} runs"
    )

    jobs = []

    run_no = 0

    for p in args.p_values:
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
                _fmt_num(args.k),
                "--dt",
                "{DT}",
                "--tmax",
                _fmt_num(args.tmax),
                "--network",
                "RANDOM",
                "--p",
                _fmt_num(p),
                "--seed",
                _fmt_num(seed),
                "--realization",
                _fmt_num(realization),
                "--sample-every",
                _fmt_num(args.sample_every),
            ]

            if args.output_prefix:
                file_name = (
                    f"{args.output_prefix}"
                    f"_P{_fmt_num(p)}"
                    f"_real{realization}"
                    f"_seed{seed}.txt"
                )

                cmd += [
                    "--output",
                    str(Path("output") / file_name)
                ]

            jobs.append(
                (run_no, p, realization, seed, cmd)
            )

    retry_dts = [args.dt, *args.retry_dts]

    def _run_job(job):

        run_no, p, realization, seed, cmd_template = job

        last_error = ""

        for attempt, dt_try in enumerate(retry_dts, start=1):

            cmd = [
                (_fmt_num(dt_try) if token == "{DT}" else token)
                for token in cmd_template
            ]

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
                    "p": p,
                    "realization": realization,
                    "seed": seed,
                    "dt": dt_try,
                }

            last_error = (
                result.stderr
                or result.stdout
                or ""
            ).strip()

        return {
            "ok": False,
            "run_no": run_no,
            "p": p,
            "realization": realization,
            "seed": seed,
            "error": last_error,
        }

    successes = []
    failures = []

    if args.threads == 1:

        for job in jobs:

            result = _run_job(job)

            if result["ok"]:

                successes.append(result)

                print(
                    f"[{len(successes)+len(failures)}/{total_runs}] "
                    f"done run={result['run_no']} "
                    f"p={_fmt_num(result['p'])} "
                    f"real={result['realization']}"
                )

            else:

                failures.append(result)

                print(
                    f"[{len(successes)+len(failures)}/{total_runs}] "
                    f"FAILED run={result['run_no']} "
                    f"p={_fmt_num(result['p'])}"
                )

    else:

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.threads
        ) as executor:

            futures = {
                executor.submit(_run_job, job): job
                for job in jobs
            }

            done = 0

            for future in concurrent.futures.as_completed(futures):

                done += 1

                result = future.result()

                if result["ok"]:
                    successes.append(result)

                    print(
                        f"[{done}/{total_runs}] "
                        f"done run={result['run_no']} "
                        f"p={_fmt_num(result['p'])}"
                    )

                else:
                    failures.append(result)

                    print(
                        f"[{done}/{total_runs}] "
                        f"FAILED run={result['run_no']} "
                        f"p={_fmt_num(result['p'])}"
                    )

    print("\nSweep finished.")
    print(f"Successful runs: {len(successes)}")
    print(f"Failed runs: {len(failures)}")

    if failures:
        raise SystemExit(1)


def main() -> None:

    args = parse_args()

    validate_args(args)

    if args.compile_first:
        compile_java(args.javac_cmd)

    run_sweep(args)


if __name__ == "__main__":
    main()