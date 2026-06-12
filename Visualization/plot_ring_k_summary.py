#!/usr/bin/env python3
"""
Ring-network K-fixed analysis for FitzHugh-Nagumo simulations.

Mirrors the random-network p-analysis, replacing p by ring_k.
For one fixed K, it computes stationary metrics as functions of the
ring neighborhood k and averages over independent realizations.

Main outputs:
- stationary_mean_v_vs_ring_k.png
- stationary_sigma_v_vs_ring_k.png
- stationary_time_vs_ring_k.png
- sync_time_vs_ring_k.png
- ring_k_summary.csv

Optional diagnostics with --time-curves:
- mean_v_vs_time_by_ring_k.png
- sigma_v_vs_time_by_ring_k.png
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INPUT_DIR = _PROJECT_ROOT / "Simulation" / "output"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "Visualization" / "ring_analysis"
_BEGIN_STATE_RE = re.compile(r"^BEGIN_STATE t=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


@dataclass(frozen=True)
class RunData:
    path: Path
    k: float
    ring_k: int
    realization: int
    times: np.ndarray
    mean_v: np.ndarray
    sigma_v: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RING outputs for one fixed K and plot stationary metrics vs ring_k."
    )
    parser.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR,
                        help="Directory with Java .txt outputs")
    parser.add_argument("--pattern", default="*ring*.txt",
                        help="Glob pattern inside input-dir")
    parser.add_argument("--k", type=float, default=0.1,
                        help="Only analyze this K value, default 0.1")
    parser.add_argument("--k-tol", type=float, default=1e-12,
                        help="Tolerance for matching K")
    parser.add_argument("--ring-k-values", type=int, nargs="*", default=None,
                        help="Optional ring_k values to include, e.g. 1 2 ... 10")
    parser.add_argument("--stationary-fraction", type=float, default=0.2,
                        help="Final fraction of samples used for stationary averages")
    parser.add_argument("--stationary-window", type=int, default=40,
                        help="Stored-sample window used to detect arrival to stationary regime")
    parser.add_argument("--stationary-epsilon", type=float, default=2e-3,
                        help="Max-min tolerance of sigma_v inside the stationary detection window")
    parser.add_argument("--stationary-fill", choices=["nan", "tmax"], default="nan",
                        help="Value if a run never satisfies stationary-window criterion")
    parser.add_argument("--sigma-threshold", type=float, default=1e-2,
                        help="Synchronization threshold for sigma_v(t)")
    parser.add_argument("--sync-window", type=int, default=20,
                        help="Require sigma below threshold for this many consecutive stored samples")
    parser.add_argument("--sync-fill", choices=["nan", "tmax"], default="nan",
                        help="Value if a run never synchronizes")
    parser.add_argument("--time-curves", action="store_true",
                        help="Also save mean_v(t) and sigma_v(t) averaged curves by ring_k")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Directory where plots and CSV are written")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively after saving")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _header_value(line: str, key: str) -> str | None:
    prefix = key + "="
    if line.startswith(prefix):
        return line.split("=", 1)[1].strip()
    return None


def parse_run(path: Path) -> RunData:
    k_value: float | None = None
    ring_k: int | None = None
    network: str | None = None
    realization = 1
    times: list[float] = []
    mean_values: list[float] = []
    sigma_values: list[float] = []
    current_t: float | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if (val := _header_value(line, "K")) is not None:
                k_value = float(val)
                continue
            if (val := _header_value(line, "ring_k")) is not None:
                ring_k = int(val)
                continue
            if (val := _header_value(line, "ring-k")) is not None:
                ring_k = int(val)
                continue
            if (val := _header_value(line, "network")) is not None:
                network = val.upper()
                continue
            if (val := _header_value(line, "realization")) is not None:
                realization = int(val)
                continue

            match = _BEGIN_STATE_RE.match(line)
            if match:
                current_t = float(match.group(1))
                continue

            if line.startswith("v ") and current_t is not None:
                values = np.fromstring(line[2:], sep=" ")
                if values.size == 0:
                    raise ValueError(f"Could not parse v values in {path}")
                times.append(current_t)
                mean_values.append(float(values.mean()))
                sigma_values.append(float(values.std()))
                current_t = None

    if network != "RING":
        raise ValueError(f"Not a RING output: {path}")
    if k_value is None or ring_k is None:
        raise ValueError(f"Missing K or ring_k in {path}")
    if not times:
        raise ValueError(f"No states found in {path}")

    order = np.argsort(np.asarray(times, dtype=float))
    return RunData(
        path=path,
        k=k_value,
        ring_k=ring_k,
        realization=realization,
        times=np.asarray(times, dtype=float)[order],
        mean_v=np.asarray(mean_values, dtype=float)[order],
        sigma_v=np.asarray(sigma_values, dtype=float)[order],
    )


def load_runs(args: argparse.Namespace) -> dict[int, list[RunData]]:
    input_dir = _resolve(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} in {input_dir}")

    wanted_ring = set(args.ring_k_values) if args.ring_k_values else None
    grouped: dict[int, list[RunData]] = defaultdict(list)
    skipped = 0

    for path in files:
        try:
            run = parse_run(path)
        except Exception:
            skipped += 1
            continue
        if abs(run.k - args.k) > args.k_tol:
            continue
        if wanted_ring is not None and run.ring_k not in wanted_ring:
            continue
        grouped[run.ring_k].append(run)

    if not grouped:
        raise SystemExit(f"No RING runs found for K={args.k:g} in {input_dir}")
    if skipped:
        print(f"Warning: skipped {skipped} non-RING/unreadable files.")
    return dict(sorted(grouped.items()))


def stationary_segment(values: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("stationary-fraction must be in (0, 1]")
    start = int(math.floor((1.0 - fraction) * len(values)))
    start = max(0, min(start, len(values) - 1))
    return values[start:]


def stationary_mean(values: np.ndarray, fraction: float) -> float:
    return float(np.mean(stationary_segment(values, fraction)))


def stationary_amplitude(values: np.ndarray, fraction: float) -> float:
    tail = stationary_segment(values, fraction)
    return float(np.max(tail) - np.min(tail))


def find_t_sync(times: np.ndarray, sigma: np.ndarray, threshold: float, window: int, sync_fill: str) -> float:
    below = sigma < threshold
    window = max(1, int(window))
    if window > len(below):
        return float(times[-1]) if sync_fill == "tmax" else float("nan")
    for i in range(0, len(below) - window + 1):
        if bool(np.all(below[i:i + window])):
            return float(times[i])
    return float(times[-1]) if sync_fill == "tmax" else float("nan")


def find_t_stationary(times: np.ndarray, sigma: np.ndarray, window: int, epsilon: float, fill: str) -> float:
    """Arrival time to stationary regime based on variation of sigma_v.

    The run is considered stationary at the first stored time t_i such that
    max(sigma_v[i:i+window]) - min(sigma_v[i:i+window]) <= epsilon.
    """
    window = max(2, int(window))
    if window > len(sigma):
        return float(times[-1]) if fill == "tmax" else float("nan")
    for i in range(0, len(sigma) - window + 1):
        segment = sigma[i:i + window]
        if float(np.max(segment) - np.min(segment)) <= epsilon:
            return float(times[i])
    return float(times[-1]) if fill == "tmax" else float("nan")


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return 0.0
    return float(np.nanstd(finite, ddof=1))


def align_runs(runs: list[RunData], attr: str) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(run.times) for run in runs)
    times = runs[0].times[:min_len]
    values = np.vstack([getattr(run, attr)[:min_len] for run in runs])
    return times, values


def plot_errorbar(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                  ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="#1f4e79",      # azul oscuro
        ecolor="#d95f02",     # rojo-anaranjado
        linewidth=2.2,
        markersize=7,
        capsize=4,
        elinewidth=1.8,
    )
    ax.set_xlabel(r"Vecinos por lado $k$", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sigma_vs_ring_k(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                         out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="#228B22",      # verde
        ecolor="#ef6c00",     # naranja
        linewidth=2.2,
        markersize=7,
        capsize=4,
        elinewidth=1.8,
    )
    ax.set_xlabel(r"Vecinos por lado $k$", fontsize=15)
    ax.set_ylabel(
        #r"Dispersión espacial $\langle \sigma_{v} \rangle_{est}$",
        r"Dispersión espacial \sigma_{v}$",
        fontsize=15,
    )
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_time_curves(summary: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]],
                     ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    for ring_k, (times, mean, std, n_reps) in summary.items():
        ax.plot(times, mean, linewidth=2.0, label=rf"$k={ring_k}$ (n={n_reps})")
        ax.fill_between(times, mean - std, mean + std, alpha=0.14)
    ax.set_xlabel(r"Tiempo $t$ (s)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_runs(args)

    rows: list[dict[str, float | int]] = []
    mean_time_summary: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    sigma_time_summary: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

    for ring_k, runs in grouped.items():
        if args.time_curves:
            times_m, mean_values = align_runs(runs, "mean_v")
            times_s, sigma_values = align_runs(runs, "sigma_v")
            mean_time_summary[ring_k] = (times_m, mean_values.mean(axis=0), mean_values.std(axis=0), len(runs))
            sigma_time_summary[ring_k] = (times_s, sigma_values.mean(axis=0), sigma_values.std(axis=0), len(runs))

        stat_mean_v = [stationary_mean(run.mean_v, args.stationary_fraction) for run in runs]
        amp_mean_v = [stationary_amplitude(run.mean_v, args.stationary_fraction) for run in runs]
        stat_sigma = [stationary_mean(run.sigma_v, args.stationary_fraction) for run in runs]
        sync_times = [find_t_sync(run.times, run.sigma_v, args.sigma_threshold, args.sync_window, args.sync_fill) for run in runs]
        stationary_times = [find_t_stationary(run.times, run.sigma_v, args.stationary_window, args.stationary_epsilon, args.stationary_fill) for run in runs]

        rows.append({
            "K": args.k,
            "ring_k": ring_k,
            "degree_total": 2 * ring_k,
            "n_reps": len(runs),
            "stationary_fraction": args.stationary_fraction,
            "stationary_window": args.stationary_window,
            "stationary_epsilon": args.stationary_epsilon,
            "sigma_threshold": args.sigma_threshold,
            "sync_window": args.sync_window,
            "stationary_mean_v_mean": finite_mean(stat_mean_v),
            "stationary_mean_v_std": finite_std(stat_mean_v),
            "stationary_mean_v_amplitude_mean": finite_mean(amp_mean_v),
            "stationary_mean_v_amplitude_std": finite_std(amp_mean_v),
            "stationary_sigma_mean": finite_mean(stat_sigma),
            "stationary_sigma_std": finite_std(stat_sigma),
            "stationary_time_mean": finite_mean(stationary_times),
            "stationary_time_std": finite_std(stationary_times),
            "sync_time_mean": finite_mean(sync_times),
            "sync_time_std": finite_std(sync_times),
        })

    x = np.asarray([int(row["ring_k"]) for row in rows], dtype=int)
    plot_errorbar(
        x,
        np.asarray([float(row["stationary_mean_v_mean"]) for row in rows]),
        np.asarray([float(row["stationary_mean_v_std"]) for row in rows]),
        r"Potencial promedio $\langle v \rangle$",
        out_dir / "stationary_mean_v_vs_ring_k.png",
    )
    plot_sigma_vs_ring_k(
        x,
        np.asarray([float(row["stationary_sigma_mean"]) for row in rows]),
        np.asarray([float(row["stationary_sigma_std"]) for row in rows]),
        out_dir / "stationary_sigma_v_vs_ring_k.png",
    )
    plot_errorbar(
        x,
        np.asarray([float(row["stationary_mean_v_amplitude_mean"]) for row in rows]),
        np.asarray([float(row["stationary_mean_v_amplitude_std"]) for row in rows]),
        r"Amplitud del potencial promedio $\Delta\langle v \rangle_{est}$",
        out_dir / "mean_v_amplitude_vs_ring_k.png",
    )
    plot_errorbar(
        x,
        np.asarray([float(row["stationary_time_mean"]) for row in rows]),
        np.asarray([float(row["stationary_time_std"]) for row in rows]),
        r"Tiempo de llegada al estacionario promedio $t_{est}$ (s)",
        out_dir / "stationary_time_vs_ring_k.png",
    )
    plot_errorbar(
        x,
        np.asarray([float(row["sync_time_mean"]) for row in rows]),
        np.asarray([float(row["sync_time_std"]) for row in rows]),
        r"Tiempo de sincronización promedio $t_{sync}$ (s)",
        out_dir / "sync_time_vs_ring_k.png",
    )

    if args.time_curves:
        plot_time_curves(
            mean_time_summary,
            r"Potencial promedio $\langle v\rangle$",
            out_dir / "mean_v_vs_time_by_ring_k.png",
        )
        plot_time_curves(
            sigma_time_summary,
            r"Dispersión espacial $\sigma_v$",
            out_dir / "sigma_v_vs_time_by_ring_k.png",
        )

    csv_path = out_dir / "ring_k_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Analyzed {sum(len(v) for v in grouped.values())} runs for K={args.k:g}.")
    print(f"Outputs written to: {out_dir}")
    print("Main plots:")
    print(f"  {out_dir / 'stationary_mean_v_vs_ring_k.png'}")
    print(f"  {out_dir / 'stationary_sigma_v_vs_ring_k.png'}")
    print(f"  {out_dir / 'stationary_time_vs_ring_k.png'}")
    print(f"  {out_dir / 'sync_time_vs_ring_k.png'}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
