#!/usr/bin/env python3
"""
Ring-network summary plots for FitzHugh-Nagumo simulations.

Reads Java output .txt files for NETWORK=RING and groups realizations by ring_k.
Useful for the TP point with K fixed (typically K=0.1):
- <v(t)> averaged over realizations, one curve per ring_k
- sigma_v(t) averaged over realizations, one curve per ring_k
- stationary sigma_v vs ring_k
- synchronization time vs ring_k
- CSV summary for tables/presentation
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INPUT_DIR = _PROJECT_ROOT / "Simulation" / "output"
_BEGIN_STATE_RE = re.compile(r"^BEGIN_STATE t=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


@dataclass
class RunData:
    path: Path
    k: float
    ring_k: int
    realization: int
    times: np.ndarray
    mean_v: np.ndarray
    sigma_v: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RING runs grouped by ring_k.")
    parser.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR, help="Directory with simulation .txt outputs")
    parser.add_argument("--pattern", default="*ring*.txt", help="Glob pattern inside input-dir")
    parser.add_argument("--k", type=float, default=0.1, help="Only analyze this K value")
    parser.add_argument("--k-tol", type=float, default=1e-12, help="Tolerance for matching K")
    parser.add_argument("--stationary-fraction", type=float, default=0.2, help="Final fraction used for stationary sigma, default 0.2")
    parser.add_argument("--sigma-threshold", type=float, default=1e-7, help="Threshold for synchronization")
    parser.add_argument("--sync-window", type=int, default=1, help="Require sigma below threshold for this many consecutive stored samples")
    parser.add_argument("--sync-fill", choices=["nan", "tmax"], default="nan", help="Value if a run never synchronizes")
    parser.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "Visualization" / "ring_analysis", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _parse_header_value(line: str, key: str) -> str | None:
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
    means: list[float] = []
    sigmas: list[float] = []
    current_t: float | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if (val := _parse_header_value(line, "K")) is not None:
                k_value = float(val)
                continue
            if (val := _parse_header_value(line, "ring_k")) is not None:
                ring_k = int(val)
                continue
            if (val := _parse_header_value(line, "network")) is not None:
                network = val.upper()
                continue
            if (val := _parse_header_value(line, "realization")) is not None:
                realization = int(val)
                continue

            match = _BEGIN_STATE_RE.match(line)
            if match:
                current_t = float(match.group(1))
                continue

            if line.startswith("v ") and current_t is not None:
                vals = np.fromstring(line[2:], sep=" ")
                if vals.size == 0:
                    raise ValueError(f"Could not parse v values in {path}")
                times.append(current_t)
                means.append(float(vals.mean()))
                sigmas.append(float(vals.std()))
                current_t = None

    if network != "RING":
        raise ValueError(f"Not a RING output: {path}")
    if k_value is None or ring_k is None:
        raise ValueError(f"Missing K or ring_k in {path}")
    if not times:
        raise ValueError(f"No states found in {path}")

    order = np.argsort(np.asarray(times))
    return RunData(
        path=path,
        k=k_value,
        ring_k=ring_k,
        realization=realization,
        times=np.asarray(times)[order],
        mean_v=np.asarray(means)[order],
        sigma_v=np.asarray(sigmas)[order],
    )


def load_runs(input_dir: Path, pattern: str, k: float, k_tol: float) -> dict[int, list[RunData]]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern} in {input_dir}")
    grouped: dict[int, list[RunData]] = defaultdict(list)
    skipped = 0
    for path in files:
        try:
            run = parse_run(path)
        except Exception:
            skipped += 1
            continue
        if abs(run.k - k) <= k_tol:
            grouped[run.ring_k].append(run)
    if not grouped:
        raise SystemExit(f"No RING runs found for K={k:g}")
    if skipped:
        print(f"Warning: skipped {skipped} non-matching/unreadable files.")
    return dict(sorted(grouped.items()))


def align_runs(runs: list[RunData], attr: str) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(run.times) for run in runs)
    times = runs[0].times[:min_len]
    values = np.vstack([getattr(run, attr)[:min_len] for run in runs])
    return times, values


def find_t_sync(times: np.ndarray, sigma: np.ndarray, threshold: float, window: int, sync_fill: str) -> float:
    below = sigma < threshold
    if window <= 1:
        idxs = np.where(below)[0]
        if len(idxs):
            return float(times[int(idxs[0])])
    else:
        for i in range(0, len(below) - window + 1):
            if bool(np.all(below[i : i + window])):
                return float(times[i])
    if sync_fill == "tmax":
        return float(times[-1])
    return float("nan")


def stationary_value(times: np.ndarray, sigma: np.ndarray, fraction: float) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("stationary-fraction must be in (0, 1]")
    start = int(math.floor((1.0 - fraction) * len(times)))
    start = max(0, min(start, len(times) - 1))
    return float(np.mean(sigma[start:]))


def plot_curves(summary: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]], ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    for ring_k, (times, mean, std, n_reps) in summary.items():
        ax.plot(times, mean, linewidth=2.0, label=f"v={ring_k} (n={n_reps})")
        ax.fill_between(times, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Tiempo", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=19)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errorbar(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=2.2, markersize=7, capsize=4)
    ax.set_xlabel("Vecindad del anillo v = ring_k", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    out_dir = _resolve(args.out_dir)
    grouped = load_runs(input_dir, args.pattern, args.k, args.k_tol)

    mean_summary: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    sigma_summary: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    rows: list[dict[str, float | int]] = []

    for ring_k, runs in grouped.items():
        times_m, mean_values = align_runs(runs, "mean_v")
        times_s, sigma_values = align_runs(runs, "sigma_v")
        mean_summary[ring_k] = (times_m, mean_values.mean(axis=0), mean_values.std(axis=0), len(runs))
        sigma_summary[ring_k] = (times_s, sigma_values.mean(axis=0), sigma_values.std(axis=0), len(runs))

        sync_times = [find_t_sync(run.times, run.sigma_v, args.sigma_threshold, args.sync_window, args.sync_fill) for run in runs]
        stat_sigmas = [stationary_value(run.times, run.sigma_v, args.stationary_fraction) for run in runs]
        sync_arr = np.asarray(sync_times, dtype=float)
        stat_arr = np.asarray(stat_sigmas, dtype=float)
        rows.append({
            "K": args.k,
            "ring_k": ring_k,
            "n_reps": len(runs),
            "stationary_sigma_mean": float(np.nanmean(stat_arr)),
            "stationary_sigma_std": float(np.nanstd(stat_arr, ddof=1)) if np.isfinite(stat_arr).sum() > 1 else 0.0,
            "sync_time_mean": float(np.nanmean(sync_arr)) if np.isfinite(sync_arr).any() else float("nan"),
            "sync_time_std": float(np.nanstd(sync_arr, ddof=1)) if np.isfinite(sync_arr).sum() > 1 else 0.0,
        })

    plot_curves(mean_summary, "Potencial promedio <v(t)>", f"Red anillo: <v(t)> para K={args.k:g}", out_dir / "mean_v_vs_time_by_ring_k.png")
    plot_curves(sigma_summary, "Dispersion espacial sigma_v(t)", f"Red anillo: sigma_v(t) para K={args.k:g}", out_dir / "sigma_v_vs_time_by_ring_k.png")

    x = np.asarray([row["ring_k"] for row in rows], dtype=int)
    stat_mean = np.asarray([row["stationary_sigma_mean"] for row in rows], dtype=float)
    stat_std = np.asarray([row["stationary_sigma_std"] for row in rows], dtype=float)
    sync_mean = np.asarray([row["sync_time_mean"] for row in rows], dtype=float)
    sync_std = np.asarray([row["sync_time_std"] for row in rows], dtype=float)

    plot_errorbar(x, stat_mean, stat_std, "Sigma_v estacionaria", f"Sigma estacionaria vs vecindad, K={args.k:g}", out_dir / "stationary_sigma_vs_ring_k.png")
    plot_errorbar(x, sync_mean, sync_std, "Tiempo de sincronizacion", f"Tiempo de sincronizacion vs vecindad, K={args.k:g}", out_dir / "sync_time_vs_ring_k.png")

    csv_path = out_dir / "ring_k_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Analyzed {sum(len(v) for v in grouped.values())} runs.")
    print(f"Outputs written to: {out_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
