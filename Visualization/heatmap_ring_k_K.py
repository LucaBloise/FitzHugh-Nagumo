#!/usr/bin/env python3
"""
2D heatmaps for FitzHugh-Nagumo RING simulations.

Reads Java output .txt files with network=RING and computes, for each pair
(K, ring_k), averaged over realizations:
- stationary sigma_v
- synchronization time

Outputs:
- stationary_sigma_heatmap_ring_k_K.png
- sync_time_heatmap_ring_k_K.png
- ring_k_K_heatmap_summary.csv
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
    k: float
    ring_k: int
    realization: int
    times: np.ndarray
    sigma_v: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D heatmaps sigma_v(K, ring_k) and t_sync(K, ring_k).")
    parser.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR, help="Directory with simulation .txt outputs")
    parser.add_argument("--pattern", default="*ring*.txt", help="Glob pattern inside input-dir")
    parser.add_argument("--k-values", type=float, nargs="*", default=None, help="Optional K values to include")
    parser.add_argument("--ring-k-values", type=int, nargs="*", default=None, help="Optional ring_k values to include")
    parser.add_argument("--stationary-fraction", type=float, default=0.2, help="Final fraction used for stationary sigma")
    parser.add_argument("--sigma-threshold", type=float, default=1e-7, help="Threshold for synchronization")
    parser.add_argument("--sync-window", type=int, default=1, help="Require sigma below threshold for this many consecutive samples")
    parser.add_argument("--sync-fill", choices=["nan", "tmax"], default="nan", help="Value if a run never synchronizes")
    parser.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "Visualization" / "ring_analysis", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _header_value(line: str, key: str) -> str | None:
    prefix = key + "="
    return line.split("=", 1)[1].strip() if line.startswith(prefix) else None


def parse_run(path: Path) -> RunData:
    k_value: float | None = None
    ring_k: int | None = None
    realization = 1
    network: str | None = None
    times: list[float] = []
    sigmas: list[float] = []
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
                vals = np.fromstring(line[2:], sep=" ")
                if vals.size == 0:
                    raise ValueError(f"Could not parse v values in {path}")
                times.append(current_t)
                sigmas.append(float(vals.std()))
                current_t = None

    if network != "RING":
        raise ValueError(f"Not a RING output: {path}")
    if k_value is None or ring_k is None:
        raise ValueError(f"Missing K/ring_k in {path}")
    if not times:
        raise ValueError(f"No states found in {path}")
    order = np.argsort(np.asarray(times))
    return RunData(k=k_value, ring_k=ring_k, realization=realization, times=np.asarray(times)[order], sigma_v=np.asarray(sigmas)[order])


def stationary_value(times: np.ndarray, sigma: np.ndarray, fraction: float) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("stationary-fraction must be in (0, 1]")
    start = int(math.floor((1.0 - fraction) * len(times)))
    start = max(0, min(start, len(times) - 1))
    return float(np.mean(sigma[start:]))


def find_t_sync(times: np.ndarray, sigma: np.ndarray, threshold: float, window: int, sync_fill: str) -> float:
    below = sigma < threshold
    if window <= 1:
        idxs = np.where(below)[0]
        if idxs.size:
            return float(times[int(idxs[0])])
    else:
        for i in range(0, len(below) - window + 1):
            if bool(np.all(below[i : i + window])):
                return float(times[i])
    if sync_fill == "tmax":
        return float(times[-1])
    return float("nan")


def load_grouped(args: argparse.Namespace) -> dict[tuple[float, int], list[RunData]]:
    input_dir = _resolve(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {input_dir}")
    wanted_k = set(args.k_values) if args.k_values else None
    wanted_ring = set(args.ring_k_values) if args.ring_k_values else None
    grouped: dict[tuple[float, int], list[RunData]] = defaultdict(list)
    skipped = 0
    for path in files:
        try:
            run = parse_run(path)
        except Exception:
            skipped += 1
            continue
        if wanted_k is not None and not any(abs(run.k - k) <= 1e-12 for k in wanted_k):
            continue
        if wanted_ring is not None and run.ring_k not in wanted_ring:
            continue
        grouped[(run.k, run.ring_k)].append(run)
    if not grouped:
        raise SystemExit("No RING runs matched the selected filters.")
    if skipped:
        print(f"Warning: skipped {skipped} non-matching/unreadable files.")
    return dict(grouped)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return 0.0 if finite.size == 1 else float("nan")
    return float(np.nanstd(finite, ddof=1))


def build_summary(args: argparse.Namespace, grouped: dict[tuple[float, int], list[RunData]]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for (k, ring_k), runs in sorted(grouped.items()):
        stat = [stationary_value(run.times, run.sigma_v, args.stationary_fraction) for run in runs]
        sync = [find_t_sync(run.times, run.sigma_v, args.sigma_threshold, args.sync_window, args.sync_fill) for run in runs]
        rows.append({
            "K": k,
            "ring_k": ring_k,
            "n_reps": len(runs),
            "stationary_sigma_mean": _nanmean(stat),
            "stationary_sigma_std": _nanstd(stat),
            "sync_time_mean": _nanmean(sync),
            "sync_time_std": _nanstd(sync),
        })
    return rows


def matrix_from_rows(rows: list[dict[str, float | int]], value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_vals = np.asarray(sorted({float(row["K"]) for row in rows}), dtype=float)
    ring_vals = np.asarray(sorted({int(row["ring_k"]) for row in rows}), dtype=int)
    mat = np.full((len(k_vals), len(ring_vals)), np.nan, dtype=float)
    k_index = {k: i for i, k in enumerate(k_vals)}
    r_index = {r: j for j, r in enumerate(ring_vals)}
    for row in rows:
        mat[k_index[float(row["K"] )], r_index[int(row["ring_k"] )]] = float(row[value_key])
    return k_vals, ring_vals, mat


def plot_heatmap(k_vals: np.ndarray, ring_vals: np.ndarray, mat: np.ndarray, title: str, cbar_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    image = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(ring_vals)))
    ax.set_xticklabels([str(v) for v in ring_vals])
    ax.set_yticks(np.arange(len(k_vals)))
    ax.set_yticklabels([format(v, "g") for v in k_vals])
    ax.set_xlabel("Vecindad del anillo v = ring_k", fontsize=15)
    ax.set_ylabel("Intensidad de acoplamiento K", fontsize=15)
    ax.set_title(title, fontsize=17)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label, fontsize=13)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                label = f"{val:.2g}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    grouped = load_grouped(args)
    rows = build_summary(args, grouped)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ring_k_K_heatmap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    k_vals, ring_vals, stat_mat = matrix_from_rows(rows, "stationary_sigma_mean")
    _, _, sync_mat = matrix_from_rows(rows, "sync_time_mean")

    plot_heatmap(k_vals, ring_vals, stat_mat, "Red anillo: dispersion estacionaria sigma_v(K, v)", "Sigma_v estacionaria", out_dir / "stationary_sigma_heatmap_ring_k_K.png")
    plot_heatmap(k_vals, ring_vals, sync_mat, "Red anillo: tiempo de sincronizacion t_sync(K, v)", "Tiempo de sincronizacion", out_dir / "sync_time_heatmap_ring_k_K.png")

    print(f"Analyzed {sum(len(v) for v in grouped.values())} runs in {len(rows)} grid cells.")
    print(f"Outputs written to: {out_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
