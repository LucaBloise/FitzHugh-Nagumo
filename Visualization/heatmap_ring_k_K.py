#!/usr/bin/env python3
"""
2D ring-network maps for FitzHugh-Nagumo simulations.

Covers the TP5 Sistema 2 / Red anillo points:
- stationary spatial dispersion sigma_v as function of K and ring_k
- arrival time to stationary regime as function of K and ring_k

It also keeps t_sync by absolute sigma threshold as an additional diagnostic.
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
        description="Build 2D heatmaps sigma_v(K, ring_k) and stationary-time(K, ring_k)."
    )
    parser.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR,
                        help="Directory with Java .txt simulation outputs")
    parser.add_argument("--pattern", default="*ring*.txt",
                        help="Glob pattern inside input-dir")
    parser.add_argument("--k-values", type=float, nargs="*", default=None,
                        help="Optional K values to include")
    parser.add_argument("--ring-k-values", type=int, nargs="*", default=None,
                        help="Optional ring_k values to include")
    parser.add_argument("--k-tol", type=float, default=1e-12,
                        help="Tolerance for matching requested K values")
    parser.add_argument("--stationary-fraction", type=float, default=0.2,
                        help="Final fraction used for stationary averages")
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
                        help="Value if a run never reaches the synchronization threshold")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Directory where plots/CSV are written")
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
    realization = 1
    network: str | None = None
    times: list[float] = []
    means: list[float] = []
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
        raise ValueError(f"Missing K/ring_k in {path}")
    if not times:
        raise ValueError(f"No states found in {path}")

    order = np.argsort(np.asarray(times, dtype=float))
    return RunData(
        path=path,
        k=k_value,
        ring_k=ring_k,
        realization=realization,
        times=np.asarray(times, dtype=float)[order],
        mean_v=np.asarray(means, dtype=float)[order],
        sigma_v=np.asarray(sigmas, dtype=float)[order],
    )


def stationary_value(values: np.ndarray, fraction: float) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("stationary-fraction must be in (0, 1]")
    start = int(math.floor((1.0 - fraction) * len(values)))
    start = max(0, min(start, len(values) - 1))
    return float(np.mean(values[start:]))


def find_t_sync(times: np.ndarray, sigma: np.ndarray, threshold: float, window: int, fill: str) -> float:
    below = sigma < threshold
    window = max(1, int(window))
    if window > len(below):
        return float(times[-1]) if fill == "tmax" else float("nan")
    for i in range(0, len(below) - window + 1):
        if bool(np.all(below[i:i + window])):
            return float(times[i])
    return float(times[-1]) if fill == "tmax" else float("nan")


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


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _finite_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return 0.0
    return float(np.nanstd(finite, ddof=1))


def _k_allowed(k: float, wanted: list[float] | None, tol: float) -> bool:
    if wanted is None:
        return True
    return any(abs(k - target) <= tol for target in wanted)


def load_grouped(args: argparse.Namespace) -> dict[tuple[float, int], list[RunData]]:
    input_dir = _resolve(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} in {input_dir}")

    wanted_k = list(args.k_values) if args.k_values else None
    wanted_ring = set(args.ring_k_values) if args.ring_k_values else None
    grouped: dict[tuple[float, int], list[RunData]] = defaultdict(list)
    skipped = 0

    for path in files:
        try:
            run = parse_run(path)
        except Exception:
            skipped += 1
            continue
        if not _k_allowed(run.k, wanted_k, args.k_tol):
            continue
        if wanted_ring is not None and run.ring_k not in wanted_ring:
            continue
        grouped[(run.k, run.ring_k)].append(run)

    if not grouped:
        raise SystemExit("No RING runs matched the selected filters.")
    if skipped:
        print(f"Warning: skipped {skipped} non-RING/unreadable files.")
    return dict(grouped)


def build_summary(args: argparse.Namespace, grouped: dict[tuple[float, int], list[RunData]]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for (k, ring_k), runs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        stat_sigma = [stationary_value(run.sigma_v, args.stationary_fraction) for run in runs]
        stat_mean_v = [stationary_value(run.mean_v, args.stationary_fraction) for run in runs]
        sync = [find_t_sync(run.times, run.sigma_v, args.sigma_threshold, args.sync_window, args.sync_fill) for run in runs]
        stationary = [find_t_stationary(run.times, run.sigma_v, args.stationary_window, args.stationary_epsilon, args.stationary_fill) for run in runs]
        rows.append({
            "K": k,
            "ring_k": ring_k,
            "degree_total": 2 * ring_k,
            "n_reps": len(runs),
            "stationary_fraction": args.stationary_fraction,
            "stationary_window": args.stationary_window,
            "stationary_epsilon": args.stationary_epsilon,
            "sigma_threshold": args.sigma_threshold,
            "sync_window": args.sync_window,
            "stationary_mean_v_mean": _finite_mean(stat_mean_v),
            "stationary_mean_v_std": _finite_std(stat_mean_v),
            "stationary_sigma_mean": _finite_mean(stat_sigma),
            "stationary_sigma_std": _finite_std(stat_sigma),
            "stationary_time_mean": _finite_mean(stationary),
            "stationary_time_std": _finite_std(stationary),
            "sync_time_mean": _finite_mean(sync),
            "sync_time_std": _finite_std(sync),
        })
    return rows


def matrix_from_rows(rows: list[dict[str, float | int]], value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_vals = np.asarray(sorted({float(row["K"]) for row in rows}), dtype=float)
    ring_vals = np.asarray(sorted({int(row["ring_k"]) for row in rows}), dtype=int)
    mat = np.full((len(k_vals), len(ring_vals)), np.nan, dtype=float)
    k_index = {k: i for i, k in enumerate(k_vals)}
    r_index = {r: j for j, r in enumerate(ring_vals)}
    for row in rows:
        mat[k_index[float(row["K"])], r_index[int(row["ring_k"])]] = float(row[value_key])
    return k_vals, ring_vals, mat


def plot_heatmap(k_vals: np.ndarray, ring_vals: np.ndarray, mat: np.ndarray,
                 cbar_label: str, out_path: Path,
                 cmap: str = "viridis",
                 annotate: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7), facecolor="white")
    masked = np.ma.masked_invalid(mat)
    image = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap)

    ax.set_xticks(np.arange(len(ring_vals)))
    ax.set_xticklabels([str(v) for v in ring_vals])
    ax.set_yticks(np.arange(len(k_vals)))
    ax.set_yticklabels([format(v, "g") for v in k_vals])
    ax.set_xlabel(r"Vecinos por lado $k$", fontsize=15)
    ax.set_ylabel(r"Intensidad de acoplamiento $K$", fontsize=15)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label, fontsize=13)

    if annotate:
        finite_vals = mat[np.isfinite(mat)]
        midpoint = float(np.nanmedian(finite_vals)) if finite_vals.size else 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    label = f"{val:.2g}"
                    color = "white" if val > midpoint else "black"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

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
    _, _, stationary_time_mat = matrix_from_rows(rows, "stationary_time_mean")
    _, _, sync_mat = matrix_from_rows(rows, "sync_time_mean")
    _, _, mean_v_mat = matrix_from_rows(rows, "stationary_mean_v_mean")

    plot_heatmap(
        k_vals,
        ring_vals,
        stat_mat,
        r"Dispersión espacial promedio $\langle \sigma_{v} \rangle_{est}$",
        out_dir / "stationary_sigma_heatmap_ring_k_K.png",
    )
    plot_heatmap(
        k_vals,
        ring_vals,
        stationary_time_mat,
        r"Tiempo de llegada al estacionario promedio $t_{est}$ (s)",
        out_dir / "stationary_time_heatmap_ring_k_K.png",
    )
    plot_heatmap(
        k_vals,
        ring_vals,
        sync_mat,
        r"Tiempo de sincronización promedio $t_{sync}$ (s)",
        out_dir / "sync_time_heatmap_ring_k_K.png",
        cmap="inferno_r",
        annotate=False,
    )
    plot_heatmap(
        k_vals,
        ring_vals,
        mean_v_mat,
        r"Potencial promedio estacionario $\langle v \rangle_{est}$",
        out_dir / "stationary_mean_v_heatmap_ring_k_K.png",
    )

    print(f"Analyzed {sum(len(v) for v in grouped.values())} runs in {len(rows)} grid cells.")
    print(f"Outputs written to: {out_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
