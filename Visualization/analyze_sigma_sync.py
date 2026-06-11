#!/usr/bin/env python3
"""
Procesa sigma_v_cache.csv y genera:
1) Tiempo de sincronizacion vs K
2) Sigma estacionaria vs K

Outputs:
- sync_time_vs_K.png
- stationary_sigma_vs_K.png
- stationary_sigma_vs_K.csv

El script admite dos formatos de entrada:
A) Por realizacion (recomendado): columnas
   k,time,realization,sigma_v
El calculo de t_sync se hace por realizacion y luego se promedia por K.
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
_DEFAULT_INPUT_CSV = _PROJECT_ROOT / "Visualization" / "sigma_v_cache.csv"
_DEFAULT_INPUT_DIR = _PROJECT_ROOT / "Simulation" / "output"
_BEGIN_STATE_RE = re.compile(r"^BEGIN_STATE t=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")
_REALIZATION_RE = re.compile(r"(?:^|[_-])real(?:ization)?(\d+)(?:[_\.-]|$)", re.IGNORECASE)


@dataclass
class RunSeries:
    k: float
    realization: str
    times: np.ndarray
    sigma: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcular tiempo de sincronizacion y sigma estacionaria desde sigma_v_cache.csv",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_INPUT_CSV,
        help=f"CSV de entrada (default: {_DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help=(
            "Directorio con outputs .txt para reconstruir realizaciones si el CSV "
            "es agregado (default: Simulation/output)."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="fhn_*.txt",
        help="Patron de archivos para reconstruir realizaciones desde input-dir (default: fhn_*.txt)",
    )
    parser.add_argument(
        "--sigma-threshold",
        type=float,
        default=0.00000001,
        help="Umbral de sincronizacion sigma_v(t) < threshold (default: 0.005)",
    )
    parser.add_argument(
        "--sync-fill",
        choices=["nan", "tmax"],
        default="nan",
        help="Valor para t_sync si no sincroniza: nan o tmax (default: nan)",
    )
    parser.add_argument(
        "--out-sync",
        type=Path,
        default=_PROJECT_ROOT / "Visualization" / "sync_time_vs_K.png",
        help="Salida PNG grafico t_sync vs K",
    )
    parser.add_argument(
        "--out-stationary",
        type=Path,
        default=_PROJECT_ROOT / "Visualization" / "stationary_sigma_vs_K.png",
        help="Salida PNG grafico sigma estacionaria vs K",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=_PROJECT_ROOT / "Visualization" / "stationary_sigma_vs_K.csv",
        help="Salida CSV final",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar graficos ademas de guardar.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def _group_sorted(times: list[float], values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    idx = np.argsort(t)
    return t[idx], v[idx]


def _parse_realization_name(path: Path) -> str:
    match = _REALIZATION_RE.search(path.name)
    if match:
        return f"real{match.group(1)}"
    return path.stem


def _parse_output_sigma_run(path: Path) -> RunSeries:
    k_value: float | None = None
    times: list[float] = []
    sigma_v: list[float] = []
    current_t: float | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("K="):
                k_value = float(line.split("=", 1)[1])
                continue

            match = _BEGIN_STATE_RE.match(line)
            if match:
                current_t = float(match.group(1))
                continue

            if line.startswith("v ") and current_t is not None:
                v_vals = np.fromstring(line[2:], sep=" ")
                if v_vals.size == 0:
                    raise ValueError(f"No se pudieron parsear valores v en {path}")
                times.append(current_t)
                sigma_v.append(float(v_vals.std()))
                current_t = None

    if k_value is None:
        raise ValueError(f"No se encontro K en header: {path}")
    if not times:
        raise ValueError(f"No se encontraron estados en archivo: {path}")

    t_arr, s_arr = _group_sorted(times, sigma_v)
    return RunSeries(
        k=float(k_value),
        realization=_parse_realization_name(path),
        times=t_arr,
        sigma=s_arr,
    )


def _load_runs_from_outputs(input_dir: Path, pattern: str) -> dict[float, list[RunSeries]]:
    if not input_dir.exists():
        raise SystemExit(f"No existe input-dir para reconstruir realizaciones: {input_dir}")

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No hay archivos '{pattern}' en {input_dir}")

    runs_by_k: dict[float, list[RunSeries]] = defaultdict(list)
    skipped = 0
    for path in files:
        try:
            run = _parse_output_sigma_run(path)
        except Exception:
            skipped += 1
            continue
        runs_by_k[run.k].append(run)

    if not runs_by_k:
        raise SystemExit("No se pudieron reconstruir realizaciones validas desde outputs .txt")

    if skipped:
        print(f"Aviso: se omitieron {skipped} archivo(s) al reconstruir realizaciones desde outputs.")

    return dict(runs_by_k)


def _load_runs(csv_path: Path, input_dir: Path, pattern: str) -> tuple[dict[float, list[RunSeries]], str]:
    """
    Returns
    -------
    runs_by_k: dict[k] -> list[RunSeries]
    """
    if not csv_path.exists():
        raise SystemExit(f"CSV no encontrado: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])

        run_level_fields = {"k", "time", "realization", "sigma_v"}
        if run_level_fields.issubset(fields):
            grouped: dict[tuple[float, str], tuple[list[float], list[float]]] = {}
            for row in reader:
                k = float(row["k"])
                realization = str(row["realization"])
                t = float(row["time"])
                s = float(row["sigma_v"])
                key = (k, realization)
                if key not in grouped:
                    grouped[key] = ([], [])
                grouped[key][0].append(t)
                grouped[key][1].append(s)

            runs_by_k: dict[float, list[RunSeries]] = defaultdict(list)
            for (k, realization), (times, sigma) in grouped.items():
                t_arr, s_arr = _group_sorted(times, sigma)
                runs_by_k[k].append(RunSeries(k=k, realization=realization, times=t_arr, sigma=s_arr))
            return dict(runs_by_k), "csv-realizations"

        aggregated_fields = {"k", "time", "mean_sigma_v", "std_sigma_v", "n_reps"}
        if aggregated_fields.issubset(fields):
            print(
                "Aviso: el CSV es agregado por K (sin realization). "
                "Se reconstruiran realizaciones parseando Simulation/output."
            )
            return _load_runs_from_outputs(input_dir, pattern), "outputs-reconstructed"

        raise SystemExit(
            "Formato CSV no reconocido. Se requiere [k,time,realization,sigma_v] "
            "o un CSV agregado compatible para disparar reconstruccion desde outputs."
        )


def _find_t_sync(times: np.ndarray, sigma: np.ndarray, threshold: float, sync_fill: str) -> float:
    if len(times) == 0:
        return float("nan")

    t_max = float(times[-1])

    above = sigma > threshold

    if not np.any(above):
        return 0.0

    if above[-1]:
        if sync_fill == "tmax":
            return t_max
        return float("nan")

    last_above_idx = int(np.where(above)[0][-1])
    return float(times[last_above_idx + 1])

    if sync_fill == "tmax":
        return t_max
    return float("nan")


def _stationary_mean_after_cutoff(times: np.ndarray, sigma: np.ndarray, t_stationary: float) -> float:
    mask = times >= t_stationary
    if not np.any(mask):
        return float("nan")
    return float(np.mean(sigma[mask]))


def _mean_curve_by_k(runs_by_k: dict[float, list[RunSeries]]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Build mean sigma(t, K) by averaging realizations at each sampled time.

    If runs for a given K have different lengths, all runs are truncated to the
    minimum length for that K to keep aligned samples.
    """
    mean_by_k: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for k, runs in runs_by_k.items():
        if not runs:
            continue

        min_len = min(len(run.times) for run in runs)
        if min_len == 0:
            continue

        times_ref = runs[0].times[:min_len]
        stacked = np.vstack([run.sigma[:min_len] for run in runs])
        mean_sigma = stacked.mean(axis=0)
        mean_by_k[k] = (times_ref, mean_sigma)

    return mean_by_k


def _safe_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan") if arr.size == 0 else 0.0
    return float(arr.std(ddof=1))


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _plot_sync_vs_k(k_vals: np.ndarray, sync_mean: np.ndarray, sync_std: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    pos_mask = k_vals > 0
    zero_mask = k_vals == 0

    if np.any(pos_mask):
        ax.errorbar(
            k_vals[pos_mask],
            sync_mean[pos_mask],
            yerr=sync_std[pos_mask],
            fmt="o-",
            linewidth=2.5,
            markersize=8,
            capsize=4,
        )
        ax.set_xscale("log")

    if np.any(zero_mask):
        ax.errorbar(
            k_vals[zero_mask],
            sync_mean[zero_mask],
            yerr=sync_std[zero_mask],
            fmt="s",
            linewidth=2.5,
            markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Intensidad de acoplamiento K", fontsize=22)
    ax.set_ylabel("Tiempo de sincronizacion (s)", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")


def _plot_stationary_vs_k(
    k_vals: np.ndarray,
    sigma_mean: np.ndarray,
    sigma_std: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    pos_mask = k_vals > 0
    zero_mask = k_vals == 0

    if np.any(pos_mask):
        ax.errorbar(
            k_vals[pos_mask],
            sigma_mean[pos_mask],
            yerr=sigma_std[pos_mask],
            fmt="o-",
            linewidth=2.5,
            markersize=8,
            capsize=4,
        )
        ax.set_xscale("log")

    if np.any(zero_mask):
        ax.errorbar(
            k_vals[zero_mask],
            sigma_mean[zero_mask],
            yerr=sigma_std[zero_mask],
            fmt="s",
            linewidth=2.5,
            markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Intensidad de acoplamiento K", fontsize=22)
    ax.set_ylabel("Dispersion espacial estacionaria", fontsize=22)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")


def _write_output_csv(
    out_path: Path,
    k_vals: np.ndarray,
    sync_mean: np.ndarray,
    sync_std: np.ndarray,
    sigma_mean: np.ndarray,
    sigma_std: np.ndarray,
    n_reals: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "K",
                "sync_time_mean",
                "sync_time_std",
                "sigma_stationary_mean",
                "sigma_stationary_std",
                "n_realizations",
            ]
        )

        for i in range(len(k_vals)):
            writer.writerow(
                [
                    f"{float(k_vals[i]):.17g}",
                    f"{float(sync_mean[i]):.17g}",
                    f"{float(sync_std[i]):.17g}",
                    f"{float(sigma_mean[i]):.17g}",
                    f"{float(sigma_std[i]):.17g}",
                    str(int(n_reals[i])),
                ]
            )


def main() -> None:
    args = parse_args()

    if args.sigma_threshold < 0:
        raise SystemExit("--sigma-threshold debe ser >= 0")
    input_csv = _resolve_path(args.csv)
    input_dir = _resolve_path(args.input_dir)
    out_sync = _resolve_path(args.out_sync)
    out_stationary = _resolve_path(args.out_stationary)
    out_csv = _resolve_path(args.out_csv)

    runs_by_k, data_source = _load_runs(input_csv, input_dir, args.pattern)
    if not runs_by_k:
        raise SystemExit("No se encontraron datos validos en el CSV.")

    # t_sync por realizacion y K
    t_sync_by_k: dict[float, list[float]] = defaultdict(list)
    t_stationary_candidates: list[float] = []

    for k, runs in runs_by_k.items():
        for run in runs:
            t_sync = _find_t_sync(
                times=run.times,
                sigma=run.sigma,
                threshold=args.sigma_threshold,
                sync_fill=args.sync_fill,
            )
            t_sync_by_k[k].append(t_sync)
            if math.isfinite(t_sync):
                t_stationary_candidates.append(t_sync)

    if t_stationary_candidates:
        t_stationary = float(np.max(np.asarray(t_stationary_candidates, dtype=float)))
    else:
        # Si ningun K sincroniza, usar t_max global para definir cola estacionaria.
        t_max_global = max(float(run.times[-1]) for runs in runs_by_k.values() for run in runs if len(run.times) > 0)
        t_stationary = t_max_global

    # sigma estacionaria desde la curva promedio mean_sigma(t, K)
    mean_curve_by_k = _mean_curve_by_k(runs_by_k)

    sorted_k = np.asarray(sorted(runs_by_k), dtype=float)
    sync_mean = []
    sync_std = []
    sigma_mean = []
    sigma_std = []
    n_realizations = []

    for k in sorted_k:
        sync_vals = t_sync_by_k[float(k)]
        times_k, mean_sigma_k = mean_curve_by_k[float(k)]
        mask_stationary = times_k >= t_stationary
        stationary_segment = mean_sigma_k[mask_stationary]

        sync_mean.append(_safe_mean(sync_vals))
        sync_std.append(_safe_std(sync_vals))
        if stationary_segment.size == 0:
            sigma_mean.append(float("nan"))
            sigma_std.append(float("nan"))
        else:
            sigma_mean.append(float(np.mean(stationary_segment)))
            sigma_std.append(float(np.std(stationary_segment, ddof=1)) if stationary_segment.size > 1 else 0.0)
        n_realizations.append(len(runs_by_k[float(k)]))

    sync_mean_arr = np.asarray(sync_mean, dtype=float)
    sync_std_arr = np.asarray(sync_std, dtype=float)
    sigma_mean_arr = np.asarray(sigma_mean, dtype=float)
    sigma_std_arr = np.asarray(sigma_std, dtype=float)
    n_real_arr = np.asarray(n_realizations, dtype=int)

    _plot_sync_vs_k(sorted_k, sync_mean_arr, sync_std_arr, out_sync)
    _plot_stationary_vs_k(sorted_k, sigma_mean_arr, sigma_std_arr, out_stationary)
    _write_output_csv(
        out_path=out_csv,
        k_vals=sorted_k,
        sync_mean=sync_mean_arr,
        sync_std=sync_std_arr,
        sigma_mean=sigma_mean_arr,
        sigma_std=sigma_std_arr,
        n_reals=n_real_arr,
    )

    print(f"CSV entrada: {input_csv}")
    print(f"Fuente de datos por realizacion: {data_source}")
    print(f"sigma_threshold = {args.sigma_threshold:g}")
    print(f"t_stationary global = {t_stationary:.6g} s")
    print(f"Grafico 1 guardado en: {out_sync}")
    print(f"Grafico 2 guardado en: {out_stationary}")
    print(f"CSV final guardado en: {out_csv}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
