#!/usr/bin/env python3
"""
Grafica el potencial promedio <v> vs probabilidad de conexión p.
Corta el transitorio (t < t_est), promedia en el tiempo para cada realización,
y luego promedia entre realizaciones graficando barras de error (desvío estándar).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INPUT_DIR = _PROJECT_ROOT / "Simulation" / "output"

_BEGIN_STATE_RE = re.compile(r"^BEGIN_STATE t=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot <v> vs p con barras de error.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help=f"Directorio con archivos .txt (default: {_DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        default="fhn_random_*.txt",
        help="Patrón para agarrar los archivos de redes aleatorias",
    )
    parser.add_argument(
        "--t-est",
        type=float,
        default=200.0,
        help="Tiempo a partir del cual se considera régimen estacionario (default: 200s)",
    )
    return parser.parse_args()


def _parse_file_stationary_v(path: Path, t_est: float) -> tuple[float, float]:
    """
    Lee un archivo txt, descarta t < t_est.
    Retorna el valor de 'p' y el promedio temporal del potencial espacial promedio.
    """
    p_value: float | None = None
    v_means_temporal: list[float] = []
    current_t: float | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("p="):
                p_value = float(line.split("=", 1)[1])
                continue

            match = _BEGIN_STATE_RE.match(line)
            if match:
                t_val = float(match.group(1))
                if t_val >= t_est:
                    current_t = t_val
                else:
                    current_t = None
                continue

            if line.startswith("v ") and current_t is not None:
                v_vals = np.fromstring(line[2:], sep=" ")
                if v_vals.size > 0:
                    v_means_temporal.append(float(v_vals.mean()))
                current_t = None

    if p_value is None:
        raise ValueError(f"p no encontrado en: {path}")
    if not v_means_temporal:
        raise ValueError(f"No se encontraron datos despues de t={t_est}s en: {path}")

    return p_value, float(np.mean(v_means_temporal))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.exists():
        raise SystemExit(f"No existe la carpeta: {input_dir}")

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No hay archivos que cumplan '{args.pattern}' en {input_dir}")

    print(f"Procesando {len(files)} archivos con corte en t >= {args.t_est} s...")

    by_p: dict[float, list[float]] = {}
    parsed = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_parse_file_stationary_v, path, args.t_est): path 
            for path in files
        }
        
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                p_value, stationary_mean = future.result()
                by_p.setdefault(p_value, []).append(stationary_mean)
                parsed += 1
            except Exception as e:
                print(f"Error procesando {path.name}: {e}")

    if not by_p:
        raise SystemExit("No se pudo parsear ningún dato válido.")

    print(f"Archivos procesados correctamente: {parsed}")

    p_keys = sorted(by_p.keys())
    means = np.array([np.mean(by_p[p]) for p in p_keys])
    stds = np.array([np.std(by_p[p]) for p in p_keys])

    fig, ax = plt.subplots(figsize=(9, 6))

    # Quitamos el label para que no genere leyenda
    ax.errorbar(
        p_keys, 
        means, 
        yerr=stds, 
        fmt="-o", 
        color="#2c3e50",
        ecolor="#e74c3c", 
        elinewidth=2, 
        capsize=5, 
        capthick=2, 
        markersize=8,
        linewidth=2
    )

    ax.set_xscale("log")
    
    # Forzamos a que aparezcan los 10 valores de p en el eje X
    ax.set_xticks(p_keys)
    # Formateamos los números para que no salgan en notación científica rara y los rotamos
    ax.set_xticklabels([f"{p:g}" for p in p_keys], rotation=45)
    
    ax.set_xlabel("Probabilidad de conexión $p$", fontsize=16)
    # Cambiamos el nombre del eje Y
    ax.set_ylabel(r"Potencial promedio $\langle v \rangle$", fontsize=16)
    
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Borramos la llamada a ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()