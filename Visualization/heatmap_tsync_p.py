#!/usr/bin/env python3
"""
Genera un Heatmap 2D del tiempo de sincronización promedio en función de p y K.
Incluye un sistema de caché y corta la lectura de cada archivo de forma temprana
apenas se alcanza el umbral de sincronización para maximizar el rendimiento.
"""

import os
import sys
import glob
import concurrent.futures
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent / "Simulation" / "output"
CACHE_FILE = SCRIPT_DIR / "cache_tsync_2d.csv"

# Umbral de dispersión espacial para considerar que el sistema está sincronizado.
THRESHOLD = 0.001 


def parse_single_file_tsync(filepath: Path) -> tuple[float, float, float] | None:
    k_val = None
    p_val = None
    t_sync = None
    last_t = 0.0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("K="):
                    k_val = float(line.split("=")[1])
                elif line.startswith("p="):
                    p_val = float(line.split("=")[1])
                elif line.startswith("BEGIN_STATE t="):
                    last_t = float(line[14:])
                elif line.startswith("v "):
                    # Evaluamos la dispersión espacial
                    v_vals = np.fromstring(line[2:], sep=" ")
                    if v_vals.size > 0:
                        sigma = v_vals.std()
                        
                        # Si cruza el umbral, registramos el tiempo y CORTAMOS la lectura
                        if sigma <= THRESHOLD:
                            t_sync = last_t
                            if k_val is not None and p_val is not None:
                                break # Optimización extrema: no seguimos leyendo el archivo

        if k_val is None or p_val is None:
            return None

        # Si el archivo terminó y nunca se sincronizó, asignamos el último tiempo simulado
        if t_sync is None:
            t_sync = last_t

        return k_val, p_val, t_sync

    except Exception as e:
        print(f"\n[ERROR] Al procesar {filepath.name}: {e}")
        return None


def main():
    if CACHE_FILE.exists():
        print(f"Cargando datos desde caché: {CACHE_FILE.name}")
        df_grid = pd.read_csv(CACHE_FILE)
    else:
        print(f"No se encontró caché. Buscando archivos en {INPUT_DIR}...")
        files = glob.glob(os.path.join(INPUT_DIR, "fhn_random_*.txt"))
        
        if not files:
            print(f"[ERROR] No se encontraron archivos en {INPUT_DIR}")
            sys.exit(1)
            
        print(f"Se encontraron {len(files)} archivos. Iniciando procesamiento en paralelo...")

        raw_data = []
        processed_count = 0

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(parse_single_file_tsync, Path(f)): f for f in files}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    raw_data.append(result)
                
                processed_count += 1
                if processed_count % 50 == 0 or processed_count == len(files):
                    print(f"Progreso: {processed_count}/{len(files)} archivos procesados...", end="\r")
        
        print("\nProcesamiento completado.")

        df_raw = pd.DataFrame(raw_data, columns=["K", "p", "t_sync"])
        
        print("Promediando realizaciones independientes...")
        df_grid = df_raw.groupby(["K", "p"])["t_sync"].mean().reset_index()
        
        df_grid.to_csv(CACHE_FILE, index=False)
        print(f"Caché guardado en: {CACHE_FILE}")

    pivot_df = df_grid.pivot(index="K", columns="p", values="t_sync")
    pivot_df = pivot_df.sort_index(ascending=True)  
    pivot_df = pivot_df.sort_index(axis=1, ascending=True) 

    X, Y = np.meshgrid(pivot_df.columns.values, pivot_df.index.values)
    Z = pivot_df.values

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # cmap="magma_r" (invertido) es bueno para tiempos: colores oscuros = mucho tiempo, claros = rápido
    mesh = ax.pcolormesh(X, Y, Z, cmap="magma_r", shading="nearest")
    
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"Tiempo de sincronización promedio $t_{sync}$ (s)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xticks(pivot_df.columns.values)
    ax.set_yticks(pivot_df.index.values)
    
    ax.set_xticklabels([f"{p:g}" for p in pivot_df.columns.values], rotation=45, fontsize=11)
    ax.set_yticklabels([f"{k:g}" for k in pivot_df.index.values], fontsize=11)

    ax.set_xlabel("Probabilidad de conexión $p$", fontsize=14, labelpad=10)
    ax.set_ylabel("Intensidad de acoplamiento $K$", fontsize=14, labelpad=10)
    
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()