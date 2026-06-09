#!/usr/bin/env python3
"""
animate_fhn.py  –  Animador de la simulación FitzHugh-Nagumo
============================================================
Lee un archivo de estado crudo producido por el simulador FHN en Java y renderiza
una animación apropiada para cada tipo de red:

  FULL   – disposición circular de nodos, nodos coloreados por potencial de membrana v_i
  RANDOM – grafo con diseño de resortes (spring-layout), nodos coloreados por v_i, aristas dibujadas
  RING   – nodos dispuestos en un anillo, coloreados por v_i, aristas dibujadas

Uso
---
    python animate_fhn.py                              # elige el archivo más nuevo en el dir de salida por defecto
    python animate_fhn.py ruta/al/fhn_*.txt            # archivo explícito
    python animate_fhn.py --output-dir /mi/salida      # directorio de salida diferente
    python animate_fhn.py --fps 30 --dpi 100           # ajustar renderizado
    python animate_fhn.py --save anim.mp4              # guardar en lugar de mostrar
    python animate_fhn.py --frames 200                 # limitar cuadros (vista previa más rápida)
    python animate_fhn.py --skip 2                     # usar 1 de cada 2 cuadros guardados
"""

import argparse
import sys
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Rutas por defecto (refleja la lógica de OutputFileName / SimulationPaths de Java)
# La raíz del proyecto Java está dos niveles arriba de Visualization/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # …/Visualization
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # raíz del proyecto
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "Simulation" / "output"


# ---------------------------------------------------------------------------
# Analizador (Parser)
# ---------------------------------------------------------------------------
def parse_header_line(line: str, key: str):
    """Retorna la cadena de valor para líneas de encabezado 'clave=valor', o None."""
    if line.startswith(key + "="):
        return line[len(key) + 1:].strip()
    return None


def parse_output_file(path: Path):
    """
    Analiza un archivo FHN_SIMULATION_OUTPUT v2.

    Retorna
    -------
    meta   : dict  – todos los pares clave=valor del encabezado (cadenas)
    edges  : list  – lista de tuplas (i, j), vacía para FULL
    times  : np.ndarray de forma (T,)
    vs     : np.ndarray de forma (T, N)
    ws     : np.ndarray de forma (T, N)
    """
    meta = {}
    edges = []
    times = []
    vs = []
    ws = []

    with open(path, "r") as fh:
        in_states = False
        cur_t = None
        cur_v = None

        for raw in fh:
            line = raw.rstrip("\n")

            # ---- sección de encabezado ----
            if not in_states:
                if line.startswith("#") or line == "":
                    if line.startswith("EDGE "):
                        parts = line.split()
                        edges.append((int(parts[1]), int(parts[2])))
                    continue
                if line.startswith("EDGE "):
                    parts = line.split()
                    edges.append((int(parts[1]), int(parts[2])))
                    continue
                if line == "BEGIN_STATE_SAMPLES":
                    in_states = True
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    meta[k.strip()] = v.strip()
                continue

            # ---- sección de estado ----
            if line.startswith("BEGIN_STATE t="):
                cur_t = float(line.split("=")[1])
                cur_v = None
                continue

            if line.startswith("v "):
                cur_v = np.fromstring(line[2:], sep=" ")
                continue

            if line.startswith("w "):
                if cur_v is not None and cur_t is not None:
                    cur_w = np.fromstring(line[2:], sep=" ")
                    times.append(cur_t)
                    vs.append(cur_v)
                    ws.append(cur_w)
                cur_t = None
                cur_v = None
                continue

            if line == "END_STATE_SAMPLES":
                break

    return meta, edges, np.array(times), np.array(vs), np.array(ws)


# ---------------------------------------------------------------------------
# Funciones auxiliares de posición de nodos
# ---------------------------------------------------------------------------
def circular_positions(n: int):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def spring_positions(n: int, edges, seed: int = 0):
    """Diseño simple dirigido por fuerzas (sin dependencia de networkx)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))

    if not edges:
        return circular_positions(n)

    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    k = np.sqrt(1.0 / max(n, 1))
    iters = max(20, min(100, 2000 // n))  # menos iteraciones para grafos grandes
    for _ in range(iters):
        disp = np.zeros((n, 2))
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist = np.maximum(dist, 1e-6)
            repulsion = (delta.T * (k * k / dist**2)).T
            repulsion[i] = 0
            disp[i] += repulsion.sum(axis=0)
        for i, j in edges:
            delta = pos[i] - pos[j]
            d = max(np.linalg.norm(delta), 1e-6)
            force = delta * d / k
            disp[i] -= force
            disp[j] += force
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        pos += disp / norms * np.minimum(norms, 0.05).reshape(-1, 1)

    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    if scale > 0:
        pos /= scale
    return pos


# ---------------------------------------------------------------------------
# Constructores de animación
# ---------------------------------------------------------------------------
CMAP = "coolwarm"   # azul (hiperpolarizado) → rojo (despolarizado)

def _edge_segments(pos, edges):
    return [np.array([pos[i], pos[j]]) for i, j in edges]


def build_animation_full(times, vs, ws, meta, fps, max_frames, skip):
    """
    Red FULL: diseño circular, nodos coloreados por v_i.
    No se dibujan aristas (N≫1 – sería ilegible).
    """
    N = vs.shape[1]
    frames_idx = range(0, len(times), skip)
    if max_frames:
        frames_idx = list(frames_idx)[:max_frames]
    frames_idx = list(frames_idx)

    pos = circular_positions(N)

    v_all = vs[frames_idx]
    vmin, vmax = v_all.min(), v_all.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(7, 7), facecolor="#0d0d0d")
    ax_net = fig.add_axes([0.05, 0.05, 0.85, 0.82], facecolor="#0d0d0d")

    ax_net.set_xlim(-1.15, 1.15)
    ax_net.set_ylim(-1.15, 1.15)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    # Dibujar aristas solo para redes pequeñas
    if N <= 30:
        full_edges = []
        for i in range(N):
            for j in range(i + 1, N):
                full_edges.append((i, j))

        segs = _edge_segments(pos, full_edges)
        lc = LineCollection(
            segs,
            colors="#2a2a2a",
            linewidths=0.5,
            zorder=1
        )
        ax_net.add_collection(lc)

    sc = ax_net.scatter(
        pos[:, 0], pos[:, 1],
        c=vs[frames_idx[0]], cmap=CMAP, norm=norm,
        s=max(2, 400 // N), zorder=3, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax_net, fraction=0.04, pad=0.02)
    cbar.set_label("Potencial de membrana $v_i$", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Título principal (estático)
    fig.text(0.5, 0.97, "FitzHugh–Nagumo  |  Red Totalmente Conectada",
             color="white", fontsize=12, ha="center", va="top")

    # Subtítulo de la red (parámetros estáticos)
    K_val = float(meta['K']) if 'K' in meta else None
    K_str = f"{K_val:g}" if K_val is not None else "?"
    fig.text(0.5, 0.93, f"N={N}, K={K_str}",
             color="#aaaaaa", fontsize=10, ha="center", va="top")

    # Visualización del tiempo (dinámico) — prominente, arriba al centro
    time_text = fig.text(
        0.5, 0.89, f"t = {times[frames_idx[0]]:.2f} s",
        color="#ffd54f", fontsize=13, fontweight="bold",
        ha="center", va="top", family="monospace",
    )

    def init():
        return (sc, time_text)

    def update(frame_num):
        fi = frames_idx[frame_num]
        sc.set_array(vs[fi])
        time_text.set_text(f"t = {times[fi]:.2f} s")
        return (sc, time_text)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_idx), init_func=init,
        interval=1000 / fps, blit=False,
    )
    return fig, ani


def build_animation_random(times, vs, ws, meta, edges, fps, max_frames, skip):
    """
    Red RANDOM: diseño de resortes, nodos coloreados por v_i, aristas visibles.
    """
    N = vs.shape[1]
    frames_idx = list(range(0, len(times), skip))
    if max_frames:
        frames_idx = frames_idx[:max_frames]

    seed = int(meta.get("seed", 0))
    pos = spring_positions(N, edges, seed=seed)

    v_all = vs[np.array(frames_idx)]
    vmin, vmax = v_all.min(), v_all.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(7, 7), facecolor="#0d0d0d")
    ax_net = fig.add_axes([0.05, 0.05, 0.85, 0.82], facecolor="#0d0d0d")

    ax_net.set_xlim(-1.2, 1.2)
    ax_net.set_ylim(-1.2, 1.2)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    # Límite de renderizado de aristas: más allá de ~2000 aristas se convierten
    # en ruido visual y dominan el tiempo de renderizado. Muestrea un subconjunto.
    MAX_DRAWN_EDGES = 2000
    edge_note = ""
    if edges:
        drawn_edges = edges
        if len(edges) > MAX_DRAWN_EDGES:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(edges), MAX_DRAWN_EDGES, replace=False)
            drawn_edges = [edges[i] for i in idx]
            edge_note = f"  (mostrando {MAX_DRAWN_EDGES}/{len(edges)} aristas)"
        segs = _edge_segments(pos, drawn_edges)
        lc = LineCollection(segs, colors="#333", linewidths=0.3, zorder=1)
        ax_net.add_collection(lc)

    sc = ax_net.scatter(
        pos[:, 0], pos[:, 1],
        c=vs[frames_idx[0]], cmap=CMAP, norm=norm,
        s=max(4, 300 // N), zorder=3, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax_net, fraction=0.04, pad=0.02)
    cbar.set_label("Potencial de membrana $v_i$", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    p_val = meta.get("p", "?")
    try:
        p_str = f"{float(p_val):g}"
    except (ValueError, TypeError):
        p_str = str(p_val)
    K_val = float(meta['K']) if 'K' in meta else None
    K_str = f"{K_val:g}" if K_val is not None else "?"

    fig.text(0.5, 0.97, "FitzHugh–Nagumo  |  Red Aleatoria (Erdős–Rényi)",
             color="white", fontsize=12, ha="center", va="top")
    fig.text(0.5, 0.93, f"p={p_str}, K={K_str}{edge_note}",
             color="#aaaaaa", fontsize=10, ha="center", va="top")

    time_text = fig.text(
        0.5, 0.89, f"t = {times[frames_idx[0]]:.2f} s",
        color="#ffd54f", fontsize=13, fontweight="bold",
        ha="center", va="top", family="monospace",
    )

    def init():
        return (sc, time_text)

    def update(frame_num):
        fi = frames_idx[frame_num]
        sc.set_array(vs[fi])
        time_text.set_text(f"t = {times[fi]:.2f} s")
        return (sc, time_text)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_idx), init_func=init,
        interval=1000 / fps, blit=False,
    )
    return fig, ani


def build_animation_ring(times, vs, ws, meta, edges, fps, max_frames, skip):
    """
    Red RING: Solo la animación circular, sin el panel de espacio-tiempo.
    Optimizada para evitar lag con N > 500.
    """
    N = vs.shape[1]
    frames_idx = list(range(0, len(times), skip))
    if max_frames:
        frames_idx = frames_idx[:max_frames]

    pos = circular_positions(N)

    v_all = vs[np.array(frames_idx)]
    vmin, vmax = v_all.min(), v_all.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Configuración de figura simple (sin Gridspec)
    fig = plt.figure(figsize=(7, 7), facecolor="#0d0d0d")
    ax_net = fig.add_axes([0.05, 0.05, 0.85, 0.82], facecolor="#0d0d0d")

    ax_net.set_xlim(-1.2, 1.2)
    ax_net.set_ylim(-1.2, 1.2)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    ring_k = int(meta.get("ring_k", 1))
    K_val = float(meta['K']) if 'K' in meta else None
    K_str = f"{K_val:g}" if K_val is not None else "?"

    # OPTIMIZACIÓN VISUAL: Solo dibujar aristas si la red es pequeña
    if N <= 100:
        simple_edges = []
        for i in range(N):
            for v in range(1, ring_k + 1):
                simple_edges.append((i, (i + v) % N))
        segs = _edge_segments(pos, simple_edges)
        lc = LineCollection(segs, colors="#2a2a2a", linewidths=0.6, zorder=1)
        ax_net.add_collection(lc)

    # OPTIMIZACIÓN VISUAL: Ajustar el tamaño del nodo según N
    node_size = max(2, 1500 // N) if N > 100 else 40

    sc = ax_net.scatter(
        pos[:, 0], pos[:, 1],
        c=vs[frames_idx[0]], cmap=CMAP, norm=norm,
        s=node_size, zorder=3, linewidths=0.5, 
        edgecolors="#000000" if N <= 100 else "none"
    )
    
    cbar_net = fig.colorbar(sc, ax=ax_net, fraction=0.04, pad=0.02)
    cbar_net.set_label("Potencial de membrana $v_i$", color="white", fontsize=9)
    cbar_net.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_net.ax.yaxis.get_ticklabels(), color="white")

    # Títulos
    fig.text(0.5, 0.97, "FitzHugh–Nagumo  |  Red Anillo",
             color="white", fontsize=12, ha="center", va="top")
    fig.text(0.5, 0.93, f"N={N}, k={ring_k}, K={K_str}",
             color="#aaaaaa", fontsize=10, ha="center", va="top")

    time_text = ax_net.text(
        0, 1.15, f"t = {times[frames_idx[0]]:.2f} s",
        color="#ffd54f", fontsize=13, fontweight="bold",
        ha="center", va="center", family="monospace",
        zorder=10, clip_on=False
    )

    def init():
        return sc, time_text

    def update(frame_num):
        fi = frames_idx[frame_num]
        sc.set_array(vs[fi])
        time_text.set_text(f"t = {times[fi]:.2f} s")
        return sc, time_text

    # OPTIMIZACIÓN DE RENDIMIENTO: blit=True
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_idx), init_func=init,
        interval=1000 / fps, blit=True,
    )
    return fig, ani


# ---------------------------------------------------------------------------
# Búsqueda de archivos
# ---------------------------------------------------------------------------
def find_newest_output(output_dir: Path) -> Path:
    files = sorted(output_dir.glob("fhn_*.txt"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos fhn_*.txt en {output_dir}")
    return files[-1]


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Anima la salida de la simulación de FitzHugh-Nagumo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="Ruta al archivo .txt de salida de la simulación. "
             "Por defecto usa el fhn_*.txt más nuevo en --output-dir.",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directorio para buscar archivos de salida (por defecto: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Cuadros por segundo de la animación (por defecto 20)")
    parser.add_argument("--dpi", type=int, default=90, help="DPI de la figura (por defecto 90)")
    parser.add_argument(
        "--save", default=None, metavar="FILE",
        help="Guarda la animación en un archivo (ej. anim.mp4 o anim.gif) en lugar de mostrarla.",
    )
    parser.add_argument(
        "--frames", type=int, default=None, metavar="N",
        help="Número máximo de cuadros de animación a usar (útil para vistas previas rápidas).",
    )
    parser.add_argument(
        "--skip", type=int, default=1, metavar="K",
        help="Usa 1 de cada K cuadros guardados (por defecto 1 = todos los cuadros).",
    )
    args = parser.parse_args()

    # resolver archivo
    if args.file:
        path = Path(args.file)
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            sys.exit(f"No se encontró el directorio de salida: {output_dir}\n"
                     "Corré una simulación primero, o pasá --output-dir.")
        path = find_newest_output(output_dir)
        print(f"Usando: {path}")

    print(f"Analizando {path} …")
    meta, edges, times, vs, ws = parse_output_file(path)

    network = meta.get("network", "FULL").upper()
    print(f"  Red: {network}  |  N={meta.get('N')}  |  K={meta.get('K')}  |"
          f"  cuadros={len(times)}  |  aristas={len(edges)}")

    matplotlib.rcParams["figure.facecolor"] = "#0d0d0d"
    matplotlib.rcParams["text.color"] = "white"

    if network == "FULL":
        fig, ani = build_animation_full(times, vs, ws, meta, args.fps, args.frames, args.skip)
    elif network == "RANDOM":
        fig, ani = build_animation_random(times, vs, ws, meta, edges, args.fps, args.frames, args.skip)
    elif network == "RING":
        fig, ani = build_animation_ring(times, vs, ws, meta, edges, args.fps, args.frames, args.skip)
    else:
        sys.exit(f"Tipo de red '{network}' desconocido en el encabezado del archivo.")

    if args.save:
        print(f"Guardando en {args.save} …")
        writer = "ffmpeg" if args.save.endswith(".mp4") else "pillow"
        ani.save(args.save, writer=writer, fps=args.fps, dpi=args.dpi)
        print("Listo.")
    else:
        plt.show()


if __name__ == "__main__":
    main()