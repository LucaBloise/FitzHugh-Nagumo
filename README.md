# TP5 — FitzHugh–Nagumo (Sistema 2)

## Simulation (Java)

```bat
cd Simulation
compile.bat
java -cp bin fhn.Main --network FULL --k 1.0 --sample-every 200
```

Output always under **`Simulation/output/`** (even if you run Java from the project root). Default `dt=0.005` for stability with `network=FULL`, `K=1`, `N>500` (with `dt=0.01` the coupling can blow up to NaN within ~0.1 s).

## Animation (Python only, for now)

```bat
cd Visualization
pip install -r requirements.txt
python animate.py
python animate.py --input ..\Simulation\output\your_run.txt --frame
```

- Default input: newest `.txt` in `Simulation/output/`
- Default output: `Visualization/output/animations/<stem>.gif`
- `--frame` saves a PNG still for slides (guía)

Other post-processing (plots, studies, observables) — add later.

## Entrega

- Motor Java: `Simulation/fhn/` (&lt; 20 kB)
- Presentación PDF + enlaces YouTube/Vimeo para animaciones
