# TP5 — FitzHugh-Nagumo (Sistema 2)

Este repositorio contiene la simulacion del modelo FitzHugh-Nagumo para el TP5,
enfocado en el sistema 2 del enunciado.

## Requisitos

- Java 17 o superior (con `javac` y `java` en PATH)
- Python 3.10+ para la visualizacion
- Paquetes de Python:
	- `numpy`
	- `matplotlib`

Instalacion de dependencias de Python:

```bash
pip install numpy matplotlib
```

## 1) Compilar la simulacion (Java)

Desde la raiz del repo:

```bash
mkdir -p Simulation/bin
javac -d Simulation/bin Simulation/fhn/*.java
```

En PowerShell (Windows), si `mkdir -p` no funciona:

```powershell
New-Item -ItemType Directory -Force Simulation/bin
javac -d Simulation/bin Simulation/fhn/*.java
```

## 2) Correr la simulacion (Sistema 2)

Comando base:

```bash
java -cp Simulation/bin fhn.Main [opciones]
```

Opciones principales:

- `--n <int>`: cantidad de neuronas (debe ser > 500 por enunciado)
- `--k <double>`: acoplamiento `K`
- `--dt <double>`: paso del integrador (default `0.005`)
- `--tmax <double>`: tiempo maximo
- `--network FULL|RANDOM|RING`: topologia de red
- `--p <double>`: probabilidad de conexion para `RANDOM`
- `--ring-k <int>`: semiancho para `RING`
- `--seed <long>`: semilla
- `--realization <int>`: etiqueta de realizacion
- `--sample-every <int>`: cada cuantos pasos se guarda estado
- `--output <path>`: ruta de salida (si no se indica, genera nombre automatico en `Simulation/output`)

Ejemplo (sistema 2, N > 500):

```bash
java -cp Simulation/bin fhn.Main --n 512 --k 1.0 --dt 0.005 --tmax 500 --network RANDOM --p 0.2 --seed 42 --realization 1
```

Al finalizar, se genera un archivo `.txt` dentro de `Simulation/output/`.

## 3) Animar resultados (Python)

Desde la raiz del repo:

```bash
python Visualization/animate.py
```

Ese comando toma automaticamente el archivo `fhn_*.txt` mas nuevo de `Simulation/output`.

Usos utiles:

```bash
# elegir archivo explicito
python Visualization/animate.py Simulation/output/mi_archivo.txt

# guardar animacion en mp4
python Visualization/animate.py --save anim.mp4

# vista previa mas rapida
python Visualization/animate.py --frames 200 --skip 2
```

Si queres cambiar la carpeta de salida:

```bash
python Visualization/animate.py --output-dir otra/carpeta/output
```

## 4) Barrido de K con N repeticiones

Para correr varias simulaciones automaticamente (barrido de `K` con `N` repeticiones por cada valor), usar:

```bash
python Visualization/sweep.py --compile-first --k-values 0.6 0.8 1.0 1.2 --repetitions 5 --n 512 --network RANDOM --p 0.2 --dt 0.005 --tmax 500
```

Para correr en paralelo con threads:

```bash
python Visualization/sweep.py --compile-first --k-values 0.6 0.8 1.0 1.2 --repetitions 5 --threads 4 --n 512 --network RANDOM --p 0.2 --dt 0.005 --tmax 500
```

Si algunas corridas fallan por inestabilidad numerica, podes reintentar automaticamente con dt mas chico:

```bash
python Visualization/sweep.py --compile-first --k-values 0.6 0.8 1.0 1.2 --repetitions 15 --threads 15 --n 512 --network FULL --dt 0.005 --retry-dts 0.0025 0.001 --tmax 500
```

Parametros utiles:

- `--k-values`: lista de valores de `K` a barrer.
- `--repetitions`: cantidad de corridas por cada `K`.
- `--base-seed`: seed inicial para generar seeds por corrida (default `42`).
- `--threads`: cantidad de corridas en paralelo (default `1`).
- `--retry-dts`: lista opcional de `dt` para reintentar corridas fallidas.
- `--network`: `FULL`, `RANDOM` o `RING`.
- `--p`: probabilidad de conexion (solo para `RANDOM`).
- `--ring-k`: semiancho de vecindad (solo para `RING`).
- `--output-prefix`: prefijo opcional para nombrar los archivos de salida.

Notas:

- El script corre `fhn.Main` una vez por realizacion.
- Las salidas quedan en `Simulation/output/`.

## 5) Graficar la evolucion temporal de <v(t)> para distintos K

Para tomar los outputs del sweep y graficar una curva por cada valor de `K`:

```bash
python Visualization/plot_mean_v.py
```

El script:

- Lee archivos en `Simulation/output/` (patron `fhn_*.txt`).
- Calcula `<v(t)>` por realizacion (promedio espacial sobre neuronas).
- Agrupa por `K` y grafica el promedio temporal por grupo.
- Si hay multiples realizaciones por `K`, agrega banda sombreada `+-1 std`.

Opciones utiles:

```bash
# guardar figura en archivo
python Visualization/plot_mean_v.py --save Visualization/mean_v_vs_time.png

# cambiar carpeta/patron de entrada
python Visualization/plot_mean_v.py --input-dir Simulation/output --pattern "fhn_*.txt"

# sin banda de desvio estandar
python Visualization/plot_mean_v.py --no-std-band

# forzar recalculo desde .txt y guardar cache CSV
python Visualization/plot_mean_v.py --csv Visualization/mean_v_cache.csv --csv-mode recompute --save Visualization/mean_v_vs_time.png

# usar un CSV ya existente (sin reparsear outputs)
python Visualization/plot_mean_v.py --csv Visualization/mean_v_cache.csv --csv-mode use --save Visualization/mean_v_vs_time.png

# recalcular con threads (0 = automatico segun cantidad de archivos/reps)
python Visualization/plot_mean_v.py --csv Visualization/mean_v_cache.csv --csv-mode recompute --threads 0 --save Visualization/mean_v_vs_time.png
```

Notas de CSV:

- `--csv`: archivo cache con los datos de `<v(t)>`.
- `--csv-mode auto` (default): si existe CSV lo usa; si no existe, recalcula y lo crea.
- `--csv-mode recompute`: siempre recalcula desde outputs y sobrescribe el CSV.
- `--csv-mode use`: usa solo el CSV existente.
- `--threads`: cantidad de threads para parsear outputs al recalcular (`0` = automatico).
- El CSV guarda solo datos agregados necesarios para graficar por cada `K` y tiempo: `k,time,mean_v,std_v,n_reps`.

## 6) Punto 3: evolucion temporal de la dispersion espacial sigma_v(t)

Para estudiar sigma_v(t) y promediar sobre varias realizaciones por cada `K`:

```bash
python Visualization/plot_sigma_v.py --csv Visualization/sigma_v_cache.csv --csv-mode recompute --threads 0 --save Visualization/sigma_v_vs_time.png
```

Uso con cache existente (sin reparsear outputs):

```bash
python Visualization/plot_sigma_v.py --csv Visualization/sigma_v_cache.csv --csv-mode use --save Visualization/sigma_v_vs_time.png
```

Filtrar y usar solo ciertos valores de `K`:

```bash
python Visualization/plot_sigma_v.py --csv Visualization/sigma_v_cache.csv --csv-mode recompute --k-values 0.2 0.4 0.6 0.8 1.0 --save Visualization/sigma_v_vs_time.png
```

Este script:

- Lee corridas desde `Simulation/output/`.
- Calcula `sigma_v(t)` por realizacion (desvio estandar espacial sobre neuronas).
- Agrupa por `K` y promedia entre realizaciones.
- Guarda CSV compacto con: `k,time,mean_sigma_v,std_sigma_v,n_reps`.
- Permite filtrar por valores especificos de `K` con `--k-values`.