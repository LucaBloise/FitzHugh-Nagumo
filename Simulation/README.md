# Simulation (FitzHugh-Nagumo)

Este README describe como usar **solo** el subproyecto `Simulation`:
- compilar y correr simulaciones en Java
- animar salidas crudas con `animate.py`

No depende de la carpeta de visualizacion externa.

## 1. Requisitos

- Java 17+ (con `javac` y `java` en PATH)
- Python 3.10+ para `animate.py`
- Paquetes Python:
  - `numpy`
  - `matplotlib`
- Opcional para exportar MP4 desde animate:
  - `ffmpeg`
- Opcional para exportar GIF:
  - `pillow`

Instalacion de paquetes Python (PowerShell):

```powershell
py -3.13 -m pip install numpy matplotlib pillow
```

## 2. Trabajar desde la carpeta Simulation

Abrir terminal y ubicarse en `Simulation`:

```powershell
Set-Location c:\Users\juani\Documents\itba\sds\FitzHugh-Nagumo\Simulation
```

Todos los comandos de este README asumen que estas parado en esa carpeta.

## 3. Compilar Java

Crear `bin` si no existe y compilar:

```powershell
New-Item -ItemType Directory -Force bin | Out-Null
javac -d bin fhn\*.java
```

## 4. Correr simulaciones (Java)

Comando base:

```powershell
java -cp bin fhn.Main [opciones]
```

Opciones principales:

- `--n <int>`: cantidad de neuronas (el proyecto usa default 512)
- `--k <double>`: acoplamiento
- `--dt <double>`: paso de integracion (default 0.005)
- `--tmax <double>`: tiempo final
- `--network FULL|RANDOM|RING`
- `--p <double>`: probabilidad de conexion (solo RANDOM)
- `--ring-k <int>`: semiancho (solo RING)
- `--seed <long>`: semilla
- `--realization <int>`: etiqueta de realizacion
- `--sample-every <int>`: cada cuantos pasos guardar estado
- `--output <path>`: archivo de salida (si no se pasa, se autogenera en `output/`)

### Ejemplo A: FULL

```powershell
java -cp bin fhn.Main --n 512 --k 0.1 --dt 0.005 --tmax 500 --network FULL --seed 42 --realization 1
```

### Ejemplo B: RANDOM

```powershell
java -cp bin fhn.Main --n 512 --k 0.1 --dt 0.005 --tmax 500 --network RANDOM --p 0.1 --seed 43 --realization 2
```

### Ejemplo C: RING

```powershell
java -cp bin fhn.Main --n 512 --k 0.1 --dt 0.005 --tmax 500 --network RING --ring-k 5 --seed 44 --realization 3
```

Salida esperada:
- un `.txt` con encabezado + muestras de estado en `output/` (o en la ruta indicada por `--output`)

## 5. Formato de salida (resumen)

Cada corrida genera un `.txt` con:
- metadatos (N, K, network, seed, etc.)
- lista de aristas (`EDGE i j`) cuando aplica
- muestras temporales en bloques:
  - `BEGIN_STATE t=...`
  - linea `v ...`
  - linea `w ...`

Ese archivo es el que consume `animate.py`.

## 6. Animar resultados con animate.py

Script incluido en esta carpeta: `animate.py`.

### 6.1 Abrir animacion del archivo mas nuevo en output/

```powershell
py -3.13 animate.py
```

### 6.2 Animar un archivo especifico

```powershell
py -3.13 animate.py output\fhn_full_K0.1_p0.5_ring3_N512_seed42_real1_20260611_225140.txt
```

### 6.3 Guardar MP4

```powershell
py -3.13 animate.py --save anim.mp4
```

### 6.4 Guardar GIF

```powershell
py -3.13 animate.py --save anim.gif
```

### 6.5 Vista rapida (menos cuadros)

```powershell
py -3.13 animate.py --frames 200 --skip 2 --fps 20
```

### 6.6 Buscar en otra carpeta de outputs

```powershell
py -3.13 animate.py --output-dir output_red_compare
```

### 6.7 Vista alternativa en grilla

```powershell
py -3.13 animate.py --grid-layout
```

## 7. Parametros utiles de animate.py

- `file`: ruta opcional al `.txt`
- `--output-dir <dir>`: carpeta donde busca el archivo mas nuevo si no pasas `file`
- `--fps <float>`: cuadros por segundo
- `--dpi <int>`: resolucion
- `--save <file.mp4|file.gif>`: exporta en vez de mostrar
- `--frames <N>`: maximo de cuadros
- `--skip <K>`: usa 1 de cada K cuadros
- `--grid-layout`: layout alternativo en grilla

## 8. Flujo minimo recomendado

```powershell
Set-Location c:\Users\juani\Documents\itba\sds\FitzHugh-Nagumo\Simulation
New-Item -ItemType Directory -Force bin | Out-Null
javac -d bin fhn\*.java
java -cp bin fhn.Main --n 512 --k 0.1 --dt 0.005 --tmax 500 --network RANDOM --p 0.1 --seed 42 --realization 1
py -3.13 animate.py
```

## 9. Troubleshooting rapido

- `Error: Could not find or load main class fhn.Main`
  - Verifica que compilaste con `javac -d bin fhn\*.java` y que usas `-cp bin`.

- `No se encontraron archivos fhn_*.txt en output`
  - Corre primero una simulacion, o usa `--output-dir` apuntando a la carpeta correcta.

- Falla al guardar MP4
  - Instala `ffmpeg` y reintenta.

- Falla al guardar GIF
  - Instala `pillow` (`py -3.13 -m pip install pillow`).
