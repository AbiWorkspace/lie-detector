# Lie Detector — Detección de engaño por análisis facial y fisiológico

Sistema de escritorio en Python que estima indicadores de engaño a partir de vídeo
(cámara en vivo o archivo), combinando análisis de microexpresiones faciales,
flujo óptico, seguimiento ocular/pupilar, pose de cabeza y estimación de pulso
por fotopletismografía remota (rPPG). Todas las señales se funden en un único
**lie-score** (0–1) mediante un modelo logístico interpretable.

> **Aviso**: proyecto académico / de investigación (memoria incluida en `docs/`).
> Los resultados son indicadores estadísticos y **no** constituyen una prueba de
> veracidad ni deben usarse para tomar decisiones sobre personas.

## Cómo funciona (pipeline por frame)

Cada frame de vídeo pasa por `LieDetector.analyze_frame()` (`detector.py`), que:

1. **Detección de rostro** — `face_recognition` (HOG) localiza la cara; `dlib`
   obtiene 68 landmarks 2-D y `face_alignment` obtiene 68 landmarks 3-D (pitch/yaw/roll).
2. **Malla facial** — MediaPipe FaceMesh (468 puntos, con iris refinados) para EAR,
   diámetro de pupila y ratio de mirada.
3. **Emoción** — se funden tres fuentes: un CNN propio (`modelo_emociones.h5`,
   Keras), DeepFace y FER, para reducir el riesgo de error de un único clasificador.
4. **Action Units (FACS)** — `py-feat` extrae AUs (tensión de labios AU14/17/24,
   cejas AU01/04) usadas como features adicionales.
5. **Microexpresiones** — `optical_flow.py` calcula flujo óptico denso
   (Farnebäck) y, si supera un umbral de movimiento, `microexpression_rules.py`
   evalúa ~12 reglas heurísticas sobre buffers temporales (ojo, boca, cejas,
   mandíbula, curvatura de labios) para clasificar el gesto: Sorpresa, Miedo,
   Asco, Labios tensos, Microsonrisa, Ceño fruncido, Dientes apretados, Boca
   apretada, Ceja única, Tensión ocular, Sonrisa falsa/genuina, etc. Cada tipo
   tiene un peso fijo en `MICRO_W` y requiere ≥2 votos en una ventana de 4
   frames antes de confirmarse (evita falsos positivos por ruido).
6. **Parpadeo** — una máquina de estados (open→closing→closed→opening) sobre el
   EAR suavizado (mediana + EMA) cuenta parpadeos con duración 30–450 ms;
   se deriva el ritmo (BPM) sobre una ventana móvil de 60 s.
7. **Pulso (rPPG)** — `ppg_chrom.py` implementa el método CHROM sobre la señal
   de color de la piel; en `detector.py` se aplica además un filtro
   Butterworth pasa-banda (0.7–3 Hz) y FFT para estimar el HR dominante.
8. **Fusión (lie-score)** — todas las señales se normalizan a 0–1 respecto a la
   línea base del sujeto y se combinan con una regresión logística
   (`_logit`, ver más abajo) para producir el lie-score del frame.

## Modelo de scoring (lie-score)

El detector no usa un único umbral; combina ~15 features normalizadas mediante
un modelo logístico simple: `score = sigmoide(bias + Σ peso_i · feature_i)`.

Los pesos heurísticos base (`BASE_WEIGHTS` en `detector.py`) son:

| Feature | Peso | Significado |
|---|---|---|
| `emo_fusion` | 0.20 | Emoción negativa intensa (fusión DeepFace+FER+CNN) |
| `ear` | 0.15 | Micro-cierre ocular respecto a la línea base |
| `blink` | 0.12 | Dinámica de parpadeo (freeze/rebote) |
| `micro_opt` | 0.12 | Microexpresión óptica reciente detectada |
| `pupil` | 0.08 | Dilatación pupilar relativa |
| `hr` | 0.08 | Variación de frecuencia cardiaca (rPPG) |
| `au_tension` | 0.07 | Tensión de labios (AUs) |
| `yaw` | 0.06 | Desvío lateral del rostro |
| `au_brows` | 0.05 | Tensión de cejas (AUs) |
| `pitch` | 0.05 | Inclinación vertical de cabeza |
| `gaze` | 0.05 | Dispersión/movimiento del iris |
| `bpm` | 0.05 | Parpadeos por minuto, normalizado respecto a línea base |
| `nostril_flare` | 0.05 | Dilatación de fosas nasales |
| `roll` | 0.04 | Cabeceo lateral |
| `smile` | 0.04 | Sonrisa genuina (↓ score) vs. falsa (↑ score) |

Estos pesos se multiplican ×3 para obtener los coeficientes logísticos por
defecto; si existe un fichero `weights.npy` en el directorio de ejecución
(no incluido en el repo) se carga y sobrescribe los pesos por defecto —
permite reentrenar/ajustar el modelo sin tocar código.

Umbrales relevantes (constantes de clase en `LieDetector`):

- `TH_LIE = 0.35` — por encima de este valor un frame se considera "sospechoso" (se vuelca a `resultados_mentira.csv`).
- `VENTANA = 30` frames y umbral `0.65` — el **veredicto de sesión** (`verdict()`) promedia el lie-score de los últimos 30 frames; si supera 0.65 se marca "Mentira".
- `CALIBRATION_FRAMES = 60` (~2 s a 30 fps) — frames iniciales usados para fijar la línea base de EAR/pupila/HR.
- `BLINK_UMBRAL_ANSIEDAD = 20` bpm y `BLINK_UMBRAL_CONCENTRACION = 10` bpm — clasifican el estado de parpadeo en NORMAL / ANSIEDAD·MENTIRA / CONCENTRACIÓN.

## Arquitectura del proyecto

| Archivo | Función |
|---|---|
| `main.py` | Aplicación principal (Tkinter): captura webcam/vídeo, HUD en vivo, exportación a CSV y a PDF, informe de sesión con gráficas |
| `gui_lie_detector.py` | Interfaz gráfica alternativa/anterior — **desactualizada**, ver [Limitaciones](#limitaciones-y-deuda-técnica-conocida) |
| `detector.py` | Clase `LieDetector`: motor de puntuación y fusión de señales (versión activa, usada por `main.py`) |
| `LieDetector.py` | Versión monolítica previa del motor (referencia histórica, no se importa desde `main.py`) |
| `camera_thread.py` | Captura de cámara en hilo independiente (`CameraThread`) |
| `microexpression_rules.py` | ~25 reglas de microexpresión sobre Action Units/geometría (FACS) + tabla de pesos `MICRO_W` |
| `micro_features.py` | `feature_vector()` / `detect_micro()`: extrae y clasifica características de microexpresión desde landmarks |
| `optical_flow.py` | Flujo óptico denso (Farnebäck) para overlay y detección de movimiento facial sutil |
| `ppg_chrom.py` | Clase `PPGChrom`: estimación de pulso por rPPG (método CHROM) |
| `utils_geom.py` | `compute_ear`, `solve_pose`, `pupil_diameter` — utilidades geométricas sobre landmarks |
| `visual_debug.py` | Overlays de depuración: malla 3-D, barras ojo/ceja/boca, lie-bar, HUD, etiquetas de emoción |

## Modelos y datos incluidos

Ambos se versionan con **Git LFS**:

- `shape_predictor_68_face_landmarks.dat` — predictor de 68 landmarks faciales (dlib, ~96 MB)
- `modelo_emociones.h5` — clasificador de emociones propio (Keras, ~856 KB)

`baseline.json` guarda un ejemplo de línea base ya calibrada (EAR, pupila, HR
en reposo). `weights.npy` (pesos del modelo logístico) **no** está en el repo;
si no existe, el detector usa los pesos por defecto derivados de `BASE_WEIGHTS`.

## Requisitos

- Python 3.10 u 3.11 (TensorFlow/dlib/mediapipe limitan la versión soportada)
- **ffmpeg/ffprobe** en el `PATH` (se usa para leer la rotación de los vídeos)
- Git LFS instalado antes de clonar
- En Windows, compilar `dlib` requiere CMake y las Build Tools de Visual C++
  (o instalar una rueda precompilada compatible con tu versión de Python)
- Webcam o archivo de vídeo como fuente

Paquetes usados por el código (no existe `requirements.txt` en el repo — instalar manualmente):

```
opencv-python
dlib
face-recognition
face-alignment
mediapipe
py-feat
deepface
fer
tensorflow
scikit-image
scipy
numpy
pillow
matplotlib
reportlab
```

`tkinter` viene con la instalación estándar de Python en Windows (no requiere `pip install`).

## Instalación

```bash
git lfs install
git clone https://github.com/AbiGuadarrama/lie-detector.git
cd lie-detector

python -m venv venv
venv\Scripts\activate        # Windows

pip install opencv-python dlib face-recognition face-alignment mediapipe ^
    py-feat deepface fer tensorflow scikit-image scipy numpy pillow matplotlib reportlab
```

> Si `pip install dlib` falla en Windows, instala primero CMake
> (`pip install cmake`) y las "Desktop development with C++" Build Tools de
> Visual Studio, o usa una rueda `.whl` precompilada para tu versión de Python.

## Configuración antes de ejecutar

`main.py` y `gui_lie_detector.py` cargan el modelo y el predictor con **rutas
absolutas hardcodeadas** (propias del equipo original):

```python
LieDetector(
    r"C:/Users/Abigu/OneDrive/Escritorio/PULIDO/proyecto_lie_detector/modelo_emociones.h5",
    r"C:/Users/Abigu/OneDrive/Escritorio/PULIDO/proyecto_lie_detector/shape_predictor_68_face_landmarks.dat"
)
```

**Debes editar estas rutas** (en `main.py`, línea ~145) para que apunten a los
ficheros `modelo_emociones.h5` y `shape_predictor_68_face_landmarks.dat` de tu
propia copia del repositorio antes de ejecutar la aplicación.

### Variables de entorno

- `LIE_LIGHT_MODE=1` — desactiva DeepFace y FER (las dos fuentes de emoción
  más pesadas) para maximizar FPS; solo lo respeta `detector.py`, no
  `LieDetector.py`.

## Uso

```bash
python main.py
```

En la ventana principal:

- **📂 Cargar video** / **🎥 Usar webcam** — cambia la fuente de vídeo en caliente.
- **Zoom rostro** — recorta el frame al rostro detectado.
- **Iniciar/Detener análisis** — activa el pipeline; al detener, se genera
  automáticamente el informe de sesión (popup con gráficas + botón para
  exportar a PDF).

Durante el análisis, el HUD muestra en vivo: emoción dominante y confianza,
lie-score, EAR, tamaño de pupila, yaw, BPM y estado de parpadeo, veredicto
acumulado, última microexpresión óptica detectada y las emociones de
DeepFace/FER en paralelo.

## Salidas generadas

Los resultados se escriben en `C:\LieDetectorData\` (ruta hardcodeada, no
sincronizada con OneDrive a propósito, ver `main.py`):

- `resultados_completos.csv` — todas las métricas por frame. Columnas:
  `fecha_hora, emo, conf, lie, ear, pupil, yaw, hr, micro, veredicto, blinks`
- `resultados_mentira.csv` — mismo formato, solo filas con `lie > TH_LIE`

Al finalizar la sesión (botón Detener análisis / cerrar ventana) se generan
además, en el directorio de trabajo:

- `graf_lie.png`, `graf_micro.png`, `graf_microopt.png` — evolución del
  lie-score y eventos de microexpresión (clásicas y por flujo óptico)
- `reporte_sesion.txt` — resumen textual (duración, lie-score medio/máximo,
  parpadeos, BPM, nº de microexpresiones, emoción predominante, veredicto)
- `microexpresiones.log` — histórico acumulado de eventos de microexpresión
- Informe en PDF (bajo demanda) en `C:\Users\<usuario>\Documentos\LieDetectorReports\` (ruta también hardcodeada en `main.py`)
- `blink_debug.log` / `logcat.txt` — logs rotativos de depuración (`LieDetector.py` / `detector.py` respectivamente)

## Calibración

`baseline.json` guarda la línea base del sujeto (EAR ocular, tamaño de pupila
y frecuencia cardiaca en reposo). El detector calibra automáticamente en los
primeros `CALIBRATION_FRAMES` (60) frames de cada sesión y guarda la línea
base actualizada al cerrar la app (`detector.save_baseline()`). Se recomienda
recalibrar por persona y por condiciones de iluminación; el archivo incluido
es solo un ejemplo.

## Limitaciones y deuda técnica conocida

- **Rutas absolutas hardcodeadas** al modelo, al predictor de landmarks y al
  directorio de salida de PDFs — deben editarse manualmente por cada usuario/máquina.
- **`gui_lie_detector.py` está desactualizado**: instancia `CameraThread(src=0, qsize=2)`,
  pero `camera_thread.py` solo acepta un único parámetro posicional
  (`CameraThread(source)`) — esa GUI alternativa fallará tal como está.
- **`LieDetector.py` es código duplicado/histórico** de `detector.py` (mismo
  motor, sin `light_mode` ni algunos ajustes de FSM de parpadeo más recientes);
  `main.py` importa `detector.py`, no este archivo.
- Sin `requirements.txt`/`pyproject.toml` — las dependencias deben instalarse
  manualmente siguiendo la lista de este README.
- Pipeline pensado para **un único rostro** por frame y para **CPU** (todos los
  modelos se cargan con `device="cpu"`); en equipos modestos el framerate real
  puede caer muy por debajo del framerate nominal del vídeo/cámara.
- El modelo de emociones propio, los umbrales de microexpresión y los pesos de
  fusión son heurísticos, ajustados de forma empírica — no se aporta una
  validación cuantitativa formal en el repo (ver `docs/` para la metodología
  descrita en la memoria).

## Documentación

La carpeta `docs/` contiene la memoria completa del proyecto en Word y PDF
(`Memoria_LieDetector.docx` / `.pdf`), con la metodología, marco teórico y
resultados experimentales.

## Dataset

El enlace al conjunto de imágenes positivas y negativas usado está en
`LINK DE LA IMAGENES POSITIVAS Y NEGATIVAS .txt` (requiere cuenta institucional).
