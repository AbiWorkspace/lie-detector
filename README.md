# Lie Detector — Detección de engaño por análisis facial y fisiológico

Sistema de escritorio en Python que estima indicadores de engaño a partir de vídeo
(cámara en vivo o archivo), combinando análisis de microexpresiones faciales,
flujo óptico, seguimiento ocular y estimación de pulso por fotopletismografía remota (rPPG).

> **Aviso**: proyecto académico / de investigación. Los resultados son indicadores
> estadísticos y **no** constituyen una prueba de veracidad ni deben usarse para
> tomar decisiones sobre personas.

## Módulos

| Archivo | Función |
|---|---|
| `main.py` | Aplicación principal: interfaz Tkinter, captura, HUD y exportación a CSV/PDF |
| `gui_lie_detector.py` | Interfaz gráfica alternativa |
| `detector.py` | Clase `LieDetector`: motor de puntuación y fusión de señales |
| `LieDetector.py` | Versión monolítica previa del motor (referencia histórica) |
| `camera_thread.py` | Captura de cámara en hilo independiente |
| `microexpression_rules.py` | Reglas de microexpresiones sobre Action Units (FACS) |
| `micro_features.py` | Extracción de características de microexpresión |
| `optical_flow.py` | Flujo óptico para detectar movimiento facial sutil |
| `ppg_chrom.py` | Estimación de pulso por rPPG (método CHROM) |
| `utils_geom.py` | Utilidades geométricas sobre landmarks faciales |
| `visual_debug.py` | Overlays de depuración: barras, mallas, HUD, lie-score |

## Modelos incluidos

Ambos se versionan con **Git LFS**:

- `shape_predictor_68_face_landmarks.dat` — predictor de 68 landmarks faciales (dlib, ~95 MB)
- `modelo_emociones.h5` — clasificador de emociones (Keras)

## Requisitos

- Python 3.10 u 3.11
- **ffmpeg/ffprobe** en el `PATH` (se usa para leer la rotación de los vídeos)
- Git LFS instalado antes de clonar

Paquetes principales:

```
opencv-python
py-feat
dlib
tensorflow
numpy
pillow
matplotlib
reportlab
```

## Instalación

```bash
git lfs install
git clone <url-del-repositorio>
cd <carpeta>

python -m venv venv
venv\Scripts\activate        # Windows
pip install opencv-python py-feat dlib tensorflow numpy pillow matplotlib reportlab
```

## Uso

```bash
python main.py
```

La aplicación permite analizar la cámara en vivo o cargar un archivo de vídeo.
Los resultados se escriben en `C:\LieDetectorData\`:

- `resultados_completos.csv` — todas las métricas por frame
- `resultados_mentira.csv` — solo los segmentos marcados como sospechosos
- Informe en PDF con las gráficas de la sesión

## Calibración

`baseline.json` guarda la línea base del sujeto (EAR ocular, tamaño de pupila y
frecuencia cardiaca en reposo). Se debe recalibrar por persona y por condiciones
de iluminación; el archivo incluido es solo un ejemplo.

## Documentación

La carpeta `docs/` contiene la memoria completa del proyecto en Word y PDF.

## Dataset

El enlace al conjunto de imágenes positivas y negativas usado está en
`LINK DE LA IMAGENES POSITIVAS Y NEGATIVAS .txt` (requiere cuenta institucional).
