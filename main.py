import cv2, csv, sys
from datetime import datetime
import time
import os	
from detector import LieDetector
from camera_thread import CameraThread
from visual_debug import draw_bars, draw_liescore, draw_mesh, draw_hud
from feat import Detector
from optical_flow import reset as reset_flow
import subprocess, json

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.set_loglevel("warning")
OUTPUT_DIR = r"C:\LieDetectorData"          # ❶ NO sincronizado con OneDrive
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_full_path = os.path.join(OUTPUT_DIR, "resultados_completos.csv")
csv_partial_path = os.path.join(OUTPUT_DIR, "resultados_mentira.csv")

import logging
logging.getLogger("feat").setLevel(logging.ERROR)
# --- PDF export (reportlab) ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
inicio_sesion = None
def open_video_auto_rotate(path):
    """
    Devuelve (cv2.VideoCapture, rotate_code) donde rotate_code es
    cv2.ROTATE_* o None si no hay que girar.
    Requiere ffprobe (viene con ffmpeg) en el PATH.
    """
    # ---- lee metadatos con ffprobe ----
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            text=True                       # str, no bytes
        )
        tags = json.loads(probe)["streams"][0].get("tags", {})
        rot  = int(tags.get("rotate", 0))
    except Exception:
        rot = 0                             # si falla, asumimos 0

    # ---- asigna código OpenCV ----
    rotate_code = { 90:  cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE }.get(rot)

    cap = cv2.VideoCapture(path)
    return cap, rotate_code


class VideoPlayer:
    def __init__(self, video_path, panel):        
      

        # ⬇️ usa el helper
        self.cap, self.rot_code = open_video_auto_rotate(video_path)
        self.panel = panel
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        if self.fps <= 0 or self.fps > 240:
            self.fps = 30
        self.update_interval = int(1000 / self.fps)
        self.running = True

    def read_frame(self):
        if not self.running:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            return None
        # 🔄 corrige orientación si es necesario
        if self.rot_code is not None:
            frame = cv2.rotate(frame, self.rot_code)
        return frame

    def stop(self):
        self.running = False
        self.cap.release()

# ─── logging a color y robusto ─────────────────────────────
class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: "\033[1;36m",    # cyan
        logging.INFO: "\033[1;32m",     # green
        logging.WARNING: "\033[1;33m",  # yellow
        logging.ERROR: "\033[1;31m",    # red
        logging.CRITICAL: "\033[1;41m", # red bg
    }
    RESET = "\033[0m"
    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"

class EveryN(logging.Filter):
    def __init__(self, n): super().__init__(); self.n = n; self.i = 0
    def filter(self, record):
        if record.levelno > logging.DEBUG:
            return True
        self.i = (self.i + 1) % self.n
        return self.i == 0

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logging.getLogger().addFilter(EveryN(2))
logger = logging.getLogger(__name__)

header = ["fecha_hora","emo","conf","lie","ear",
          "pupil","yaw","hr","micro","veredicto","blinks"]
try:
    # a)  Modo 'w' para sobrescribir y escribir cabeceras
    csv_file = open(csv_full_path, "w", newline='', encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    # b)  Otro CSV más pequeño si lo quieres
    csv_file_small = open(csv_partial_path, "w", newline='', encoding="utf-8")
    csv_writer_small = csv.writer(csv_file_small)
    csv_writer_small.writerow(header)

    logger.info("[main] CSVs inicializados correctamente.")
except Exception as e:
    logger.critical(f"[main] ERROR al inicializar CSVs: {e}", exc_info=True)
    sys.exit(1)


# =============== VENTANA PRINCIPAL ===============
root = tk.Tk()
root.title("Lie Detector 4.0 — Unificado")
root.geometry("1280x780")
root.resizable(True, True)

frame_top = tk.Frame(root, bg="#222")
frame_top.pack(side="top", fill="x", pady=2)
# ─── Inicializar detector & cámara ─────────────────────────
try:
    detector = LieDetector(
        r"C:/Users/Abigu/OneDrive/Escritorio/PULIDO/proyecto_lie_detector/modelo_emociones.h5",
        r"C:/Users/Abigu/OneDrive/Escritorio/PULIDO/proyecto_lie_detector/shape_predictor_68_face_landmarks.dat"
    )
    csv_buffer = []
    logger.info("[main] LieDetector inicializado correctamente.")
except Exception as e:
    logger.critical(f"[main] ERROR al inicializar LieDetector: {e}", exc_info=True)
    sys.exit(1)

video_path = None
usando_video = False 
cam = CameraThread(0)  # Webcam por default
cam.start()

detector.fps = getattr(cam, 'fps', 30)

logger.info("[main] Modo: WEBCAM (por defecto)")

label_video = tk.Label(root, text="Fuente: Webcam", font=("Arial", 9), bg="#222", fg="#0f0")
label_video.pack(side="top", fill="x")

# Función para cargar video desde botón
def load_video():
    global player, usando_video
    path = filedialog.askopenfilename(
        title="Selecciona un archivo de video",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
    )
    if path:
        try:
            logger.info(f"[main] Cambiando a VIDEO '{path}'")
            if cam:
                cam.stop()
            usando_video = True
            player = VideoPlayer(path, panel)
            detector.fps = player.fps
            label_video.config(text=f"Fuente: {path}", fg="#ff0")
            messagebox.showinfo("Video cargado", f"Analizando: {path}")
            reset_flow()                    # limpia el frame previo de flujo
            update_frame()
        except Exception as e:
            logger.error(f"[main] Error al cargar video: {e}")
            messagebox.showerror("Error", f"No se pudo cargar el video:\n{e}")


# Función para regresar a webcam
def use_webcam():
    global cam, usando_video
    try:
        logger.info("[main] Regresando a WEBCAM")
        if player:
            player.stop()
        usando_video = False
        if cam:
            cam.stop()
        cam = CameraThread(0)
        cam.start()
        label_video.config(text="Fuente: Webcam", fg="#0f0")
        reset_flow() 
        update_frame()
    except Exception as e:
        logger.error(f"[main] Error al activar webcam: {e}")
        messagebox.showerror("Error", "No se pudo activar la webcam.")


btn_video = tk.Button(frame_top, text="📂 Cargar video", font=("Arial", 10), command=load_video)
btn_video.pack(side="left", padx=5, pady=3)
btn_webcam = tk.Button(frame_top, text="🎥 Usar webcam", font=("Arial", 10), command=use_webcam)
btn_webcam.pack(side="left", padx=5, pady=3)

# ─── Variables de control para GUI ──────────────
analyzing = False
zoom_face = False
show_debug_mesh = True

def toggle_analysis():
    global analyzing
    analyzing = not analyzing
    btn_toggle.config(
        text="Detener análisis" if analyzing else "Iniciar análisis"
    )
    if not analyzing:                       # se acaba de parar
        csv_file.flush(); csv_file_small.flush()      # ← ### CAMBIO
        mostrar_informe()


def toggle_zoom():
    global zoom_face
    zoom_face = not zoom_face
    btn_zoom.config(text="Zoom rostro: ON" if zoom_face else "Zoom rostro: OFF")

btn_zoom = tk.Button(frame_top, text="Zoom rostro: OFF", font=("Arial", 10), command=toggle_zoom)
btn_zoom.pack(side="left", padx=5, pady=3)


btn_toggle = tk.Button(frame_top, text="Iniciar análisis", font=("Arial", 13), command=toggle_analysis)
btn_toggle.pack(side="left", padx=5, pady=3)



chk_frame = tk.Frame(frame_top, bg="#222")
chk_frame.pack(side="left", padx=10)

# Panel para el video
panel = tk.Label(root, bg="black")
panel.pack(side="top", expand=True, fill="both")

def exportar_pdf_informe():
    try:
        logger.info("[pdf] Generando informe PDF...")

        # Ruta donde guardar el PDF
        REPORTS_DIR = r"C:\Users\Abigu\Documentos\LieDetectorReports"
        os.makedirs(REPORTS_DIR, exist_ok=True)

        resumen = resumen_sesion(detector, csv_full_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_pdf = os.path.join(REPORTS_DIR, f"informe_{timestamp}.pdf")

        c = canvas.Canvas(ruta_pdf, pagesize=A4)
        width, height = A4

        # Encabezado
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Informe de la sesión — Lie Detector 4.0")

        # Texto del informe
        c.setFont("Helvetica", 10)
        text = c.beginText(50, height - 80)
        for line in resumen.split("\n"):
            text.textLine(line)
        c.drawText(text)

        # Añadir las gráficas
        y_position = height - 330
        for ruta_img in ("graf_lie.png", "graf_micro.png", "graf_microopt.png"):
            try:
                img = ImageReader(ruta_img)
                c.drawImage(img, 50, y_position, width=500, height=140, preserveAspectRatio=True)
                y_position -= 150
            except Exception as e:
                logger.warning(f"[pdf] No se pudo incluir imagen '{ruta_img}': {e}")

        c.save()
        ruta_absoluta = os.path.abspath(ruta_pdf)
        logger.info(f"[pdf] Informe PDF guardado como '{ruta_absoluta}'")
        messagebox.showinfo("Informe PDF", f"PDF generado en:\n{ruta_absoluta}")

    except Exception as e:
        logger.error(f"[pdf] Error al generar PDF: {e}", exc_info=True)
        messagebox.showerror("Error", "No se pudo generar el informe PDF.")

# ———— INFORME Y SESIÓN ————
def mostrar_informe():
    informe = resumen_sesion(detector, csv_full_path)
    logger.info(f"\n========= INFORME COMPLETO SESIÓN =========\n{informe}\n" + "="*48)
    with open("reporte_sesion.txt", "w", encoding="utf-8") as f:
        f.write(informe)
    with open("microexpresiones.log", "a", encoding="utf-8") as f:
        for evt in detector.events:
            f.write(f"{datetime.now()} – {evt.tipo} en frame {evt.frame_id}\n")
    logger.info("[informe] Archivos reporte_sesion.txt y microexpresiones.log escritos.")

    popup = tk.Toplevel(root)
    popup.title("Informe de la sesión")
    popup.geometry("920x680")
    tk.Button(popup, text="📄 Exportar como PDF", font=("Arial", 12), command=exportar_pdf_informe).pack(pady=6)

    txt = tk.Text(popup, height=10, wrap="word", font=("Consolas", 10))
    txt.insert("1.0", informe)
    txt.configure(state="disabled")
    txt.pack(fill="x", padx=6, pady=4)

    for ruta in ("graf_lie.png", "graf_micro.png", "graf_microopt.png"):
        try:
            img = Image.open(ruta).resize((880, 260), Image.LANCZOS)
            imtk = ImageTk.PhotoImage(img)
            lbl = tk.Label(popup, image=imtk)
            lbl.image = imtk
            lbl.pack(pady=4)
            logger.info(f"[informe] '{ruta}' mostrada en popup.")
        except Exception as e:
            logger.error(f"[informe] No se pudo mostrar '{ruta}': {e}")

    tk.Button(popup, text="Cerrar informe", font=("Arial", 12), command=popup.destroy).pack(pady=8)
    logger.info("[informe] Popup con informe mostrado.")

def resumen_sesion(det, path_csv="resultados_completos.csv"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import time
    from PIL import Image

    global inicio_sesion

    try:
        df = pd.read_csv(path_csv)
        logger.debug(f"[resumen_sesion] DataFrame leído correctamente: {df.shape[0]} filas, columnas: {list(df.columns)}")
    except Exception as e:
        logger.error(f"[resumen_sesion] Error leyendo CSV: {e}")
        return "No se pudo leer el CSV de resultados."

    if df.empty:
        return "No se capturó ningún dato útil."

    # Extraer datos del CSV
    lies   = df["lie"].values
    ears   = df["ear"].values
    pupils = df["pupil"].values
    yaws   = np.abs(df["yaw"].values)
    emos   = df["emo"].values.tolist()
    dur_s  = time.time() - inicio_sesion if inicio_sesion else len(df) / 30
    blink_tot = df["blinks"].iloc[-1] if "blinks" in df else 0
    micro_tot = df["micro"].sum() if "micro" in df else 0
    micro_opt = len(det.optical_micro_events) if hasattr(det, "optical_micro_events") else 0
    bpm = blink_tot / dur_s * 60 if dur_s > 0 else 0

    # Estadísticas finales
    lie_mean  = lies.mean()
    lie_max   = lies.max()
    emo_dom   = max(set(emos), key=emos.count)
    veredicto = "Posible Mentira" if lie_mean > det.TH_LIE else "Veraz"

    # Gráfica Lie-score
    plt.figure()
    plt.plot(lies)
    plt.xlabel("Frame")
    plt.ylabel("Lie-score")
    plt.title("Evolución del lie-score")
    plt.tight_layout()
    plt.savefig("graf_lie.png")
    plt.close()

    # Gráfica Micro-expresiones (si hay eventos)
    if det.events:
        xs = [e.frame_id for e in det.events]
        plt.figure()
        plt.bar(xs, np.ones_like(xs))
        plt.xlabel("Frame")
        plt.ylabel("Micro-expresión")
        plt.title("Micro-expresiones detectadas")
        plt.tight_layout()
        plt.savefig("graf_micro.png")
        plt.close()
    else:
        Image.new("RGB", (880, 260), color=(240, 240, 240)).save("graf_micro.png")

    # Gráfica Micro-optical
    if det.optical_micro_events:
        xs = [evt[0] for evt in det.optical_micro_events]
        plt.figure()
        plt.bar(xs, np.ones_like(xs))
        plt.xlabel("Frame")
        plt.ylabel("Micro-optical")
        plt.title("Microexpresiones detectadas (Optical Flow)")
        plt.tight_layout()
        plt.savefig("graf_microopt.png")
        plt.close()
    else:
        Image.new("RGB", (880, 260), color=(230, 230, 230)).save("graf_microopt.png")
    logger.info(
        f"[resumen_sesion] Lie_mean: {lie_mean:.3f}, Lie_max: {lie_max:.3f}, "
        f"Duración: {dur_s:.1f}s, Parpadeos: {blink_tot}, "
        f"Micro: {micro_tot}, Veredicto: {veredicto}"
    )
    # Texto del informe
    return (
        "────────  INFORME DE LA SESIÓN  ────────\n"
        f"Duración                   : {dur_s:4.1f} s\n"
        f"Lie-score medio            : {lie_mean:4.2f}\n"
        f"Lie-score máximo           : {lie_max:4.2f}\n"
        f"Parpadeos totales          : {blink_tot}\n"
        f"Frecuencia de parpadeo     : {bpm:4.1f} /min\n"
        f"N.º micro-expresiones      : {micro_tot}\n"
        f"Microexpresiones ópticas   : {micro_opt}\n"
        f"Emoción predominante       : {emo_dom}\n"
        f"Veredicto global           : {veredicto}\n"
        "Las gráficas se han guardado como graf_lie.png y graf_micro.png."
    )


def on_close():
    global after_id
    try:
        if after_id is not None:
            root.after_cancel(after_id)
    except Exception as e:
        logger.warning("[on_close] Error cancelando after_id: %s", e)

    cam.stop()
    cv2.destroyAllWindows()

    # 1️⃣  —— primero, asegura que todo está escrito ——
    if csv_buffer:
        csv_writer.writerows(csv_buffer)
        csv_writer_small.writerows(csv_buffer)
        logger.info("[on_close] Se volcaron %d filas pendientes.", len(csv_buffer))

    csv_file.flush(); csv_file_small.flush()             # ← ya estaba: ok
    # 2️⃣  —— ahora ya puedes generar el informe ——
    mostrar_informe()

    # 3️⃣  —— y finalmente cierras los ficheros siempre ——
    csv_file.close()
    csv_file_small.close()

    detector.save_baseline()
    root.destroy()


def process_frame(frame):
    global analyzing, zoom_face, inicio_sesion

    if frame is None:
        return frame

    if not analyzing:
        return frame

    if inicio_sesion is None:
        inicio_sesion = time.time()

    # ZOOM al rostro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detector(gray)
    logger.debug(f"[DEBUG] Faces detected: {len(faces) if faces else 0}")
    
    
    
    if faces and zoom_face:
        f = faces[0]
        x1 = max(0, f.left() - 30)
        y1 = max(0, f.top() - 30)
        x2 = min(frame.shape[1], f.right() + 30)
        y2 = min(frame.shape[0], f.bottom() + 30)
        if x2 - x1 > 60 and y2 - y1 > 60:
            frame = frame[y1:y2, x1:x2]

    data, lm = detector.analyze_frame(frame)
    detector.draw_overlays(frame)
    logger.debug(f"[DEBUG] analyze_frame data={data} lm={type(lm)}")
    # ─────── GUARDAR DATOS DEL FRAME ────────────────────────────────
    if data:
        flag_micro = int(getattr(detector, "_flag_micro_recent", False))

        fila = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get("emo", ""),
            f"{data.get('conf', 0):.3f}",
            f"{data.get('lie', 0):.4f}",
            f"{data.get('ear', 0):.3f}",
            f"{data.get('pupil', 0):.2f}",
            f"{data.get('yaw', 0):.1f}",
            f"{data.get('hr', 0):.1f}",
            flag_micro,
            detector.verdict() or "",
            len(detector.blink_times)
        ]
        try:
            csv_writer.writerow(fila)
            csv_writer_small.writerow(fila)
            if detector.frame_cnt % 150 == 0:
                csv_file.flush()
                csv_file_small.flush()
        except Exception as e:
            logger.warning("[CSV] %s — fila guardada en buffer", e)
            csv_buffer.append(fila)


    try:
        logger.debug(
            f"[BUFFERS] hist_len={len(detector.hist)} "
            f"ear_hist_len={len(detector._ear_hist)} "
            f"blink_times_len={len(detector.blink_times)} "
            f"feat_buf_len={len(detector.feat_buf)}"
        )
    except Exception as e:
        logger.warning(f"[BUFFERS] Error al imprimir buffers: {e}")

    veredicto = detector.verdict()
    logger.debug(
        "[landmarks] Detectados: %s, EAR: %s, pupil: %s, yaw: %s",
        lm is not None,
        data.get('ear', None) if data else None,
        data.get('pupil', None) if data else None,
        data.get('yaw', None) if data else None
    )

 


    # --- 2. Textos del HUD (SIEMPRE visibles)
    y0 = 100           # Más margen arriba
    line_height = 42   # Más separación

    def safe_fmt(val, fmt="{:.1f}"):
        import numpy as np
        try:
            if val is None:
                return "--"
            if isinstance(val, (float, int)):
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    return "--"
            return fmt.format(val)
        except Exception:
            return "--"

    def fmt_px(val):
        s = safe_fmt(val, "{:.0f}")
        return s if s == "--" else f"{s}px"

    def fmt_pct(val):
        s = safe_fmt(val, "{:.1f}")
        return s if s == "--" else f"{s}%"

    def fmt_deg(val):
        s = safe_fmt(val, "{:.0f}")
        return s if s == "--" else f"{s}°"

    y = y0

    if hasattr(detector, "last_pose3d") and detector.last_pose3d is not None:
        pitch, yaw, roll = detector.last_pose3d

    if data:
        txt_emo = f"{data.get('emo','--')} ({fmt_pct(data.get('conf',0)*100)})   lie:{safe_fmt(data.get('lie'))}"
        cv2.putText(frame, txt_emo, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        y += line_height

        txt_short = (
            f"EAR: {safe_fmt(data.get('ear'), '{:.3f}')}   "
            f"pupil: {fmt_px(data.get('pupil'))}   "
            f"yaw: {fmt_deg(data.get('yaw'))}   "
            f"blinks: {safe_fmt(data.get('blinks'), '{:d}')}"
        )
        cv2.putText(frame, txt_short, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.93, (255,255,255), 2)
        y += line_height

        # Estado BPM/parpadeos
        status, bpm = detector.get_blink_status()
        col = (100,255,100) if status == "NORMAL" else (40,120,255) if status == "CONCENTRACION" else (0,0,255)
        txt_bpm = f"BPM: {safe_fmt(bpm)}  ({status})"
        cv2.putText(frame, txt_bpm, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.97, col, 2)
        y += line_height

        # Veredicto
        cv2.putText(frame, f"Veredicto: {veredicto or '—'}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.97, (0,165,255), 2)
        y += line_height + 6

        # ======= SIEMPRE muestra Micro (Optical) última detectada =======
        last_micro = "--"
        if detector.optical_micro_events:
            last_micro = detector.optical_micro_events[-1][1]
        cv2.putText(frame, f"Micro (Optical): {last_micro}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.97, (255,140,0), 2)
        y += line_height

        # ======= SIEMPRE muestra DeepFace y FER aunque no haya micro =======
        deep_emo = data.get("deepface_emotion", "—")
        fer_emo  = data.get("fer_emotion",      "—")
        cv2.putText(frame,
                    f"DeepFace: {deep_emo}   |   FER: {fer_emo}",
                    (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        y += line_height

    else:
        cv2.putText(frame, "NO DATA", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.97, (0,0,255), 2)
        y += line_height


    # --- SIEMPRE muestra el mensaje de calibración si aplica, pero no oculta HUD ---
    if detector.frame_cnt <= detector.CALIBRATION_FRAMES:
        y_calib = y + 10
        cv2.putText(frame,
            ascii_safe(f"CALIBRANDO mantente quieto ({detector.frame_cnt}/{detector.CALIBRATION_FRAMES})"),
            (16, y_calib), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0,255,255), 2
        )

    # -- Visualización extra --
    if lm is not None:
        from visual_debug import draw_mesh
        draw_mesh(frame, lm, highlight=detector.events and detector.frame_cnt - detector.events[-1].frame_id < 30)

    if hasattr(detector, "feat_buf") and detector.feat_buf and len(detector.feat_buf[-1]) == 3:
        from visual_debug import draw_bars
        eye, brow, mouth = detector.feat_buf[-1]
        draw_bars(frame, eye, brow, mouth)

    # ─── Dibujar wire-frame 3D + ejes pose ─────────────────────────────
    if getattr(detector, "last_landmarks3d", None) is not None:
        from visual_debug import draw_landmarks3d

        # Pose almacenada (viene de face_alignment); si no existe, usa ceros
        pitch, yaw, roll = getattr(detector, "last_pose3d", (0.0, 0.0, 0.0))

        # Wire-frame y ejes; draw_axes=True es la opción por defecto
        draw_landmarks3d(
            frame,                       # frame BGR que se muestra
            detector.last_landmarks3d,   # np.ndarray (68,3)
            pitch=pitch,
            yaw=yaw,
            roll=roll                    # → ejes X (rojo), Y (verde), Z (azul)
        )

    if data:
        col = (0,255,0) if data.get("lie", 0) < 0.30 else (0,0,255)
        cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), col, 4)

    return frame


after_id = None
# ------------------------------------------------------------------
#  helpers   (sitúalos antes de process_frame y update_frame)
# ------------------------------------------------------------------
def ascii_safe(text):
    rep = {  # … tal como la tienes …
    }
    for k, v in rep.items():
        text = text.replace(k, v)
    return ''.join(c if 32 <= ord(c) < 127 else '?' for c in str(text))

def get_last_micro_events(events, window=30):
    """Devuelve un dict tipo: {tipo: frame_id} solo para eventos recientes (últimos X frames)."""
    last = {}
    for e in events[::-1]:
        if e.frame_id < detector.frame_cnt - window:
            break
        if e.tipo not in last:
            last[e.tipo] = e.frame_id
    return last

def get_last_optical_micro(opt_events, window=30):
    """Devuelve dict de micro ópticas recientes {tipo: frame_id}."""
    last = {}
    for fid, tipo in reversed(opt_events):
        if fid < detector.frame_cnt - window:
            break
        if tipo not in last:
            last[tipo] = fid
    return last

def put_line(img, text, *, y_ref, color=(255,255,255), scale=0.82, thick=2):
    cv2.putText(img, ascii_safe(text), (16, y_ref),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return y_ref + int(29 * scale)
# ------------------------------------------------------------------
#  ahora viene  process_frame()
# ------------------------------------------------------------------


def update_frame():
    global player, cam, usando_video, analyzing, after_id
    frame = None

    # Lee el frame correcto primero
    if usando_video:
        frame = player.read_frame() if player else None
    else:
        frame = cam.read() if cam else None

    if frame is not None:
        # Procesa/dibuja solo si hay frame válido
        if analyzing:
            detector.process(frame)
            frame_out = process_frame(frame)
        else:
            frame_out = frame  # muestra frame crudo

        # Validación de dimensiones
        panel_width = max(panel.winfo_width(), 1)
        panel_height = max(panel.winfo_height(), 1)
        if frame_out.shape[1] != panel_width or frame_out.shape[0] != panel_height:
            frame_out = cv2.resize(frame_out, (panel_width, panel_height), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img_pil, master=panel)
        panel.imgtk = imgtk  # Refuerza referencia
        panel.config(image=imgtk)
        panel.lift()  # Asegura que está visible arriba de todo
        panel.update_idletasks()  # Fuerza actualización inmediata

        # Para video: respeta el framerate del archivo
        if usando_video:
            interval = player.update_interval if player else 33
        else:
            interval = int(1000 / (getattr(cam, 'fps', 30) or 30))
        after_id = root.after(interval, update_frame)
    else:
        # No hay frame: si era video, detén el player
        if usando_video and player:
            print("[INFO] Video terminado.")
            player.stop()
        # Opcional: podrías poner aquí una imagen negra o mensaje en el panel

update_frame()
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()



