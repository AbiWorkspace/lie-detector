# micro_features.py
from collections import namedtuple, deque
import numpy as np
from microexpression_rules import (
    is_sorpresa, is_miedo, is_asco, is_labios_tensos,
    is_ceño_fruncido, is_fruncir_los_labios, is_microsonrisa,
    is_levantar_una_ceja, is_tension_ojos
)

MicroEvt = namedtuple("MicroEvt", ["tipo", "frame_id"])

# Buffers persistentes (ventanas temporales)
win_sorpresa     = deque(maxlen=4)
win_miedo        = deque(maxlen=4)
win_asco         = deque(maxlen=4)
win_labios       = deque(maxlen=4)

def feature_vector(pts):
    # Calculo de features a partir de puntos faciales (dlib 68 landmarks)
    eye_h = (pts[37][1] + pts[38][1] - pts[41][1] - pts[40][1]) / 2
    eye_w = max(pts[39][0] - pts[36][0], 1e-3)
    eye_open = np.clip((eye_h / eye_w) * 1.8, 0, 1)

    brow_raw = ((pts[19][1] + pts[24][1]) / 2) - ((pts[37][1] + pts[44][1]) / 2)
    eye_dist = pts[45][0] - pts[36][0]
    brow_norm = np.clip(-brow_raw / (0.35 * eye_dist), -0.2, 1.2)
    brow_raise = (brow_norm + 0.2) / 1.4

    mouth = pts[66][1] - pts[62][1]
    mouth_open = np.clip(mouth / (0.45 * eye_dist), 0, 1)

    return eye_open, brow_raise, mouth_open

def detect_micro(self, feat_buf, events, fid, extra_buffers=None):
    """
    Detecta microexpresiones Ekman básicas y otras reglas, usando los umbrales calibrados de self.
    Debe ser llamada desde la clase LieDetector, pasando self como primer argumento.
    """
    # Buffer mínimo requerido
    if len(feat_buf) < 6:
        return

    win = list(feat_buf)

    # ---- SAFE EXTRACTION (evita IndexError) ----
    # Solo agrega si el elemento tiene longitud suficiente
    eye   = [v[0] for v in win if isinstance(v, (list, tuple)) and len(v) >= 1]
    brow  = [v[1] for v in win if isinstance(v, (list, tuple)) and len(v) >= 2]
    mouth = [v[2] for v in win if isinstance(v, (list, tuple)) and len(v) >= 3]

    # Si alguno de los buffers está vacío, no procesar
    if not (eye and brow and mouth):
        return

    # Usa los umbrales calibrados si están definidos, si no usa None y la función toma el default.
    eye_thr   = getattr(self, 'eye_thr', None)
    mouth_thr = getattr(self, 'mouth_thr', None)
    brow_thr  = getattr(self, 'brow_thr', None)

    # ---- MICROEXPRESIONES EKMAN CON BUFFER ----
    win_sorpresa.append(int(is_sorpresa(eye, mouth, eye_thr, mouth_thr)))
    if sum(win_sorpresa) >= 2:
        events.append(MicroEvt("Sorpresa", fid))

    win_miedo.append(int(is_miedo(brow, eye, brow_thr, eye_thr)))
    if sum(win_miedo) >= 2:
        events.append(MicroEvt("Miedo", fid))

    win_asco.append(int(is_asco(brow, mouth, brow_thr, mouth_thr)))
    if sum(win_asco) >= 2:
        events.append(MicroEvt("Asco", fid))

    win_labios.append(int(is_labios_tensos(mouth)))
    if sum(win_labios) >= 2:
        events.append(MicroEvt("Labios_tensos", fid))

    # --- Extra: requiere buffers extendidos (manejo seguro) ---
    if extra_buffers:
        if "eyebrows_dist" in extra_buffers and "brow_shape" in extra_buffers:
            if is_ceño_fruncido(extra_buffers["eyebrows_dist"], extra_buffers["brow_shape"], brow_thr):
                events.append(MicroEvt("Ceño_fruncido", fid))
        if "mouth_shape" in extra_buffers and "lips_curve" in extra_buffers:
            if is_fruncir_los_labios(extra_buffers["mouth_shape"], extra_buffers["lips_curve"]):
                events.append(MicroEvt("Fruncir_labios", fid))
        if is_microsonrisa(mouth, getattr(self, 'baseline_mouth', 8)):
            events.append(MicroEvt("Microsonrisa", fid))
        if "eyebrow_l" in extra_buffers and "eyebrow_r" in extra_buffers:
            if is_levantar_una_ceja(extra_buffers["eyebrow_l"], extra_buffers["eyebrow_r"]):
                events.append(MicroEvt("Una_ceja", fid))
        if "eyelid_tension" in extra_buffers:
            if is_tension_ojos(extra_buffers["eyelid_tension"], eye_thr):
                events.append(MicroEvt("Tension_ojos", fid))
