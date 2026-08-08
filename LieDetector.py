# ── IMPORTS BÁSICOS ───────────────────────────────────────────────────────
import os, math, logging                                     #  ← añadimos logging
from logging.handlers import RotatingFileHandler              #  ← logger rotativo
import cv2, dlib                                              #  ← cv2 lo usas luego
import numpy as np                                            #  ← idem
from dataclasses import dataclass, field
from collections import deque, Counter, namedtuple, defaultdict
from typing import Deque, List, Tuple, Dict, Optional
import json, scipy.signal as sig

# ── LIBRERÍAS DE VISIÓN / ML ──────────────────────────────────────────────
import mediapipe as mp
from feat import Detector
from optical_flow import flow_overlay
import face_alignment
from skimage import io as skio
from deepface import DeepFace
from fer import FER
from tensorflow.keras.models import load_model
from utils_geom import compute_ear, solve_pose, pupil_diameter
from micro_features import feature_vector, detect_micro, MicroEvt
from ppg_chrom import PPGChrom
import face_recognition                                 # HOG acelerado

# ── REGLAS DE MICROEXPRESIÓN ──────────────────────────────────────────────
from microexpression_rules import (
    is_sorpresa, is_miedo, is_asco, is_labios_tensos,
    is_ceño_fruncido, is_dientes_apretados, is_sonrisa_falsa,
    is_boca_apretada, is_levantar_una_ceja, is_tension_ojos,
    is_sonrisa_genuina, compute_lip_curvature, auto_thr, MICRO_W
)

# ── CONFIGURAR LOG ROTATIVO ───────────────────────────────────────────────
LOG_PATH = "blink_debug.log"
handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3)
handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s"))

handler.setLevel(logging.DEBUG)          # ① NUEVO
root = logging.getLogger()
root.setLevel(logging.DEBUG)             # ② NUEVO

if not root.handlers:
    root.addHandler(handler)

print("Log en:", os.path.abspath(LOG_PATH))

# ── MediaPipe utilidades ──────────────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
def nostril_area(lm, idxs, w, h):
    pts = np.array([[lm.landmark[i].x*w, lm.landmark[i].y*h] for i in idxs],
                   dtype=np.int32)
    return cv2.contourArea(pts)

# --- PESOS INICIALES (puedes retocarlos a tu gusto) ---------------
BASE_WEIGHTS = {
    # ← métricas continuas 0-1
    "emo_fusion": 0.20,   # emoción negativa intensa (DeepFace+FER+CNN)
    "ear"       : 0.15,   # micro-cierre instantáneo
    "pupil"     : 0.08,   # dilatación relativa
    "hr"        : 0.08,   # variación cardiaca rPPG
    "yaw"       : 0.06,   # desvío lateral rostro
    "pitch"     : 0.05,   # inclinación vertical cabeza (nuevo)
    "roll"      : 0.04,   # cabeceo lateral
    "gaze"      : 0.05,   # movimiento/dispersión iris
    "blink"     : 0.12,   # dinámica freeze-rebound
    "bpm"       : 0.05,   # blinks-per-minute normalizado  (nuevo)
    "micro_opt" : 0.12,   # micro-expresión óptica reciente (nuevo)
    "smile"     : 0.04    # sonrisa genuina↓ / falsa↑ (ya tenías)
}

# --- vector fijo de características usadas por _logit() ------------
W_KEYS = list(BASE_WEIGHTS.keys())        # genera la lista automáticamente



logger = logging.getLogger(__name__)

Event = namedtuple("Event", ["tipo", "frame_id"])

# --------- MICROSONRISA: Peak detector y ventana temporal ----------
def detect_peak(buffer, delta=3):
    """
    Detecta un pico abrupto hacia arriba en los últimos frames.
    """
    if len(buffer) < 5:
        return False
    return buffer[-3] - buffer[-4] > delta and buffer[-3] - buffer[-2] > delta

micro_win_sonrisa = deque(maxlen=6)

def is_microsonrisa(mouth, baseline_mouth=8):
    """
    Detección robusta de microsonrisa usando baseline, delta y peak.
    """
    if len(mouth) < 5:
        return False
    delta = mouth[-3] - baseline_mouth
    peak = detect_peak(mouth, delta=3)
    ok = delta > 4 and peak
    logger.debug("[is_microsonrisa] Δ=%.2f peak=%s mouth[-3]=%.2f baseline=%.2f",
                delta, peak, mouth[-3], baseline_mouth)
    return _log("Microsonrisa", ok, mouth_m3=mouth[-3], baseline=baseline_mouth)

def is_microsonrisa_buffered(mouth, baseline_mouth=8):
    """
    Refuerzo usando una ventana temporal: si se detecta microsonrisa en al menos 2 frames de 6, la consideramos real.
    """
    res = is_microsonrisa(mouth, baseline_mouth)
    micro_win_sonrisa.append(int(res))
    return sum(micro_win_sonrisa) >= 2

def safe_get(of_pack, key, default=None):
    """
    Extrae un campo de un dict, DataFrame, Fex, etc. Si no existe, devuelve default y loguea.
    """
    if hasattr(of_pack, "columns") and key in of_pack.columns:
        return of_pack[key].iloc[0] if len(of_pack[key]) > 0 else default
    elif hasattr(of_pack, key):
        return getattr(of_pack, key)
    elif isinstance(of_pack, dict) and key in of_pack:
        return of_pack[key]
    else:
        print(f"[WARN][detector.py] '{key}' no encontrado en resultado py-feat. columns={getattr(of_pack, 'columns', type(of_pack))}")
        return default

class LieDetector:
    BLINK_MIN_FRAMES = 1
    CALIBRATION_FRAMES   = 60
    EMA_ALPHA            = 0.30
    EAR_UMBRAL_ABS       = 0.18
    YAW_SOFT_LIMIT       = 45
    YAW_HARD_LIMIT       = 20
    ROLL_LIMIT_DEG       = 20
    MICRO_WINDOW_FRAMES  = 150
    VENTANA              = 30
    TH_LIE               = 0.35
    TH_BLINKS_MIN        = 4
    LUM_EMA_ALPHA        = 0.35
    RPPG_BUF             = 150
    HR_WEIGHT            = 0.14
    GAZE_WEIGHT          = 0.10
    AU_BUF               = 30
    GAZE_BUF             = 30
    BLINK_MIN_NORMAL = 15
    BLINK_MAX_NORMAL = 20
    BLINK_UMBRAL_ANSIEDAD = 20
    BLINK_UMBRAL_CONCENTRACION = 10
    BLINK_WIN_SEC = 60 
    
    # ─────────────────────────────────────────────────────────
    #  Propiedad de acceso rápido al último lie-score
    # ─────────────────────────────────────────────────────────
    @property
    def lie_score(self):
        return (getattr(self, "_last_data", {}) or {}).get("lie", 0.0)
        
    # ─── dentro de class LieDetector ─────────────────────────────────────
    def draw_overlays(self, frame_bgr):
        from visual_debug import (
            draw_mesh, draw_bars, draw_liescore, draw_emotions,
            _shadow_text, draw_landmarks3d              # ahora existe
        )

        import numpy as np           # asegúrate de tenerlo arriba
        from micro_features import feature_vector

        # ── Destello micro-optical ────────────────────────────────────
        if getattr(self, "_flag_micro_recent", False):
            tipo = self.optical_micro_events[-1][1] if self.optical_micro_events else "?"
            _shadow_text(frame_bgr,
                        f"Micro-optical: {tipo}",
                        (20, frame_bgr.shape[0] - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6,
                        (0, 140, 255))      # naranja-rosa

        # ── Malla 3-D (wireframe) ─────────────────────────────────────
        if hasattr(self, "last_landmarks3d") and self.last_landmarks3d is not None:
            pitch, yaw, roll = getattr(self, "last_pose3d", (None, None, None))
            draw_landmarks3d(frame_bgr, self.last_landmarks3d, pitch, yaw, roll)

        # ── Barras ojo-ceja-boca usando puntos 2-D (MediaPipe o dlib) ─
        lm2d = getattr(self, "_last_lm", None)
        if lm2d is not None:
            # Conversión segura de landmarks a np.ndarray
            if hasattr(lm2d, "landmark") and hasattr(lm2d.landmark[0], "x"):
                # MediaPipe
                pts = np.array([(pt.x * frame_bgr.shape[1],
                                pt.y * frame_bgr.shape[0]) for pt in lm2d.landmark],
                            dtype=np.float32)
            else:
                # dlib / face_alignment
                pts = np.asarray(lm2d, dtype=np.float32)

            if pts.shape[0] >= 68:                       # condición mínima
                eye, brow, mouth = feature_vector(pts)
                draw_bars(frame_bgr, eye, brow, mouth)
        # ─── Flujo óptico (cada 2 frames ≈15 fps) ───────────────────────
        if self.frame_cnt % 2 == 0:
            flow_overlay(frame_bgr,          # ← se dibuja sobre el mismo frame
                        step=24,            # separación un poco mayor
                        color=(0,255,255))  # cian-verde para diferenciarse

        # ── lie-bar + emociones ───────────────────────────────────────
        data = getattr(self, "_last_data", None)
        if data is not None:
            draw_liescore(frame_bgr, data["lie"], anchor="bl", offset=(20, 20))
            draw_emotions(frame_bgr,
                        deep_emo=data.get("deepface_emotion"),
                        fer_emo=data.get("fer_emotion"),
                        anchor="tr",
                        offset=(12, 12))




    
    def _update_pupil_peak(self, pupil_norm: float) -> None:
            """
            Marca dilataciones súbitas de pupila (>4 %) con duración <0.5 s
            """
            if pupil_norm > 0.04:
                if not getattr(self, "pupil_peak", False):
                    self.pupil_peak = True
                    self._pupil_fid = self.frame_cnt
            elif getattr(self, "pupil_peak", False) and (self.frame_cnt - self._pupil_fid) > 15:
                self.pupil_peak = False

    def _compute_norms_flags(
            self, ear, pupil_corr, yaw_sm, pitch_sm, roll_sm,
            emo_score, emo_label, micro_weight, bpm):
        # ------------ métricas continuas -----------------
        norm = {
            "ear"  : np.clip((self.baseline["ear"] - ear) / 0.12, 0, 1) if self.baseline["ear"] else 0,
            "pupil": np.clip((pupil_corr - self.baseline["pupil"]) / 0.08, 0, 1) if self.baseline["pupil"] else 0,
            "yaw"  : np.clip(abs(yaw_sm)   / 20, 0, 1), 
            "au_tension": np.clip((self.au_vec[0]-1.5)/1.5, 0, 1) if hasattr(self,"au_vec") else 0,
            "au_brows"  : np.clip((self.au_vec[1]-1.5)/1.5, 0, 1) if hasattr(self,"au_vec") else 0,
            "pitch": np.clip(abs(pitch_sm) / 20, 0, 1),          
            "roll" : np.clip(abs(roll_sm)  / 20, 0, 1),
            "hr"   : np.clip((self.hr - self.hr_baseline) / 7.0, 0, 1) if self.hr_baseline and self.hr else 0,
            "gaze" : abs(self.gaze_ratio - 0.5) * 2 if self.gaze_ratio else 0,
            "bpm"  : np.clip((bpm - self.bpm_baseline) / (self.bpm_sigma or 8), 0, 1) if self.bpm_baseline else 0
        }
        # ------------ flags discretos --------------------
        flags = {
            "emo_neg"  : int(emo_label in ("Enojo","Miedo","Asco","Sorpresa") and emo_score > 0.45),
            "micro_opt": micro_weight,
            "blink_fr" : int(getattr(self, "blink_freeze", False)),
            "blink_rb" : int(getattr(self, "blink_reb",    False)),
            "pupil_pk" : int(getattr(self, "pupil_peak",   False)),
        }
        self._update_pupil_peak(norm["pupil"])
        return norm, flags





    def _logit(self, feats: Dict[str, float], *, return_parts=False):
        """
        Sigmoide logística.  Si return_parts=True devuelve (score, contribs)
        donde contribs = {clave: peso*valor}.
        """
        z = self.weights.get("bias", 0.0)
        contribs = {}
        for k, v in feats.items():
            w  = self.weights.get(k, 0.0)
            c  = w * v
            z += c
            contribs[k] = c
        z = max(min(z, 20), -20)          # anti-overflow
        score = 1.0 / (1.0 + math.exp(-z))
        return (score, contribs) if return_parts else score

    
    
    def __init__(self, ruta_modelo_emo: str, ruta_landmarks: str, use_openface: bool = True):
        logger.info("Inicializando LieDetector 4.0 …")

        self.nostril_buf = deque(maxlen=12)
        self.nostril_baseline = None
        BASE_WEIGHTS["nostril_flare"] = 0.05
        # ——— referencias seguras para el primer frame ———
        self._last_lm   = None        # landmarks MediaPipe del frame anterior
        self._last_data = None        # dict con los resultados del frame anterior
       


        self._last_micro_weight = 0.0


        self.calibrating_micro = True
        self.calib_frames_micro = 60  # Número de frames para baseline, ajústalo si quieres
        self.baseline_eye_open = []
        self.baseline_mouth_open = []
        self.baseline_brow_raise = []
        self.eye_thr = None
        self.mouth_thr = None
        self.brow_thr = None
        # ————— solución recomendada —————
        self.au_detector = Detector(
            device="cpu",
            face_detector="retinaface",    # o "mediapipe"
            landmark_model="mobilenet",    # ← cambia aquí
            action_units=True
        )

        self.au_buf = deque(maxlen=5)
        BASE_WEIGHTS.update({           # ↙ añade sólo una vez
            "au_tension": 0.07,
            "au_brows"  : 0.05
        })
        self.pitch_buf       = deque(maxlen=5)
        self.bpm_baseline    = None
        self.bpm_sigma       = None

        # Face detector / predictor / modelo
        self.detector  = self._cnn_face_detector
        self.predictor = dlib.shape_predictor(ruta_landmarks)
        self.model     = load_model(ruta_modelo_emo, compile=False)
        self.fps       = 30       # valor por defecto; se sobrescribe desde main.py
        self.win_smile = 3        # ventana temporal para sonrisas (se recalcula)

        # Face-alignment 2D/3D
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu')
        # DeepFace detector (nada que inicializar, es funcional)
        # FER (emotions)
        self.fer_detector = FER(mtcnn=True)



        # ─── Pesos logísticos (coef = 3×peso_heurístico) ──────────────────────
        DEFAULT_LOG_WEIGHTS = {"bias": -1.0}
        for k, v in BASE_WEIGHTS.items():
            DEFAULT_LOG_WEIGHTS[k] = 3*v


        import numpy as np

        try:
            self.weights = np.load("weights.npy", allow_pickle=True).item()
            # Asegura que todas las claves estén presentes
            for k in DEFAULT_LOG_WEIGHTS:
                self.weights.setdefault(k, DEFAULT_LOG_WEIGHTS[k])
            logger.info("[Pesos] weights.npy cargado correctamente. Claves faltantes rellenadas.")
        except Exception as e:
            logger.warning("[Pesos] %s → uso DEFAULT_LOG_WEIGHTS", e)
            self.weights = DEFAULT_LOG_WEIGHTS.copy()


        # Históricos y calibración
        self.hist_genuine = deque(maxlen=10)
        self.hist_false   = deque(maxlen=10)
        self.init_frames     = 60
        self.curve_baseline  = []
        self.eye_baseline    = []
        self.thr_curve       = 6.0
        self.thr_eye         = 52.0

        # ─── dentro de LieDetector.__init__ ──────────────────────────────
        self._ear_med       = deque(maxlen=5)   # filtro mediana
        self._ear_ema       = None
        self._blink_state   = "open"
        self._blink_start_f = 0
      



        self.emotion_labels = ['Enojo','Asco','Miedo','Felicidad','Tristeza','Sorpresa','Neutral']
        self.prev_gray      = None
        self.optical_micro_events = []

        # MediaPipe FaceMesh
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.ear_mu     = None      # ← añadir
        self.ear_sigma  = None      # ← añadir
        self.bpm_baseline = None
        self.bpm_sigma    = None
        # ————— Buffers para microexpresiones —————
        self.eye_buffer           = deque(maxlen=8)
        self.mouth_buffer         = deque(maxlen=8)
        self.brow_buffer          = deque(maxlen=8)
        self.mouth_shape_buffer   = deque(maxlen=8)
        self.lips_curve_buffer    = deque(maxlen=8)

        # ————— Buffers adicionales que faltaban —————
        self.jaw_buffer           = deque(maxlen=8)
        self.eyebrows_dist_buffer = deque(maxlen=8)
        self.brow_shape_buffer    = deque(maxlen=8)
        self.eyebrow_l_buffer     = deque(maxlen=8)
        self.eyebrow_r_buffer     = deque(maxlen=8)
        self.mouth_corners        = deque(maxlen=8)
        self.eyelid_tension       = deque(maxlen=8)

        self.YAW_HARD_LIMIT = 60


        # Historial extra, por si lo usas
        self.pupil_history        = deque(maxlen=30)

        # Resto de vars de estado
        self.hist        = deque(maxlen=self.VENTANA)
        self.baseline    = {"ear": None, "pupil": None}
        self.frame_cnt   = 0
        self._ear_hist: Deque[float] = deque(maxlen=200)   # guarda los EAR crudos
        self._ear_calibrated        = False
        self.ear_mu = None      # μ del EAR  (se llena tras la 1.ª calibración)
        self.ear_sd = 1e-3      # σ “dummy”  para que exista el atributo
        self._ear_last_recal = 0 
        self.STABLE_LAP_VAR = 80
        self.STABLE_SECONDS = 5

        self._stable_frames = 0


        self.feat_buf    = deque(maxlen=30)
        self.events      = []
        self.yaw_buf     = deque(maxlen=5)

        self.blink_counter = 0
        self.blink_start   = False
        self.last_ear      = 0
        self.blink_times   = deque(maxlen=150)

        self.rppg_buf      = deque(maxlen=self.RPPG_BUF)
        self.hr            = None
        self.hr_baseline   = None

        self.gaze_ratio    = None
        self.roll_sm       = 0.0

        self.lum_ema       = None
        self.last_pupil    = 0

        self.use_openface  = use_openface
        self.aus_buffer    = deque(maxlen=self.AU_BUF)
        self.gaze_buffer   = deque(maxlen=self.GAZE_BUF)

        self.ppg_chrom     = PPGChrom(fps=30)
        self.fps = 30
        self.BLINK_MIN_FRAMES = max(2, int(0.08 * self.fps))
        # ≤450 ms ⇒ 0.45 s · fps
        self.BLINK_MAX_FRAMES = int(0.45 * self.fps)
        self.win_smile = 3
        
        # Microsonrisa
        self.calibrated_micro = False
        self.baseline_mouth   = 8
        self._load_baseline()

        logger.info("LieDetector 4.0 listo ✅")
        
    def calibrate_micro_baseline(self, eye_open, mouth_open, brow_raise):
        """
        Almacena los primeros N frames para calcular los baseline y umbrales
        dinámicamente.
        """
        if self.calibrating_micro:
            self.baseline_eye_open.append(eye_open)
            self.baseline_mouth_open.append(mouth_open)
            self.baseline_brow_raise.append(brow_raise)

            if len(self.baseline_eye_open) >= self.calib_frames_micro:
                self.eye_thr = np.mean(self.baseline_eye_open) + 2*np.std(self.baseline_eye_open)
                self.mouth_thr = np.mean(self.baseline_mouth_open) + 2*np.std(self.baseline_mouth_open)
                self.brow_thr = np.mean(self.baseline_brow_raise) + 2*np.std(self.baseline_brow_raise)
                self.calibrating_micro = False
                print(f"[CALIB] Micro umbrales: eye={self.eye_thr:.2f} mouth={self.mouth_thr:.2f} brow={self.brow_thr:.2f}")
    
    def _update_aus(self, frame_bgr):
        """
        Extrae intensidades de Action Units (AUs) y mantiene un buffer suavizado.
        Convierte el frame a RGB y a PIL.Image para evitar el error
        "'numpy.ndarray' object has no attribute 'read'".
        """
        try:
            # 1) OpenCV → RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 2) ndarray → PIL.Image (algunas versiones de feat lo necesitan)
            from PIL import Image
            pil_img = Image.fromarray(frame_rgb)

            # 3) Detección AU
            out = self.au_detector.detect_image(pil_img)
            if out.au_outputs is None or out.au_outputs.empty:
                return

            # 4) Procesado y suavizado
            au = out.au_outputs.iloc[0]
            au_tension = float(au[["AU14", "AU17", "AU24"]].mean())  # labios/presión
            au_brows   = float(au[["AU01", "AU04"]].mean())          # cejas

            self.au_buf.append((au_tension, au_brows))
            self.au_vec = np.mean(self.au_buf, axis=0)

        except Exception as e:
            logger.debug(f"[AUs] {e}")


    def _update_blinks(self, ear_raw: float, lap_var: float, yaw: float) -> None:
        """
        Autómata open→closing→closed→opening→open con umbrales AUTO-ajustados
        y sin inflar el EAR cuando la cabeza gira.
        """
        # 0) Salida rápida si la imagen está borrosa
        if lap_var < 10:          # poca textura: evita falsos positivos
            return


        # 2) Suavizado muchísimo más ligero  (mediana-3  +  EMA α=0.15)
        if not hasattr(self, "_ear_med3"):
            self._ear_med3 = deque(maxlen=3)
            self._ear_ema  = ear_corr
        self._ear_med3.append(ear_corr)
        ear_med = float(np.median(self._ear_med3))
        ear     = 0.15*ear_med + 0.85*self._ear_ema
        self._ear_ema = ear

       

        # 3) Baseline μ/σ con z-score (top-20 %)  — recalibra cada 30 s
        self._ear_hist.append(ear)
        if (   not hasattr(self, "_ear_last_recal")
            or self.frame_cnt - self._ear_last_recal > self.fps*30):
            vals = np.sort(np.array(self._ear_hist, np.float32))
            if len(vals) >= 120:
                top20        = vals[-int(len(vals)*0.20):]
                self.ear_mu  = float(top20.mean())
                self.ear_sd  = float(top20.std() or 1e-3)
                self._ear_last_recal = self.frame_cnt
                self._ear_hist.clear()
                logging.info("[Blink] μ=%.3f  σ=%.3f (re-cal)", self.ear_mu, self.ear_sd)
                print(f"[EAR]  μ={self.ear_mu:.3f}  σ={self.ear_sd:.3f}  "
                    f"thrC={self.ear_mu - 1.3*self.ear_sd:.3f}  "
                    f"thrO={self.ear_mu - 0.4*self.ear_sd:.3f}")
        
        if not hasattr(self, "ear_mu"):
            return                                  # espera baseline

        # ─── salvaguarda — sal de la función si aún no hay baseline —──
        if self.ear_mu is None or self.ear_sd is None:
            return                      # todavía calibrando

        # 4) Umbrales dinámicos  (ahora SÍ podemos usarlos)
        thr_close = self.ear_mu - 1.3*self.ear_sd   # ≃  μ – 1.3σ
        thr_open  = self.ear_mu - 0.4*self.ear_sd 

         
        # DEBUG 1 – mira cada frame
        print(f"[BLINK] f={self.frame_cnt:05d} st={self._blink_state:<7} "
            f"ear={ear_raw:.3f}->{ear:.3f}  thrC={thr_close:.3f}  thrO={thr_open:.3f}")
            
        # 5) FSM ───────────────────────────────────────────────────
        st = self._blink_state
        if st == "open":
            if ear < thr_close:
                self._blink_state   = "closing"
                self._blink_start_f = self.frame_cnt
        elif st == "closing":
            if ear < thr_close * 0.9:
                self._blink_state = "closed"
            elif ear > thr_open:                    # falsa alarma
                self._blink_state = "open"
        elif st == "closed":
            dur = self.frame_cnt - self._blink_start_f
            if ear > thr_open:
                if 1 <= dur <= int(0.45*self.fps):  # 30–450 ms
                    self.blink_counter += 1
                    self.blink_times.append(self.frame_cnt)
                    self.add_event("Blink", self.frame_cnt)
                    print(f"[BLINK] ✓ blink #{self.blink_counter}  dur={dur} frames")
                    logging.debug("[Blink] ✓ blink %d (dur=%d f)", 
                                self.blink_counter, dur)
                self._blink_state = "opening"
        elif st == "opening":
            if ear > thr_open*1.05:
                self._blink_state = "open"

        # Seguridad: resetea si lleva demasiado en CLOSED
        if (self._blink_state == "closed" 
            and self.frame_cnt - self._blink_start_f > int(0.6*self.fps)):
            self._blink_state = "open"




 # ────────────────────────────────────────────────────────────────────
    #  Ejecuta una 2ª calibración cuando llevamos ≥10 s de vídeo
    # ────────────────────────────────────────────────────────────────────
    def _maybe_recalibrate_ear(self) -> None:
        """
        Segunda pasada de calibración: usa las últimas muestras para
        actualizar μ y σ si aún no se hizo.  Sólo una vez por sesión.
        """
        # sólo si la cabeza está razonablemente recta
        if abs(np.mean(self.yaw_buf)) > 20:
            return

        if getattr(self, "_ear_recalibrated", False) or len(self._ear_hist) < 300:
            return

        logger.info("[CALIB-extra] Re-calibrando EAR con datos frescos…")
        self._ear_calibrated = False           # deja que _calibrate_… actúe
        self._ear_hist = deque(list(self._ear_hist)[-300:], maxlen=300)
        self._calibrate_ear_baseline()
        self._ear_recalibrated = True


    def process(self, frame):
                data, self._last_lm = self.analyze_frame(frame, show_debug_mesh=False)
                self._last_data = data
                return data
    def draw_landmarks(self, frame_rgb):
                if hasattr(self, "_last_lm") and self._last_lm is not None:
                    from visual_debug import draw_mesh
                    draw_mesh(frame_rgb, self._last_lm)
    def draw_emotion_bars(self, frame_rgb):
                """
                Dibuja las barras (ojo-ceja-boca) si existe al menos un vector de
                características reciente.
                """
                if not self.feat_buf:
                    return
                eye, brow, mouth = self.feat_buf[-1]
                from visual_debug import draw_bars
                draw_bars(frame_rgb, eye, brow, mouth)

    
    def verdict(self) -> str:
        """
        Devuelve 'Mentira' si el lie-score medio de los últimos VENTANA
        frames (50 por defecto) supera 0.65; en caso contrario 'Veraz'.
        """
        if len(self.hist) < self.VENTANA:
            return ""
        
        lie_avg = np.mean([h["lie"] for h in list(self.hist)[-self.VENTANA:]])
        logger.debug("[verdict] lie_avg=%.2f  (umbral 0.65)", lie_avg)

        return "Mentira" if lie_avg > 0.65 else "Veraz"


    
    # ───────────────────────────────────────────────────────────────
    #  Sustituye TODO el método por esta versión
    # ───────────────────────────────────────────────────────────────
    def _calibrate_ear_baseline(self) -> None:
        """
        Calcula μ y σ del EAR usando sólo el 30 % superior de las muestras
        recogidas (ojos abiertos).  Evita que pestañeos o semicierres contaminen
        la media y coloca un suelo de seguridad en 0.18–0.22.
        """
        if self._ear_calibrated or len(self._ear_hist) < 120:          # necesitamos algo más de 3 s
            return

        vals = np.sort(np.array(self._ear_hist, dtype=np.float32))
        top  = vals[-int(len(vals) * 0.30):]                           # 30 % superior
        self.ear_mu    = float(top.mean())
        self.ear_sigma = float(top.std() or 1e-3)

        # Salvaguarda: μ nunca por debajo de 0.18
        if self.ear_mu < 0.18:
            logger.warning("[CALIB] EAR μ=%.3f demasiado bajo – forzado a 0.22", self.ear_mu)
            self.ear_mu = 0.22

        logger.info("[CALIB] EAR listo   μ=%.3f  σ=%.3f", self.ear_mu, self.ear_sigma)
        self._ear_calibrated = True


    
    def _calibrate_smile_thresholds(self):
        if len(self.lips_curve_buffer) >= 1 and len(self.eye_buffer) >= 1:
            if self.frame_cnt <= self.init_frames:
                self.curve_baseline.append(self.lips_curve_buffer[-1])
                self.eye_baseline.append(self.eye_buffer[-1])
            elif self.frame_cnt == self.init_frames + 1:
                μc, σc = np.mean(self.curve_baseline), np.std(self.curve_baseline)
                μe, σe = np.mean(self.eye_baseline),   np.std(self.eye_baseline)
                self.thr_curve = μc + 2*σc
                self.thr_eye   = μe + 2*σe
                self.hist_genuine.clear()
                self.hist_false.clear()
                logger.info(f"[Calibración] thr_curve={self.thr_curve:.1f}, thr_eye={self.thr_eye:.1f}")

    _ear         = staticmethod(compute_ear)
    _solve_pose  = staticmethod(solve_pose)
    _pupil_diam  = staticmethod(pupil_diameter)
    def _feat_vector(self, pts): return feature_vector(pts)
    def _detect_micro(self, fid, extra_buffers=None): detect_micro(self.feat_buf, self.events, fid, extra_buffers)

    # ─── reemplaza _blinks_per_minute() COMPLETO ────────────────────
    def _blinks_per_minute(self) -> float:
        if len(self.blink_times) < 2:
            return 0.0

        f_now  = self.frame_cnt
        win_f  = self.BLINK_WIN_SEC * self.fps
        # ① mantén sólo los últimos 60 s
        self.blink_times = deque([f for f in self.blink_times
                                if f_now - f <= win_f],
                                maxlen=self.blink_times.maxlen)
        if len(self.blink_times) < 2:
            return 0.0

        ibi = np.diff(self.blink_times) / self.fps      # seg
        # ② recorte 10-90 % y filtro outliers extremos
        ibi = ibi[(ibi > 0.1) & (ibi < 2.0)]
        if len(ibi) == 0:
            return 0.0
        ibi_sorted = np.sort(ibi)
        p10, p90 = np.percentile(ibi_sorted, [10, 90])
        ibi_trim  = ibi_sorted[(ibi_sorted >= p10) & (ibi_sorted <= p90)]
        return 60.0 / np.median(ibi_trim)


    def add_event(self, tipo, frame_id):
        logger.debug("[EVENT] Se agregó evento: %s en frame %d", tipo, frame_id)
        self.events.append(Event(tipo=tipo, frame_id=frame_id))


    def get_blink_status(self):
            """
            Retorna el estado del parpadeo:
            - 'NORMAL'
            - 'ANSIEDAD/MENTIRA'
            - 'CONCENTRACION'
            Además retorna el valor numérico de BPM.
            """
            bpm = self._blinks_per_minute()
            if bpm > self.BLINK_UMBRAL_ANSIEDAD:
                status = "ANSIEDAD/MENTIRA"
            elif bpm < self.BLINK_UMBRAL_CONCENTRACION:
                status = "CONCENTRACION"
            else:
                status = "NORMAL"
            return status, bpm
    
    def _update_rppg(self, color_roi):
        g = np.mean(color_roi[:, :, 1])
        self.rppg_buf.append(g)
        if len(self.rppg_buf) < self.RPPG_BUF:
            return
        sig_arr = np.array(self.rppg_buf, dtype=np.float32)
        sig_arr -= sig_arr.mean()
        b, a = sig.butter(2, [0.7 / self.fps, 3 / self.fps], btype="band")
        sig_f = sig.filtfilt(b, a, sig_arr)

        freqs = np.fft.rfftfreq(len(sig_f), d=1 / self.fps)
        spec  = np.abs(np.fft.rfft(sig_f))

        mask  = (freqs >= 0.8) & (freqs <= 3.0)
        
        # ---- al final de _update_rppg() ----
        if self.hr and self.hr_baseline is None and self.frame_cnt > 10*self.fps:
            self.hr_baseline = self.hr
            logger.info("[Calibración] HR baseline = %.1f ppm", self.hr_baseline)

        if not np.any(mask):
            return
        self.hr = freqs[mask][np.argmax(spec[mask])] * 60
    def _cnn_face_detector(self, gray_img):
        """
        Usa face_recognition con HOG (más rápido que CNN).
        Cambia a 'cnn' si usas GPU o haces threading.
        """
        face_locations = face_recognition.face_locations(gray_img, model="hog")  # ← CAMBIADO A HOG
        rects = []
        for (top, right, bottom, left) in face_locations:
            rect = dlib.rectangle(left, top, right, bottom)
            rects.append(rect)
        return rects

    def _gaze_ratio_from_iris(self, lm3d, w, h):
        try:
            c = lm3d.landmark[468]
            left = lm3d.landmark[33]; right = lm3d.landmark[133]
            cx,lx,rx = c.x*w, left.x*w, right.x*w
            return np.clip((cx-lx)/max(rx-lx,1e-3), 0, 1)
        except Exception:
            return None

    def _deepface_dict(self):
        """Devuelve siempre un dict, nunca una lista, para DeepFace."""
        val = getattr(self, 'deepface_last', {})
        if isinstance(val, list):
            return val[0] if len(val) > 0 else {}
        return val if isinstance(val, dict) else {}

    def analyze_frame(self, frame, show_debug_mesh: bool = True):
        """
        Procesa un frame de vídeo y devuelve (resultado, landmarks_2d).

        ─ Parámetros ───────────────────────────────────────────────
        frame : np.ndarray  — frame BGR original.
        show_debug_mesh     — si True, dibuja overlay de depuración
                              (se usa en la GUI / main.py).

        ─ Devuelve ─────────────────────────────────────────────────
        (dict con métricas, landmarks MediaPipe | None)
        """
        # ───────────────────────────── contadores ─────────────────────────────
        self.frame_cnt  += 1
        self.last_frame = frame
        self._calibrate_smile_thresholds()

        # ───────────────────────────── pre-chequeos ───────────────────────────
        if frame is None:
            logger.warning("Frame es None")
            return None, None
        logger.info(f"Frame shape: {frame.shape}")

        # ───── conversión y ecualización (se hace una sola vez) ──────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # ────────────────── micro-expresión óptica (UNA llamada) ──────────────
        res_micro = self.detect_optical_micro(gray)          # ← llamada única
        if res_micro:
            tipo_micro, peso_micro = res_micro               # tupla (tipo, peso)
            self._last_micro_type  = tipo_micro
        else:
            tipo_micro, peso_micro = None, 0.0

        # ─────────────────── Landmarks 3-D (face_alignment) ───────────────────
        landmarks_3d = None
        try:
            preds_3d = self.fa.get_landmarks(frame)
            if preds_3d is not None and len(preds_3d):
                landmarks_3d = preds_3d[0]

                nose, chin          = landmarks_3d[30], landmarks_3d[8]
                left_eye, right_eye = landmarks_3d[36], landmarks_3d[45]

                vector  = chin - nose
                pitch   = np.arctan2(vector[1],
                                    np.linalg.norm(vector[[0, 2]])) * 180 / np.pi
                yaw     = np.arctan2(vector[0], vector[2]) * 180 / np.pi
                eye_vec = right_eye - left_eye
                roll    = np.arctan2(eye_vec[1], eye_vec[0]) * 180 / np.pi

                logger.info("[face-alignment] pitch=%.2f  yaw=%.2f  roll=%.2f",
                            pitch, yaw, roll)

                self.last_pose3d      = (pitch, yaw, roll)
                self.last_landmarks3d = landmarks_3d
            else:
                self.last_pose3d      = (None, None, None)
                self.last_landmarks3d = None
        except Exception as e:
            logger.warning("[face-alignment] No se pudo obtener landmarks: %s", e)
            self.last_pose3d      = (None, None, None)
            self.last_landmarks3d = None

        # ─────────────── Landmarks 2-D (MediaPipe FaceMesh) ───────────────────
        lm = None
        try:
            out = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            logger.info("[DEBUG] MediaPipe process: %s", out is not None)

            if hasattr(out, "multi_face_landmarks") and out.multi_face_landmarks:
                lm = out.multi_face_landmarks[0]
                logger.info("[DEBUG] Landmarks detectados: %d puntos.",
                            len(lm.landmark))
            else:
                logger.warning("[DEBUG] NO se detectaron landmarks con MediaPipe.")
        except Exception as e:
            logger.warning("FaceMesh falló: %s", e)

        # ─────────────── Emociones (DeepFace  +  FER) ─────────────────────────
        try:
            deepface_res = DeepFace.analyze(
                frame, actions=['emotion'], enforce_detection=False
            )
            logger.debug("[deepface] %s", deepface_res)
            self.deepface_last = deepface_res
        except Exception as e:
            logger.warning("[deepface] No se pudo analizar: %s", e)

        try:
            fer_res = self.fer_detector.detect_emotions(
                cv2.resize(frame, (640, 480))
            )
            self.fer_last_raw = fer_res
            logger.debug("[fer] %s", fer_res)
            if fer_res and 'emotions' in fer_res[0]:
                fer_emo           = max(
                    fer_res[0]['emotions'],
                    key=fer_res[0]['emotions'].get
                )
                self.fer_last_emo = fer_emo
                logger.debug("[fer] Emoción dominante: %s", fer_emo)
        except Exception as e:
            logger.warning("[fer] Error en FER: %s", e)

        # ───────────────────── detección de rostro HOG ────────────────────────
        faces = self.detector(gray)
        self._update_aus(frame)

        # ──────── rama rápida si NO hay rostro dlib / predictor ──────────────
        if not faces or (faces and self.predictor is None):
            if lm is not None:
                pts = np.array(
                    [(pt.x * frame.shape[1], pt.y * frame.shape[0])
                     for pt in lm.landmark],
                    dtype=np.float32
                )
                logger.info("[DEBUG] MediaPipe pts.shape=%s", pts.shape)

                # -------- EAR (con puntos MediaPipe) --------
                try:
                    def ear_mp(arr, idx):
                        A = np.linalg.norm(arr[idx[1]] - arr[idx[5]])
                        B = np.linalg.norm(arr[idx[2]] - arr[idx[4]])
                        C = np.linalg.norm(arr[idx[0]] - arr[idx[3]])
                        return (A + B) / (2.0 * C)

                    left_idx  = [33, 160, 158, 133, 153, 144]
                    right_idx = [362, 385, 387, 263, 373, 380]

                    if pts.shape[0] > max(right_idx):
                        ear_left  = ear_mp(pts, left_idx)
                        ear_right = ear_mp(pts, right_idx)
                        ear       = (ear_left + ear_right) / 2.0
                    else:
                        logger.warning("[MediaPipe] Pocos puntos para EAR")
                        ear = 0.0

                    pupil = 0.0
                    if pts.shape[0] > 477:
                        try:
                            pupil = self._pupil_diam(
                                lm, [474, 475, 476, 477],
                                frame.shape[1], frame.shape[0]
                            )
                        except Exception as e:
                            logger.warning("[MediaPipe] Pupil error: %s", e)
                            pupil = 0.0

                    yaw_val = (self.last_pose3d[1]
                               if self.last_pose3d
                               and self.last_pose3d[1] is not None else 0.0)

                except Exception as e:
                    logger.warning("[MediaPipe] Error calc features: %s", e)
                    ear, pupil, yaw_val = 0.0, 0.0, 0.0

                res_stub = {
                    "emo"   : getattr(self, "last_emo", "Neutral"),
                    "conf"  : 0.0,
                    "ear"   : float(ear),
                    "yaw"   : float(yaw_val),
                    "pupil" : float(pupil),
                    "lie"   : 0.0,
                    "blinks": len(self.blink_times),
                    "hr"    : self.hr or 0
                }
                if hasattr(self, "fer_last_emo"):
                    res_stub["fer_emotion"] = self.fer_last_emo

                self.hist.append(res_stub)
                return res_stub, lm

            # ─ sin landmarks 2D tampoco ─
            self.feat_buf.clear()
            res_prev = getattr(self, "last_valid_res", {
                "emo": "Neutral", "conf": 0.0, "ear": 0.0, "yaw": 0.0,
                "pupil": 0.0, "lie": 0.0, "blinks": 0, "hr": 0
            })
            self.hist.append(res_prev)
            return res_prev, lm

        # ────────── pipeline normal (rostro + predictor dlib ok) ──────────────
        res = self._process_face(gray, faces[0], lm, frame.shape)

        # ───── baseline micro-sonrisa tras 60 frames de buffer ─────
        if not self.calibrated_micro and len(self.mouth_buffer) >= 60:
            self.baseline_mouth  = np.median(list(self.mouth_buffer)[:10])
            self.calibrated_micro = True
            logger.info("[Calibración] baseline boca = %.2f", self.baseline_mouth)

        # ─────────── mantenimiento de históricos / EMA ────────────
        if res:
            self.last_valid_res = res.copy()

            # —— primeras 60 f: baseline ear / pupil ——
            if self.frame_cnt <= self.CALIBRATION_FRAMES:
                for k in ("ear", "pupil"):
                    self.baseline[k] = (
                        res[k] if self.baseline[k] is None
                        else (self.baseline[k] + res[k]) / 2
                    )

            # —— baseline HR después de 60 f estables ——
            if (self.frame_cnt > self.CALIBRATION_FRAMES
                    and self.hr and self.hr_baseline is None):
                self.hr_baseline = self.hr

            # —— estadísticos blink / ear al final de calibración ——
            if self.frame_cnt == self.CALIBRATION_FRAMES and self.hist:
                ears           = [h['ear'] for h in self.hist]
                self.ear_mu    = np.mean(ears)
                self.ear_sigma = np.std(ears) or 1e-3

                raw      = list(self.blink_times)
                bpm_seq  = np.diff(raw) / 30 * 60
                if bpm_seq.size:
                    self.bpm_baseline = bpm_seq.mean()
                    self.bpm_sigma    = bpm_seq.std() or 5

            # —— suavizado EMA en tiempo real ——
            if self.hist:
                prev = self.hist[-1]
                for k in ("ear", "pupil", "lie"):
                    res[k] = (self.EMA_ALPHA * res[k]
                              + (1 - self.EMA_ALPHA) * prev[k])

            self.hist.append(res)
        else:
            res = getattr(self, "last_valid_res", {
                "emo": "Neutral", "conf": 0.0, "ear": 0.0, "yaw": 0.0,
                "pupil": 0.0, "lie": 0.0, "blinks": 0, "hr": 0
            })
            self.hist.append(res)

        # ─────────────── dibujado overlay de depuración ─────────────
        if res and show_debug_mesh:
            from visual_debug import draw_mesh, draw_bars, draw_landmarks3d
            draw_mesh(frame, lm,
                      highlight=self.events and
                      self.frame_cnt - self.events[-1].frame_id < 30)
            if self.feat_buf:
                eye, brow, mouth = self.feat_buf[-1]
                draw_bars(frame, eye, brow, mouth)
            if self.last_landmarks3d is not None:
                pitch, yaw, roll = self.last_pose3d
                draw_landmarks3d(frame, self.last_landmarks3d,
                                 pitch, yaw, roll)

        # ─────────── logging de detección o ausencia de rostro ──────
        if faces:
            logger.info("Rostro detectado en frame %d", self.frame_cnt)
        else:
            logger.warning("No se detectó rostro en frame %d", self.frame_cnt)

        # ─────── estado de parpadeo / BPM para GUI ───────
        if res is not None:
            status, bpm_now   = self.get_blink_status()
            res["blink_status"] = status
            res["bpm"]          = bpm_now
            logger.debug("[Blink] BPM=%.1f  Estado=%s", bpm_now, status)

        # ─────── info extra de DeepFace to GUI ───────
        if res is not None:
            deepface_info           = self._deepface_dict()
            res["deepface_age"]     = deepface_info.get('age')
            res["deepface_gender"]  = deepface_info.get('gender')
            res["deepface_race"]    = deepface_info.get('dominant_race')
            res["deepface_emotion"] = (deepface_info.get('dominant_emotion')
                                       or "Neutral")
            res["fer_emotion"]      = getattr(self, 'fer_last_emo', "Neutral")

        return res, lm




# ─── helper privado ─────────────────────────────────────────────
    def _ear_smooth(self, ear: float) -> float:
        """Mediana (5) + EMA(α=0.4) para estabilizar EAR."""
        self._ear_med.append(ear)
        med = float(np.median(self._ear_med))
        self._ear_ema = med if self._ear_ema is None \
            else 0.4*med + 0.6*self._ear_ema
        return self._ear_ema


    # --------------------------------------------------------------------
    #  MÉTODO COMPLETO  detect_optical_micro()
    #  – devuelve SIEMPRE una tupla (tipo, peso) o bien None
    # --------------------------------------------------------------------
    def detect_optical_micro(self, frame_gray):
        """
        Detecta micro-expresiones vía flujo óptico + reglas heurísticas.
        Devuelve:

            (tipo:str, peso:float)   – cuando hay micro detectada
            None                     – si no se detecta nada

        El peso se obtiene del dict MICRO_W y sirve directamente como
        contribución a la regresión logística.
        """
        try:
            logger.debug(
                "[Buffers] eye=%s mouth=%s brow=%s jaw=%s lips_curve=%s "
                "eyebrows_dist=%s brow_shape=%s eyebrow_l=%s eyebrow_r=%s "
                "mouth_corners=%s eyelid_tension=%s",
                [f"{v:.2f}" for v in list(self.eye_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.mouth_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.brow_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.jaw_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.lips_curve_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.eyebrows_dist_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.brow_shape_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.eyebrow_l_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.eyebrow_r_buffer)[-4:]],
                [f"{v:.2f}" for v in list(self.mouth_corners)[-4:]],
                [f"{v:.2f}" for v in list(self.eyelid_tension)[-4:]],
            )
        except Exception as e:
            print("[DEBUG][Buffers] Error al imprimir buffers:", e)

        # ─────────────────────────── pre-checks ──────────────────────────
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            return None

        if self.prev_gray.shape != frame_gray.shape:
            self.prev_gray = cv2.resize(
                self.prev_gray, (frame_gray.shape[1], frame_gray.shape[0])
            )

        if len(frame_gray.shape) > 2:
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
        if len(self.prev_gray.shape) > 2:
            self.prev_gray = cv2.cvtColor(self.prev_gray, cv2.COLOR_BGR2GRAY)

        # ───────────────────────── flujo óptico ──────────────────────────
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag               = np.linalg.norm(flow, axis=2)
        motion_intensity  = np.mean(mag)
        area_pct          = np.mean(mag > 2.0) * 100
        self.prev_gray    = frame_gray.copy()
        self._flag_micro_recent = False

        if not hasattr(self, "_votes_micro"):
            self._votes_micro = defaultdict(lambda: deque(maxlen=4))


        if motion_intensity > 5.0 or area_pct > 8.0:
            tipo = "indefinida"

            # ────── EMOCIONES / MICROS ORDENADAS POR PRIORIDAD ──────
            if is_sorpresa(self.eye_buffer, self.mouth_buffer, self.brow_buffer,
                            self.frame_cnt):
                    tipo = "Sorpresa"

            elif is_miedo(self.brow_buffer, self.eye_buffer, self.frame_cnt):
                    tipo = "Miedo"

            elif is_asco(self.brow_buffer, self.mouth_buffer, self.frame_cnt):
                    tipo = "Asco"

            elif is_labios_tensos(self.mouth_buffer, self.frame_cnt):
                    tipo = "Labios_tensos"

            elif (self.calibrated_micro and
                    is_microsonrisa_buffered(self.mouth_buffer, self.baseline_mouth)):
                    tipo = "Microsonrisa"

            elif is_ceño_fruncido(self.eyebrows_dist_buffer, self.brow_shape_buffer,
                                    self.frame_cnt):
                    tipo = "Ceño_fruncido"

            elif is_dientes_apretados(self.mouth_buffer, self.jaw_buffer,
                                        self.frame_cnt):
                    tipo = "Dientes_apretados"

            elif is_sonrisa_falsa(None, self.lips_curve_buffer, self.eye_buffer,
                                    self.frame_cnt):
                    tipo = "Sonrisa_falsa"

            elif is_boca_apretada(self.mouth_buffer, self.jaw_buffer,
                                    self.frame_cnt):
                    tipo = "Boca_apretada"

            elif is_levantar_una_ceja(self.eyebrow_l_buffer, self.eyebrow_r_buffer,
                                        self.frame_cnt):
                    tipo = "Una_ceja"

            elif is_tension_ojos(self.eye_buffer, self.frame_cnt):
                    tipo = "Tension_ojos"


            # filtro anti-repetición
            if self.optical_micro_events:
                last_fid, last_tipo = self.optical_micro_events[-1]
                if tipo == last_tipo and (self.frame_cnt - last_fid < 20):
                    return None

            # votación temporal (≥2 detecciones/4 frames)
            self._votes_micro[tipo].append(1)
            if sum(self._votes_micro[tipo]) < 2:
                return None

            # peso asociado a la micro-expresión
            peso = MICRO_W.get(tipo, MICRO_W["indefinida"])
            self._last_micro_weight = peso

            # registrar evento y notificar
            self.optical_micro_events.append((self.frame_cnt, tipo))
            self._flag_micro_recent = (tipo != "indefinida")
            logger.info("[MICRO] Microexpresión detectada: %s  (peso=%.2f)", tipo, peso)

            return tipo, peso

        # ────────────────────────── sin micro ───────────────────────────
        return None



    def _process_face(self, gray, rect, lm3d, shape):
        """
        Procesa un rostro: extrae rasgos faciales, actualiza buffers y
        devuelve un dict con los valores útiles para el lie-score.
        Logs detallados para depuración.
        """
        h, w = shape[:2]
        x1, y1, x2, y2 = [max(0, min(v, lim))
                        for v, lim in zip((rect.left(), rect.top(),
                                            rect.right(), rect.bottom()),
                                            (w-1, h-1, w-1, h-1))]
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"[_process_face] Recorte inválido: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
            return None
        face = gray[y1:y2, x1:x2]
        if face.size == 0:
            logger.warning("[_process_face] Face recortado vacío")
            return None

        logger.debug(f"[_process_face] Tamaño face: {face.shape}")

        # rPPG — frecuencia cardiaca
        self._update_rppg(self.last_frame[y1:y2, x1:x2])

        # luminancia media (normalizada con EMA)
        lum_now = cv2.mean(face)[0]
        old_lum_ema = self.lum_ema
        self.lum_ema = lum_now if self.lum_ema is None else \
                    self.LUM_EMA_ALPHA * lum_now + (1 - self.LUM_EMA_ALPHA) * self.lum_ema
        logger.debug(f"[_process_face] luminancia: {lum_now:.2f}, lum_ema anterior: {old_lum_ema}, nueva: {self.lum_ema:.2f}")

        # emoción CNN
        # ======== EMOCIÓN (fusión) ===========================================
        roi = cv2.resize(face, (64,64)).astype("float32")/255.
        roi = roi[np.newaxis, ..., np.newaxis]
        preds = self.model.predict(roi, verbose=0)[0]

        fer_res  = getattr(self, "fer_last_raw", [{}])
        deep_res = self._deepface_dict()

        emo, emo_score = self._fusion_emotion(preds, fer_res, deep_res)
        if emo_score < 0.25:            # confianza mínima
            logger.debug("[_process_face] Confianza emoción muy baja (%.2f)", emo_score)
            return None

        if np.isnan(preds).any():
            logger.warning("[_process_face] Predicción contiene NaN")
            return None
        conf = emo_score  # ← usamos la confianza fusionada
        logger.debug(f"[_process_face] Emoción FUSIONADA: {emo} conf={conf:.3f}")
        if conf < 0.25:   # mismo umbral que arriba
            logger.debug("[_process_face] Confianza demasiado baja, se ignora este frame")
            return None


        # 68 puntos dlib
        lm68 = self.predictor(gray, rect)
        pts = np.float32([[lm68.part(i).x, lm68.part(i).y] for i in range(68)])
        logger.debug(f"[_process_face] Primeros puntos landmarks 68: {pts[:5]} ...")

        # Calcula características microexpresión
        eye_open, brow_raise, mouth_open = feature_vector(pts)
        logger.debug(f"[_process_face] eye_open={eye_open:.3f}, brow_raise={brow_raise:.3f}, mouth_open={mouth_open:.3f}")
        self.calibrate_micro_baseline(eye_open, mouth_open, brow_raise)

        self.lips_curve_buffer.append(compute_lip_curvature(pts))

        try:
            # apertura ocular/bucal
            left_eye = np.linalg.norm(pts[37] - pts[41]) + np.linalg.norm(pts[38] - pts[40])
            right_eye = np.linalg.norm(pts[43] - pts[47]) + np.linalg.norm(pts[44] - pts[46])
            eye_val = (left_eye + right_eye) / 2
            self.eye_buffer.append(eye_val)

            mouth_open_val = np.linalg.norm(pts[62] - pts[66]) + np.linalg.norm(pts[63] - pts[65])
            self.mouth_buffer.append(mouth_open_val)

            brow_l = abs(pts[19][1] - pts[37][1])
            brow_r = abs(pts[24][1] - pts[44][1])
            brow_val = (brow_l + brow_r) / 2
            self.brow_buffer.append(brow_val)

            self.eyebrows_dist_buffer.append(np.linalg.norm(pts[21] - pts[22]))
            self.brow_shape_buffer.append((pts[21][1] - pts[17][1]) - (pts[22][1] - pts[26][1]))
            self.eyebrow_l_buffer.append(pts[19][1] - pts[37][1])
            self.eyebrow_r_buffer.append(pts[24][1] - pts[44][1])

            self.mouth_corners.append(pts[54][1] - pts[48][1])
            eyelid_t = abs(pts[38][1] - pts[37][1]) + abs(pts[44][1] - pts[43][1])
            self.eyelid_tension.append(eyelid_t)

            self.jaw_buffer.append(np.linalg.norm(pts[8] - pts[57]))

            logger.debug(f"[_process_face] Buffers actualizados: eye={eye_val:.3f}, mouth={mouth_open_val:.3f}, brow={brow_val:.3f}, eyelid_tension={eyelid_t:.3f}")
        except Exception as e:
            logger.error(f"[_process_face] Error llenando buffers: {e}")

        ear = np.mean([self._ear(pts[36:42]), self._ear(pts[42:48])])
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.debug(f"[_process_face] EAR calculado: {ear:.4f}, Laplacian var: {lap_var:.2f}")


        try:
            yaw, pitch, roll = self._solve_pose(lm3d, w, h)
            # ────────────────────────────────
            #  NORMALIZA YAW a [-90°, 90°]
            # ────────────────────────────────
            yaw_orig = yaw
            if yaw > 180:
                yaw -= 360
            if yaw < -180:
                yaw += 360
            if yaw > 90:
                yaw = 180 - yaw
            elif yaw < -90:
                yaw = -180 - yaw
            if yaw != yaw_orig:
                logger.debug(f"[Yaw] Normalizado {yaw_orig:.2f}° → {yaw:.2f}°")
            # ────────────────────────────────
            logger.debug(
                f"[_process_face] Pose calculada: yaw={yaw:.2f}, "
                f"pitch={pitch:.2f}, roll={roll:.2f}"
            )
        except Exception as e:
            logger.warning(f"[_process_face] Error calculando pose: {e}")
            yaw = pitch = roll = 0.0

        self.yaw_buf.append(yaw)
        self.pitch_buf.append(pitch)  # ← necesitarás self.pitch_buf = deque(maxlen=5) en __init__
        pitch_sm = np.mean([v for v in self.pitch_buf if v is not None]) if self.pitch_buf else 0.0

        self.roll_sm = 0.8 * self.roll_sm + 0.2 * roll
        yaw_sm = np.mean([v for v in self.yaw_buf if v is not None]) if self.yaw_buf else 0.0
        logger.debug(f"[_process_face] yaw suavizado: {yaw_sm:.2f}, roll suavizado: {self.roll_sm:.2f}")
        self._update_blinks(ear, lap_var, yaw_sm)

        if abs(yaw_sm) > self.YAW_HARD_LIMIT:
            logger.warning(f"[_process_face] Cara demasiado girada (yaw={yaw_sm:.2f}), se descarta frame")
            return None

        pupil = self._pupil_diam(lm3d, [474, 475, 476, 477], w, h) if lm3d else self.last_pupil
        self.last_pupil = pupil
        self.pupil_history.append(pupil)
        logger.debug(f"[_process_face] Tamaño pupila: {pupil:.4f}")

        self._detect_micro(self.frame_cnt, extra_buffers={
            "eyebrows_dist": self.eyebrows_dist_buffer,
            "brow_shape": self.brow_shape_buffer,
            "mouth_shape": self.mouth_shape_buffer,
            "lips_curve": self.lips_curve_buffer,
            "eyebrow_l": self.eyebrow_l_buffer,
            "eyebrow_r": self.eyebrow_r_buffer,
            "eyelid_tension": self.eye_buffer
        })

        # Flags sonrisa
        res_genuine = False
        res_false = False
        try:
            res_genuine = is_sonrisa_genuina(
                pts, self.lips_curve_buffer, self.eye_buffer,
                threshold_curve=self.thr_curve, threshold_eye=self.thr_eye,
                win=self.win_smile)
            res_false = is_sonrisa_falsa(
                pts, self.lips_curve_buffer, self.eye_buffer,
                threshold_curve=self.thr_curve, threshold_eye=self.thr_eye)
            if res_genuine:
                self.add_event("Sonrisa_genuina", self.frame_cnt)
            elif res_false:
                self.add_event("Sonrisa_falsa", self.frame_cnt)
        except Exception as e:
            logger.error(f"[_process_face] Error detectando sonrisa: {e}")

        bpm_now = self._blinks_per_minute()
        norm, flags = self._compute_norms_flags(
        ear, pupil / max(self.lum_ema, 1e-3),
        yaw_sm, pitch_sm, self.roll_sm,
        emo_score, emo,
        self._last_micro_weight,            # ← pasa el peso, no el flag
        bpm_now)

        feats = {
        "emo_fusion": flags["emo_neg"] * emo_score,                # ponderado
        "micro_opt" : flags["micro_opt"],
        "blink"     : max(flags["blink_fr"], flags["blink_rb"]),
        "ear"       : norm["ear"],
        "pupil"     : max(norm["pupil"], flags["pupil_pk"]),
        "hr"        : norm["hr"],
        "yaw"       : norm["yaw"],
        "au_tension": norm["au_tension"],
        "au_brows"  : norm["au_brows"],

        "pitch"     : norm["pitch"],                               # ← NUEVO
        "roll"      : norm["roll"],
        "gaze"      : norm["gaze"],
        "bpm"       : norm["bpm"]                                  # ← NUEVO
    }



        lie_score = self._logit(feats)

        if lap_var < 80 or lm3d is None:
            lie_score *= 0.9

        self.hist_genuine.append(int(res_genuine))
        self.hist_false.append(int(res_false))
        if sum(self.hist_genuine) >= 3:
            self.add_event("Sonrisa_genuina", self.frame_cnt)
        elif sum(self.hist_false) >= 3:
            self.add_event("Sonrisa_falsa", self.frame_cnt)

        if lap_var > self.STABLE_LAP_VAR:
            self._stable_frames += 1
        else:
            self._stable_frames = 0

        if self._stable_frames == self.STABLE_SECONDS * self.fps:
            if self.baseline["pupil"] is None:
                self.baseline["pupil"] = np.median(self.pupil_history) or pupil
                logger.info(f"[_process_face] Baseline pupil calibrado: {self.baseline['pupil']:.3f}")
            if self.hr and self.hr_baseline is None:
                self.hr_baseline = self.hr
                logger.info(f"[_process_face] Baseline HR calibrado: {self.hr_baseline:.1f} bpm")

        self._ear_hist.append(ear)
        if (not self._ear_calibrated) and len(self._ear_hist) >= 150:
            self._calibrate_ear_baseline()

        logger.debug(f"[_process_face] Resultado lie_score: {lie_score:.4f}")

        return {
            "emo": emo, "conf": conf,
            "ear": ear, "yaw": yaw_sm, "pupil": pupil,
            "lie": lie_score, "blinks": len(self.blink_times),
            "hr": self.hr or 0
        }

            # ──────────────────────────────────────────────────────────────
    #  Fusión de emociones CNN + FER + DeepFace
    # ──────────────────────────────────────────────────────────────
    def _fusion_emotion(
        self,
        cnn_preds,           # ← np.array shape=(7,) – salida de self.model.predict
        fer_res,             # ← lista de dicts de FER -↓
        deep_res,            # ← dict de DeepFace.analyze()
        weights=(0.50, 0.30, 0.20)   # (w_cnn, w_fer, w_deep)
    ):
        """
        Devuelve (label_español, confianza 0-1) fusionando:

        • cnn_preds: tu modelo Keras entrenado (7 clases).
        • fer_res  : salida crudísima de FER (lista con un dict que
                        contiene 'emotions': {'angry':0.12, …}).
        • deep_res : dict de DeepFace con distintas firmas según versión.

        Estrategia:
            – Se normalizan todas las probabilidades a 0-1.
            – Se hace un *weighted sum* con los pesos indicados.
            – Se escoge la emoción con la mayor puntuación y se devuelve
            esa puntuación dividida por la suma de pesos realmente usados,
            de forma que la confianza final ∈ [0, 1].
        """
        import numpy as np

        w_cnn, w_fer, w_dp = weights
        labels_es = self.emotion_labels  # ['Enojo','Asco', … 'Neutral']
        scores = {lab: 0.0 for lab in labels_es}

        # —— helper de mapeo EN ➜ ES ————————————————
        def eng2spa(lbl):
            m = {
                "angry": "Enojo",  "anger": "Enojo",
                "disgust": "Asco", "disgusted": "Asco", "contempt": "Asco",
                "fear": "Miedo",   "fearful": "Miedo",
                "happy": "Felicidad", "happiness": "Felicidad",
                "sad": "Tristeza", "sadness": "Tristeza",
                "surprise": "Sorpresa", "surprised": "Sorpresa",
                "neutral": "Neutral"
            }
            return m.get(lbl.lower())

        # —— CNN (tu modelo Keras) ————————————————
        probs_cnn = np.asarray(cnn_preds, dtype=np.float32)
        if probs_cnn.shape != (7,):
            probs_cnn = np.zeros(7, dtype=np.float32)

        for idx, lab in enumerate(labels_es):
            scores[lab] += w_cnn * float(probs_cnn[idx])

        # —— FER ————————————————————————————————
        if isinstance(fer_res, list) and fer_res and 'emotions' in fer_res[0]:
            for k, v in fer_res[0]['emotions'].items():
                lab = eng2spa(k)
                if lab in scores:
                    scores[lab] += w_fer * float(v)

        # —— DeepFace ————————————————————————————————
        if isinstance(deep_res, dict):
            # 1) diccionario completo de probabilidades
            emo_dict = deep_res.get('emotion') or deep_res.get('emotions')
            if isinstance(emo_dict, dict):
                for k, v in emo_dict.items():
                    lab = eng2spa(k)
                    if lab in scores:
                        scores[lab] += w_dp * float(v)

            # 2) clave 'dominant_emotion'
            dom = deep_res.get('dominant_emotion')
            if dom:
                lab = eng2spa(dom)
                if lab in scores:
                    scores[lab] += w_dp * 1.0  # confianza máxima a la dominante

        # —— Selección final ————————————————————
        label = max(scores, key=scores.get)
        used_w = (
            (w_cnn if probs_cnn.sum() else 0) +
            (w_fer if fer_res else 0) +
            (w_dp  if deep_res else 0)
        ) or 1.0  # evita división por 0

        conf = float(np.clip(scores[label] / used_w, 0.0, 1.0))
        return label, conf


    def _load_baseline(self):
        try:
            with open("baseline.json", "r") as f:
                data = json.load(f)
                self.baseline = data.get("baseline", self.baseline)
                self.hr_baseline = data.get("hr_baseline", None)
                self.bpm_baseline = data.get("bpm_baseline", None)
                logger.info("[Baseline] Cargado desde baseline.json")
        except:
            logger.warning("[Baseline] No se encontró baseline.json o error al cargar")

    def save_baseline(self):
        data = {
            "baseline": self.baseline,
            "hr_baseline": self.hr_baseline,
            "bpm_baseline": self.bpm_baseline
        }
        with open("baseline.json", "w") as f:
            json.dump(data, f, indent=2)
            logger.info("[Baseline] Guardado en baseline.json")

    def _update_blink_flags(self, ear: float) -> None:
            """
            Marca freeze y rebound del parpadeo.
            • freeze  = EAR cae ≥ 35 % bajo baseline
            • rebound = EAR supera +35 % del baseline en ≤ 6 frames
            """
            if self.ear_mu is None:          # aún sin baseline
                return

            thr_freeze  = self.ear_mu * 0.65        # −35 %
            thr_rebound = self.ear_mu * 1.35        # +35 %

            # —— freeze ——
            if not self.blink_freeze and ear < thr_freeze:
                self.blink_freeze = True
                self._freeze_fid  = self.frame_cnt
                self.add_event("Blink_freeze", self.frame_cnt)

            # —— rebound ——
            if self.blink_freeze and not self.blink_reb:
                if (self.frame_cnt - self._freeze_fid) <= 6 and ear > thr_rebound:
                    self.blink_reb = True
                    self.add_event("Blink_rebound", self.frame_cnt)

            # —— reset tras 10 frames ——
            if self.blink_freeze and (self.frame_cnt - self._freeze_fid) > 10:
                self.blink_freeze = self.blink_reb = False
                