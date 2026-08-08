import logging
from collections import deque
import numpy as np
logger = logging.getLogger(__name__)

_ceño_votes   = deque(maxlen=5)
_STOP_CEÑO_FOR = 20          # frames que debe “reposar” el gesto

_last_frown_fid = -9999      # inicial

def detect_peak(buffer, delta=3):
    if len(buffer) < 5:
        return False
    return buffer[-3] - buffer[-4] > delta and buffer[-3] - buffer[-2] > delta

# ------------- UMBRALES: AJUSTA SEGÚN TU DATA -----------------
UMBRAL_EYE_ABRE    = 48     # Ojo abierto
UMBRAL_MOUTH_ABRE  = 8      # Boca abierta
UMBRAL_BROW_UP     = 80     # Ceja alzada
UMBRAL_MOUTH_TENSO = 15     # Boca tensa
UMBRAL_JAW         = 105    # Mandíbula tensa
UMBRAL_COMISURAS   = -10    # Comisuras abajo
UMBRAL_CEJA_DIFF   = 2      # Dif. ceja izq-der para levantar una
win_labios = deque(maxlen=4)
win_ceño = deque(maxlen=4)
win_miedo = deque(maxlen=4)

# ----- Peso individual que aporta cada micro-expresión  (0-1) -----------
# ----- Peso individual que aporta cada micro-expresión  (0-1) ---------------
MICRO_W = {
    "Sorpresa"         : 0.18,
    "Miedo"            : 0.16,
    "Asco"             : 0.14,
    "Labios_tensos"    : 0.12,
    "Microsonrisa"     : 0.10,
    "Ceño_fruncido"    : 0.09,
    "Dientes_apretados": 0.09,
    "Boca_apretada"    : 0.08,
    "Una_ceja"         : 0.07,
    "Tension_ojos"     : 0.06,
    "Sonrisa_falsa"    : 0.09,   # ← NUEVA ENTRADA
    # fallback
    "indefinida"       : 0.04
}



def _log(name, ok, **vals):
    logger.debug(
        f"[Rules] {name:18} -> {'✔' if ok else '✘'} | " +
        " ".join(f"{k}={v:.2f}" for k, v in vals.items())
    )
    return ok

# -------- MICROSONRISA: REGLA AVANZADA CON BASELINE, PEAK, DELTA --------
micro_win_sonrisa = deque(maxlen=6)

def is_microsonrisa(mouth_buffer, baseline_mouth=8, delta_thr=4):
    if len(mouth_buffer) < 5:
        return False
    delta = mouth_buffer[-3] - baseline_mouth
    peak  = detect_peak(mouth_buffer, delta=3)
    ok    = delta > delta_thr and peak
    _log("Microsonrisa", ok, delta=delta, peak=int(peak))
    return ok

def is_microsonrisa_buffered(mouth_buffer, baseline_mouth=8, delta_thr=4):
    res = is_microsonrisa(mouth_buffer, baseline_mouth, delta_thr)
    micro_win_sonrisa.append(int(res))
    return sum(micro_win_sonrisa) >= 2

def is_sorpresa(
    eye, mouth, brow,
    eye_thr=None, mouth_thr=None, brow_thr=None,
    delta_eye=3, delta_mouth=2, delta_brow=3
):
    """
    Detección robusta de SORPRESA:
      • pico súbito y gran valor absoluto en OJOS + BOCA
      • cejas alzadas refuerzan (no obligatorias, pero ayudan)
      • voto positivo si se confirma ≥2 veces en 4 frames
    """
    if len(eye) < 5 or len(mouth) < 5 or len(brow) < 5:
        return False

    # ── Umbrales absolutos ───────────────────────────────────────
    thr_eye   = eye_thr   if eye_thr   is not None else UMBRAL_EYE_ABRE
    thr_mouth = mouth_thr if mouth_thr is not None else UMBRAL_MOUTH_ABRE
    thr_brow  = brow_thr  if brow_thr  is not None else UMBRAL_BROW_UP

    # ── Picos (cambios bruscos) ──────────────────────────────────
    eye_peak   = detect_peak(eye,   delta=delta_eye)
    mouth_peak = detect_peak(mouth, delta=delta_mouth)
    brow_peak  = detect_peak(brow,  delta=delta_brow)

    # ── Condiciones básicas ──────────────────────────────────────
    ojos_ok   = eye_peak   and eye[-3]   > thr_eye
    boca_ok   = mouth_peak and mouth[-3] > thr_mouth
    cejas_ok  = brow_peak  and brow[-3]  > thr_brow   # refuerzo, no obligatorio

    ok_frame  = ojos_ok and boca_ok and (cejas_ok or True)

    # ── Votación temporal (reduce falsos positivos) ──────────────
    _sorpresa_votes.append(int(ok_frame))
    ok = sum(_sorpresa_votes) >= 2            # al menos 2/4 frames

    logger.debug(
        "[DEBUG][is_sorpresa] eye=%.2f mouth=%.2f brow=%.2f "
        "eye_pk=%s mouth_pk=%s brow_pk=%s votos=%d/4 thr_e=%.1f thr_m=%.1f thr_b=%.1f => %s",
        eye[-3], mouth[-3], brow[-3],
        eye_peak, mouth_peak, brow_peak,
        sum(_sorpresa_votes), thr_eye, thr_mouth, thr_brow, ok
    )
    return _log("Sorpresa", ok,
                eye_m3=eye[-3], mouth_m3=mouth[-3], brow_m3=brow[-3])

def is_miedo(brow, eye, brow_thr=None, eye_thr=None):
    if len(brow) < 5 or len(eye) < 5:
        return False
    threshold_brow = brow_thr if brow_thr is not None else UMBRAL_BROW_UP
    threshold_eye = eye_thr if eye_thr is not None else UMBRAL_EYE_ABRE
    ok = (
        brow[-3] > threshold_brow and
        eye[-3] > threshold_eye
    )
    logger.debug(f"[DEBUG][is_miedo] brow[-3]={brow[-3]:.2f}, eye[-3]={eye[-3]:.2f}, threshold_brow={threshold_brow}, threshold_eye={threshold_eye} => ok={ok}")
    return _log("Miedo", ok, brow_m3=brow[-3], eye_m3=eye[-3])

def is_asco(brow, mouth, brow_thr=None, mouth_thr=None):
    if len(brow) < 4 or len(mouth) < 4:
        return False
    threshold_brow = brow_thr if brow_thr is not None else UMBRAL_BROW_UP
    threshold_mouth = mouth_thr if mouth_thr is not None else UMBRAL_MOUTH_TENSO
    ok = brow[-3] > threshold_brow and mouth[-3] < threshold_mouth
    logger.debug(f"[DEBUG][is_asco] brow[-3]={brow[-3]:.2f}, mouth[-3]={mouth[-3]:.2f}, threshold_brow={threshold_brow}, threshold_mouth_tenso={threshold_mouth} => ok={ok}")
    return _log("Asco", ok, brow_m3=brow[-3], mouth_m3=mouth[-3])

def is_felicidad(mouth, mouth_thr=None):
    if len(mouth) < 4:
        return False
    threshold_felicidad = (mouth_thr if mouth_thr is not None else UMBRAL_MOUTH_ABRE) * 2
    ok = mouth[-3] > threshold_felicidad
    logger.debug(f"[DEBUG][is_felicidad] mouth[-3]={mouth[-3]:.2f}, threshold_felicidad={threshold_felicidad:.2f} => ok={ok}")
    return _log("Felicidad", ok, mouth_m3=mouth[-3])

def is_tristeza(mouth, brow, mouth_thr=None, brow_thr=None):
    if len(mouth) < 4 or len(brow) < 4:
        return False
    threshold_mouth = mouth_thr if mouth_thr is not None else UMBRAL_MOUTH_ABRE
    threshold_brow = brow_thr if brow_thr is not None else UMBRAL_BROW_UP / 2
    ok = mouth[-3] < threshold_mouth and brow[-3] < threshold_brow
    logger.debug(f"[DEBUG][is_tristeza] mouth[-3]={mouth[-3]:.2f}, brow[-3]={brow[-3]:.2f}, threshold_mouth={threshold_mouth}, threshold_brow={threshold_brow} => ok={ok}")
    return _log("Tristeza", ok, mouth_m3=mouth[-3], brow_m3=brow[-3])

def is_enojo(brow, mouth, brow_thr=None, mouth_thr=None):
    if len(brow) < 4 or len(mouth) < 4:
        return False
    threshold_brow = brow_thr if brow_thr is not None else UMBRAL_BROW_UP / 2
    threshold_mouth = mouth_thr if mouth_thr is not None else UMBRAL_MOUTH_ABRE / 2
    ok = brow[-3] < threshold_brow and mouth[-3] < threshold_mouth
    logger.debug(f"[DEBUG][is_enojo] brow[-3]={brow[-3]:.2f}, mouth[-3]={mouth[-3]:.2f}, threshold_brow={threshold_brow}, threshold_mouth={threshold_mouth} => ok={ok}")
    return _log("Enojo", ok, brow_m3=brow[-3], mouth_m3=mouth[-3])

# Ventana y cool-down propios de la micro
_votes_labios_tensos = deque(maxlen=4)
_last_labios_fid     = [-999]      # lista → mutable dentro de la función

def is_labios_tensos(mouth_buf, fid,
                     z_thr=0.8, delta_thr=2.0,
                     cooldown=15, min_votes=2):
    """
    Versión compacta usando detect_X.
    mouth_buf : deque con la apertura/tensión labial
    fid       : frame id actual (int)
    """
    return detect_X(
        buf_A = mouth_buf,          # usamos la MISMA métrica dos veces
        buf_B = mouth_buf,
        fid   = fid,
        z_thr_A = z_thr,  sign_A = +1,      # queremos que suba
        z_thr_B = 0,      sign_B = +1,      # sin 2ª métrica real
        delta_thr_A = delta_thr, delta_thr_B = 0,
        votes_win = _votes_labios_tensos,
        min_votes = min_votes,
        cooldown  = cooldown,
        last_fid  = _last_labios_fid
    )


# ------------------------------------------------------------------
#  Boca apretada 2.0  –  menos falsos positivos
# ------------------------------------------------------------------
from collections import deque

# Ventana de votación (4 frames ≈ 130 ms a 30 fps)
_win_boca_apretada = deque(maxlen=4)

def is_boca_apretada(mouth_buf, jaw_buf,
                     mouth_thr=None, jaw_thr=None,
                     delta_mouth=-2.0,   # ↓ cierre brusco
                     delta_jaw=  3.0,   # ↑ tensión mandíbula
                     min_votes=2):
    """
    Detecta “boca apretada” combinando:
      • Umbral absoluto      (boca poco abierta + mandíbula alta)
      • Dinámica repentina   (cierre boca y subida mandíbula)
      • Votación temporal    (≥ min_votes en _win_boca_apretada)
    """
    if len(mouth_buf) < 5 or len(jaw_buf) < 5:
        return False

    thr_mouth = mouth_thr if mouth_thr is not None else UMBRAL_MOUTH_ABRE
    thr_jaw   = jaw_thr   if jaw_thr   is not None else UMBRAL_JAW

    # --- 1) nivel absoluto ------------------------------------------------
    level_ok = (mouth_buf[-3] < thr_mouth) and (jaw_buf[-3] > thr_jaw)

    # --- 2) dinámica repentina (picos) ------------------------------------
    mouth_close = (mouth_buf[-3] - mouth_buf[-4]) < delta_mouth
    jaw_up      = (jaw_buf[-3]   - jaw_buf[-4])   > delta_jaw

    cond_ok = level_ok and mouth_close and jaw_up

    # --- 3) votación temporal ---------------------------------------------
    _win_boca_apretada.append(int(cond_ok))
    ok = sum(_win_boca_apretada) >= min_votes

    logger.debug(
        "[DEBUG][is_boca_apretada] m[-3]=%.2f Δm=%.2f thr_m=%.2f | "
        "j[-3]=%.2f Δj=%.2f thr_j=%.2f | votes=%d/%d → ok=%s",
        mouth_buf[-3], mouth_buf[-3]-mouth_buf[-4], thr_mouth,
        jaw_buf[-3],   jaw_buf[-3]-jaw_buf[-4],   thr_jaw,
        sum(_win_boca_apretada), _win_boca_apretada.maxlen, ok
    )
    return _log("Boca_apretada", ok,
                mouth_m3=mouth_buf[-3], jaw_m3=jaw_buf[-3])


def is_comisuras_abajo(mouth_corners, corners_thr=None):
    if len(mouth_corners) < 4:
        return False
    threshold = corners_thr if corners_thr is not None else UMBRAL_COMISURAS
    ok = mouth_corners[-3] < threshold
    logger.debug(f"[DEBUG][is_comisuras_abajo] corners[-3]={mouth_corners[-3]:.2f}, threshold_comisuras={threshold} => ok={ok}")
    return _log("Comisuras_abajo", ok, corners_m3=mouth_corners[-3])

def is_dientes_apretados(mouth, jaw, mouth_thr=None, jaw_thr=None, delta_thr=3):
    """
    Detecta dientes apretados evitando falsos positivos:
    - Boca poco abierta
    - Mandíbula baja de forma repentina (no sostenida)
    - Cambio brusco en los últimos frames
    """
    if len(mouth) < 5 or len(jaw) < 5:
        return False

    threshold_mouth = mouth_thr if mouth_thr is not None else UMBRAL_MOUTH_ABRE
    threshold_jaw = jaw_thr if jaw_thr is not None else UMBRAL_JAW

    # Valor actual
    mouth_now = mouth[-3]
    jaw_now = jaw[-3]

    # Cambios recientes (dinámica)
    mouth_delta = mouth[-3] - mouth[-5]
    jaw_delta = jaw[-3] - jaw[-5]

    # La boca debe estar cerrada, la mandíbula baja de golpe
    ok = (
        mouth_now < threshold_mouth and
        jaw_now > threshold_jaw and
        mouth_delta < 1.0 and    # boca no abre más, ¡cierra!
        jaw_delta > delta_thr    # mandíbula baja rápido
    )

    logger.debug(
        f"[DEBUG][is_dientes_apretados] mouth[-3]={mouth_now:.2f}, jaw[-3]={jaw_now:.2f}, "
        f"mouth_delta={mouth_delta:.2f}, jaw_delta={jaw_delta:.2f}, "
        f"threshold_jaw={threshold_jaw}, threshold_mouth={threshold_mouth} => ok={ok}"
    )
    return _log("Dientes_apretados", ok, mouth_m3=mouth_now, jaw_m3=jaw_now, mouth_delta=mouth_delta, jaw_delta=jaw_delta)

_sorpresa_votes = deque(maxlen=4) 

def is_sonrisa_falsa(pts, lips_curve_buffer, eye_buffer, threshold_curve=6, threshold_eye=48, asym_thr=4):
    if len(lips_curve_buffer) < 4 or len(eye_buffer) < 4:
        return False
    curva = lips_curve_buffer[-3]
    ojo   = eye_buffer[-3]
    ok    = (curva > threshold_curve) and (ojo < threshold_eye)
    if pts is not None and len(pts) > 55 and is_asimetrica(pts, diff_thr=asym_thr):
        ok = ok and True
    return _log("Sonrisa_falsa", ok, curva=curva, ojo=ojo)

def _calibrate_smile_thresholds(self):
    if len(self.lips_curve_buffer) == 0 or len(self.eye_buffer) == 0:
        return
    if self.frame_cnt <= self.init_frames:
        self.curve_baseline.append(self.lips_curve_buffer[-1])
        self.eye_baseline.append(self.eye_buffer[-1])
    elif self.frame_cnt == self.init_frames + 1:
        self.thr_curve = auto_thr(self.curve_baseline, k=1.8)
        self.thr_eye   = auto_thr(self.eye_baseline,   k=1.5)
        self.win_smile = max(1, int(self.fps * 0.10))
        self.hist_genuine.clear(); self.hist_false.clear()
        logger.info(f"[Calibración] thr_curve={self.thr_curve:.2f}  "
                    f"thr_eye={self.thr_eye:.2f}  win_smile={self.win_smile}")
        self.hist_false.clear()
        logger.info(f"[Calibración] thr_curve={self.thr_curve:.2f}  "
                    f"thr_eye={self.thr_eye:.2f}  "
                    f"win_smile={self.win_smile}")

def auto_thr(buf, k: float = 2.0) -> float:
    mu = np.mean(buf);  sigma = np.std(buf)
    return mu + k * sigma

def avg(buf, n=3):
    arr = list(buf)[-n:]
    return sum(arr)/len(arr) if arr else 0.0

def is_asimetrica(pts, diff_thr=4):
    left = pts[48][1]
    right = pts[54][1]
    center = pts[51][1]
    return abs((left + right)/2 - center) > diff_thr

def au6_intensity(pts):
    try:
        return np.linalg.norm(pts[158] - pts[160])
    except IndexError:
        return 0.0

def is_sonrisa_genuina(pts, lips_buf, eye_buf, threshold_curve, threshold_eye, win, speed_thr=1.5, au6_thr=1.5):
    if len(lips_buf) < win+2 or len(eye_buf) < win+2:
        return False
    curva   = avg(lips_buf, win)
    curva_d = lips_buf[-2] - lips_buf[-3]
    ojo     = avg(eye_buf,  win)
    ok = (curva > threshold_curve and
          ojo   > threshold_eye   and
          abs(curva_d) < speed_thr)
    if pts is not None and len(pts) > 160:
        try:
            au6 = np.linalg.norm(pts[158] - pts[160])
            ok = ok and (au6 > au6_thr)
        except IndexError:
            pass
    return _log("Sonrisa_genuina", ok, curva=curva, ojo=ojo, speed=curva_d)

def is_sonrisa_genuina_both_eyes(pts, lips_buf, left_eye_buf, right_eye_buf, threshold_curve, threshold_eye, au6_thr=1.5):
    if (len(lips_buf) < 4 or len(left_eye_buf) < 4 or len(right_eye_buf) < 4):
        return False
    curva  = lips_buf[-3]
    ojo_l  = left_eye_buf[-3]
    ojo_r  = right_eye_buf[-3]
    ok = (curva > threshold_curve and ojo_l > threshold_eye and ojo_r > threshold_eye)
    if pts is not None and len(pts) > 160:
        try:
            ok = ok and (np.linalg.norm(pts[158]-pts[160]) > au6_thr)
        except IndexError:
            pass
    return _log("Sonrisa_genuina", ok, curva=curva, ojo_l=ojo_l, ojo_r=ojo_r)

# ------------------------------------------------------------------
#  Ceño fruncido 2.0  –  menos falsos positivos
# ------------------------------------------------------------------
from collections import deque

# Ventana temporal de votación  (4 frames ≈ 130 ms a 30 fps)
_win_ceno_fruncido = deque(maxlen=4)

def is_ceño_fruncido(buf_dist, buf_shape, fid,   # ← frame id actual
                     z_thr=0.7, delta_thr=-0.9,
                     min_votes=3):
    """
    buf_dist   : distancia entre cejas (px)
    buf_shape  : curvatura negativa (px)
    fid        : nº de frame (int)

    Devuelve True una sola vez cuando detecta el gesto;  
    luego espera _STOP_CEÑO_FOR frames antes de volver a disparar.
    """
    global _last_frown_fid

    if len(buf_dist) < 6:        # necesitamos historia
        return False

    # ── 0) anti-rebote ───────────────────────────────────────────
    if fid - _last_frown_fid < _STOP_CEÑO_FOR:
        return False

    # ── 1) z-scores respecto a la mediana de los 50 últimos frames ─
    def z(x, buf):
        ref = np.array(buf)[-50:]
        mu, sd = np.median(ref), np.std(ref) or 1e-3
        return (x - mu) / sd

    z_dist  = z(buf_dist[-3],  buf_dist)
    z_shape = z(buf_shape[-3], buf_shape)

    # ── 2) dinámica rápida (derivada) ────────────────────────────
    dz_dist  = buf_dist[-3]  - buf_dist[-4]
    dz_shape = buf_shape[-3] - buf_shape[-4]

    cond_ok = (
        z_dist  < -z_thr and
        z_shape < -z_thr and
        dz_dist  <  delta_thr and
        dz_shape <  delta_thr
    )

    # ── 3) votación temporal ────────────────────────────────────
    _ceño_votes.append(int(cond_ok))
    ok = sum(_ceño_votes) >= min_votes

    if ok:
        _last_frown_fid = fid     # marca la “culminación” del gesto

    logger.debug(
        "[Frown] z_dist=%.2f z_shape=%.2f dz_d=%.2f dz_s=%.2f votes=%d/%d → %s",
        z_dist, z_shape, dz_dist, dz_shape,
        sum(_ceño_votes), _ceño_votes.maxlen, ok
    )
    return _log("Ceño_fruncido", ok,
                z_dist=z_dist, z_shape=z_shape,
                dz_dist=dz_dist, dz_shape=dz_shape)
    
def detect_X(buf_A, buf_B, fid,
             z_thr_A, sign_A,
             z_thr_B, sign_B,
             delta_thr_A, delta_thr_B,
             votes_win, min_votes,
             cooldown, last_fid):
    """
    • buf_A / buf_B  : buffers (deque) de la métrica que quieras
    • fid            : nº de frame actual (int)
    • z_thr_*        : umbral z-score (ej. +0.8 ó –0.8)
    • sign_*         : +1 si buscas subida, –1 si buscas bajada
    • delta_thr_*    : cambio brusco mínimo (mismo signo_A/B)
    • votes_win      : deque(maxlen=4)   ventana de votación
    • last_fid       : lista con un único int  → anti-rebote
    Devuelve True sólo cuando (1) nivel, (2) pico y (3) votos cumplen.
    """
    # —— cool-down ——————————————————————————
    if fid - last_fid[0] < cooldown or len(buf_A) < 6:
        return False

    # —— z-scores (normaliza al rostro) —————————
    def z(x, buf):
        ref = np.array(buf)[-50:]
        mu, sd = np.median(ref), np.std(ref) or 1e-3
        return (x - mu) / sd

    zA, zB = z(buf_A[-3], buf_A), z(buf_B[-3], buf_B)
    dA, dB = buf_A[-3] - buf_A[-4], buf_B[-3] - buf_B[-4]

    level_ok =  (sign_A * zA) >  z_thr_A  and (sign_B * zB) >  z_thr_B
    spike_ok =  (sign_A * dA) >  delta_thr_A and (sign_B * dB) >  delta_thr_B
    cond_ok  = level_ok and spike_ok

    votes_win.append(int(cond_ok))
    ok = sum(votes_win) >= min_votes
    if ok:
        last_fid[0] = fid        # marca la detección para el cool-down
    return ok

# ---------------------------------------------------------------
# Levantar_una_ceja 2.0
# ---------------------------------------------------------------
from collections import deque
_win_ceja = deque(maxlen=4)          # ventana  (~130 ms a 30 fps)

def is_levantar_una_ceja(eyebrow_l_buf, eyebrow_r_buf,
                         ceja_diff_thr=None,
                         delta_thr=-1.2,          # ↑ alzar = y desciende (-px)
                         min_votes=2):
    """
    Detecta la acción de levantar una sola ceja (izq o der):

      • diff_abs  ≥ UMBRAL_CEJA_DIFF
      • Δ (alzada) < delta_thr   (subida rápida)
      • la otra ceja cambia poco (- evita movimientos globales)
      • al menos `min_votes` en `_win_ceja`
    """

    # buffers suficientemente largos
    if len(eyebrow_l_buf) < 5 or len(eyebrow_r_buf) < 5:
        return False

    thr_diff = ceja_diff_thr if ceja_diff_thr is not None else UMBRAL_CEJA_DIFF

    # valores actuales y previos
    l_now, l_prev = eyebrow_l_buf[-3], eyebrow_l_buf[-4]
    r_now, r_prev = eyebrow_r_buf[-3], eyebrow_r_buf[-4]

    diff_abs = l_now - r_now          # >0  = sube izquierda;  <0  = sube derecha

    # --- ¿qué ceja sube? ----------------------------------------------------
    if diff_abs > 0:                  # sube IZQUIERDA
        side_up_fast  = (l_now - l_prev) < delta_thr
        other_stable  = abs(r_now - r_prev) < abs(delta_thr)/2
    else:                             # sube DERECHA
        side_up_fast  = (r_now - r_prev) < delta_thr
        other_stable  = abs(l_now - l_prev) < abs(delta_thr)/2

    cond_ok = (abs(diff_abs) > thr_diff) and side_up_fast and other_stable

    # votación temporal
    _win_ceja.append(int(cond_ok))
    ok = sum(_win_ceja) >= min_votes

    logger.debug(
        "[DEBUG][is_levantar_una_ceja] L=%.2f ΔL=%.2f | R=%.2f ΔR=%.2f | "
        "diff=%.2f thr=%.2f votes=%d/%d → ok=%s",
        l_now, l_now-l_prev, r_now, r_now-r_prev,
        diff_abs, thr_diff, sum(_win_ceja), _win_ceja.maxlen, ok
    )
    return _log("Una_ceja", ok,
                eyebrow_l=l_now, eyebrow_r=r_now, diff=diff_abs)

def is_tension_ojos(eyelid_tension, eye_thr=None):
    if len(eyelid_tension) < 5:
        return False
    threshold = eye_thr if eye_thr is not None else UMBRAL_EYE_ABRE
    ok = eyelid_tension[-3] > threshold and eyelid_tension[-4] < threshold
    logger.debug(f"[DEBUG][is_tension_ojos] eyelid_tension[-3]={eyelid_tension[-3]:.2f}, eyelid_tension[-4]={eyelid_tension[-4]:.2f}, threshold_eye_abre={threshold} => ok={ok}")
    return _log("Tension_ojos", ok, eyelid_tension=eyelid_tension[-3])

def is_mirada_evasiva(gaze):
    if len(gaze) < 4:
        return False
    ok = abs(gaze[-3]) > 2 and abs(gaze[-4]) < 2
    logger.debug(f"[DEBUG][is_mirada_evasiva] gaze[-3]={gaze[-3]:.2f}, gaze[-4]={gaze[-4]:.2f} => ok={ok}")
    return _log("Mirada_evasiva", ok, gaze_m3=gaze[-3], gaze_m4=gaze[-4])

def is_parpadeo_excesivo(blinks):
    if len(blinks) < 15:
        return False
    ok = sum(blinks[-15:]) > 2
    logger.debug(f"[DEBUG][is_parpadeo_excesivo] sum[-15:]={sum(blinks[-15:])} => ok={ok}")
    return _log("Parpadeo_excesivo", ok, total=sum(blinks[-15:]))

def is_parpadeo_reducido(blinks):
    if len(blinks) < 75:
        return False
    ok = sum(blinks[-75:]) == 0
    logger.debug(f"[DEBUG][is_parpadeo_reducido] sum[-75:]={sum(blinks[-75:])} => ok={ok}")
    return _log("Parpadeo_bajo", ok, total=sum(blinks[-75:]))

def is_pupilas_dilatadas(pupil_size):
    if len(pupil_size) < 4:
        return False
    ok = pupil_size[-3] > 2
    logger.debug(f"[DEBUG][is_pupilas_dilatadas] pupil_size[-3]={pupil_size[-3]:.2f} => ok={ok}")
    return _log("Pupilas_dilatadas", ok, pupil=pupil_size[-3])

def compute_lip_curvature(pts):
    left_corner_y  = pts[48][1]
    right_corner_y = pts[54][1]
    center_y       = pts[51][1]
    curvature = -((left_corner_y + right_corner_y) / 2 - center_y)
    return curvature

def print_buffers_maxmin(buffers_dict):
    logger.debug("\n[DEBUG][Buffers Max/Min]")
    for key, buf in buffers_dict.items():
        if buf:
            vals = [float(x) for x in buf if isinstance(x, (float, int, str)) and str(x).replace('.','',1).replace('-','',1).isdigit()]
            if vals:
                logger.debug(f"{key:15} min={min(vals):.2f} max={max(vals):.2f}")
            else:
                logger.debug(f"{key:15} vacío o no numérico")
        else:
            logger.debug(f"{key:15} vacío")

def is_fruncir_los_labios(mouth_shape, lips_curve, mouth_shape_thr=2, lips_curve_thr=2):
    if len(mouth_shape) < 4 or len(lips_curve) < 4:
        return False
    ok = mouth_shape[-3] < mouth_shape_thr and abs(lips_curve[-3]) > lips_curve_thr
    logger.debug(f"[DEBUG][is_fruncir_los_labios] mouth_shape[-3]={mouth_shape[-3]:.2f}, lips_curve[-3]={lips_curve[-3]:.2f} => ok={ok}")
    return ok
