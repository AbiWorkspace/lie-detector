import mediapipe as mp
import cv2, numpy as np
from typing import Tuple
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

import cv2
import numpy as np
from typing import Tuple, Optional

# ─── Paleta fosforescente ──────────────────────────────────────────────
FLUO_GREEN   = (0, 255, 0)          # BGR
FLUO_DGREEN  = (0, 180, 0)
FLUO_CYAN    = (0, 255, 255)
FLUO_PINK    = (255, 102, 255)
FLUO_PURPLE  = (204, 153, 255)

# ─── Parámetros globales ───────────────────────────────────────────────
HUD_W_DEF  = 200
GLOW_BLUR  = 15
GLOW_GAIN  = 0.6
MAX_BAR_W  = 60                    # ← Nuevo: ancho máx. de cada barra


mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh

_BAR_STATS = {
    "eye":   {"min": 1e9, "max": -1e9},
    "brow":  {"min": 1e9, "max": -1e9},
    "mouth": {"min": 1e9, "max": -1e9},
}

_BAR_COLORS = {
    "eye":   (0, 255, 255),      # cyan
    "brow":  (255, 102, 255),    # rosa
    "mouth": (0, 255,   0),      # verde
}

__all__ = [
    "draw_bars",
    "draw_liescore",
    "draw_mesh",
    "draw_hud",
    "draw_technologies",
    "draw_landmarks3d",
    "draw_emotions",           # ← NUEVO
    "_shadow_text",          #  ←  AÑADIDO
]

# ═══════════════════ NUEVO – Overlay de Emociones DeepFace + FER ═══════
def draw_emotions(frame: np.ndarray,
                  deep_emo: Optional[str] = None,
                  fer_emo: Optional[str] = None,
                  *,
                  anchor: str = "tr",
                  offset: Tuple[int, int] = (12, 12)):
    """
    Muestra las emociones dominantes de DeepFace y FER
    en una pequeña tarjeta sobre el vídeo.

    • deep_emo y fer_emo son cadenas.  Se muestran sólo si no son None.
    • anchor (‘tl’,‘tr’,‘bl’,‘br’) define la esquina.
    """
    if deep_emo is None and fer_emo is None:
        return

    h, w         = frame.shape[:2]
    ox, oy       = offset
    pad          = 8
    lines        = []
    if deep_emo is not None:
        lines.append(f"DeepFace : {deep_emo}")
    if fer_emo is not None:
        lines.append(f"FER      : {fer_emo}")

    txt_h        = 20 * len(lines) + pad * 2
    txt_w        = 170                      # fijo: suficiente para las dos líneas

    # ---- coordenadas según ancla --------------------------------------
    if   anchor == "tl":
        x1, y1 = ox, oy
    elif anchor == "tr":
        x1, y1 = w - txt_w - ox, oy
    elif anchor == "bl":
        x1, y1 = ox, h - txt_h - oy
    elif anchor == "br":
        x1, y1 = w - txt_w - ox, h - txt_h - oy
    else:
        raise ValueError(f"anchor desconocido: {anchor}")

    x2, y2 = x1 + txt_w, y1 + txt_h

    _blend_rect(frame, (x1, y1), (x2, y2), (30, 30, 30), 0.45)
    for i, linea in enumerate(lines):
        _shadow_text(frame, linea, (x1 + pad, y1 + pad + 18*i),
             cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))  # rojo fuerte



def draw_3d_mask(frame: np.ndarray, lm, color: Tuple[int, int, int] = FLUO_GREEN, alpha: float = 0.19, blur: int = 31):
    """
    Dibuja una 'máscara 3D' sobre el rostro usando los landmarks de MediaPipe (468 puntos).
    Rellena el área facial principal con un color fosforescente, semitransparente y opcionalmente difuminado.
    """
    if lm is None or not hasattr(lm, "landmark") or len(lm.landmark) < 468:
        return

    h, w = frame.shape[:2]
    # Landmarks faciales principales que definen la silueta
    indices_contour = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]

    # Extrae los puntos de la silueta
    pts = []
    for idx in indices_contour:
        pt = lm.landmark[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)

    # Crea la máscara en negro
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color)

    # Aplica desenfoque para un efecto glow suave
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    # Mezcla la máscara en el frame original
    cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0, dst=frame)

    # Dibuja el contorno fosforescente encima
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_landmarks3d(frame: np.ndarray,
                     pts3d: np.ndarray,
                     pitch: float = None,
                     yaw:   float = None,
                     roll:  float = None,
                     *,
                     draw_axes: bool = True):
    """
    Wire-frame 3-D + ejes de pose + leyenda P/Y/R.
    `pts3d` = array (68, 3) devuelto por face_alignment.
    """
    if pts3d is None or pts3d.shape[0] < 68:
        return

    h, w   = frame.shape[:2]
    thick  = max(int(round(w / 640)), 1)
    radius = max(int(round(w / 900)), 1)

    # ----------  Conectividad «clásica» 68 pts ----------------------
    EDGES = [
        list(range(0, 17)),                      # Mandíbula
        list(range(17, 22)), list(range(22, 27)),# Cejas
        list(range(27, 31)), list(range(31, 36)),# Nariz
        [36,37,38,39,40,41,36],                  # Ojo izq.
        [42,43,44,45,46,47,42],                  # Ojo der.
        list(range(48, 60))+[48],                # Labios ext.
        list(range(60, 68))+[60]                 # Labios int.
    ]

    # ----------  Proyección ortográfica a 2-D -----------------------
    pts2d = pts3d[:, :2].astype(int)

    # ----------  Dibuja líneas -------------------------------------
    for edge in EDGES:
        cv2.polylines(frame, [pts2d[edge]], False,
                      (255, 255, 255), thick, cv2.LINE_AA)

    # ----------  Puntos opcionales ---------------------------------
    for (x, y) in pts2d:
        cv2.circle(frame, (x, y), radius,
                   (255, 255, 255), -1, cv2.LINE_AA)
        # ----------  Leyenda con la orientación -------------------------
    if pitch is not None and yaw is not None and roll is not None:
        txt = f"3D pose:  pitch={pitch:+.1f}°  yaw={yaw:+.1f}°  roll={roll:+.1f}°"
        _shadow_text(
            frame, txt,
            (10, 24),                      # ↖ esquina sup-izq
            cv2.FONT_HERSHEY_DUPLEX, 0.68, # tamaño homogéneo
            (0, 255, 255)                  # amarillo fosforescente
        )

    # ----------  Ejes de pose + texto ------------------------------
    if draw_axes and None not in (pitch, yaw, roll):
        nose      = pts2d[30]                         # puente nariz
        axis_len  = int(max(w, h) * 0.10)

        # Matriz de rotación (Rz·Ry·Rx)
        p, y_, r  = np.deg2rad([pitch, yaw, roll])
        Rx = np.array([[1, 0, 0],
                       [0,  np.cos(p), -np.sin(p)],
                       [0,  np.sin(p),  np.cos(p)]])
        Ry = np.array([[ np.cos(y_), 0, np.sin(y_)],
                       [0, 1, 0],
                       [-np.sin(y_), 0, np.cos(y_)]])
        Rz = np.array([[np.cos(r), -np.sin(r), 0],
                       [np.sin(r),  np.cos(r), 0],
                       [0, 0, 1]])
        R  = Rz @ Ry @ Rx

        axes   = np.float32([[axis_len,0,0],
                             [0,axis_len,0],
                             [0,0,axis_len]]) @ R.T
        axes2d = axes[:, :2] + nose

        # X-rojo, Y-verde, Z-azul
        for (x, y), c in zip(axes2d.astype(int),
                             [(0,0,255), (0,255,0), (255,0,0)]):
            cv2.arrowedLine(frame, tuple(nose), (int(x), int(y)),
                            c, thick+1, cv2.LINE_AA, tipLength=0.08)

        # ── Leyenda con la orientación ────────────────────────────
        txt = f"P:{pitch:+.0f}°  Y:{yaw:+.0f}°  R:{roll:+.0f}°"
        _shadow_text(frame, txt,
                     (nose[0] + 10, nose[1] - 10),     # desplázala a tu gusto
                     cv2.FONT_HERSHEY_DUPLEX, 0.55,
                     (0, 255, 255))                    # cian fosforescente


def _blend_rect(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
                color: Tuple[int, int, int] = FLUO_GREEN, alpha: float = 0.6,
                glow: bool = False):
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    if glow:
        _glow_rect(img, p1, p2, color)


def _glow_rect(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
               color: Tuple[int, int, int]):
    glow = np.zeros_like(img, dtype=np.uint8)
    cv2.rectangle(glow, p1, p2, color, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), GLOW_BLUR)
    cv2.addWeighted(glow, GLOW_GAIN, img, 1.0, 0, dst=img)


def _vertical_gradient(h: int, color_top: Tuple[int, int, int],
                       color_bot: Tuple[int, int, int]) -> np.ndarray:
    top = np.array(color_top, float)
    bot = np.array(color_bot, float)
    t   = np.linspace(0.0, 1.0, h)[:, None]
    grad = (top + t * (bot - top)).astype(np.uint8)
    return grad[:, None, :]                           # (h,1,3)

# ═══════════════════ Barras Ojo / Ceja / Boca ═════════════════════════
def draw_bars(frame: np.ndarray, eye: float, brow: float, mouth: float, *,
              alpha: float = 0.85):
    """
    Tres barras verticales (OJO, CEJA, BOCA) normalizadas
    y colocadas en la esquina inferior-derecha.
    """
    h, w = frame.shape[:2]
    barras, esp = 3, 14                       # nº barras, espaciado
    bar_w = min(max(int(w*0.04), 16), MAX_BAR_W)
    bar_h = int(h * 0.18)
    base_y = h - 16                           # margen inferior

    # ---- normalización dinámica --------------------------------------
    vals_norm = {
        "eye":   _norm_and_update("eye",   eye),
        "brow":  _norm_and_update("brow",  brow),
        "mouth": _norm_and_update("mouth", mouth),
    }

    orden = ["eye", "brow", "mouth"]
    for i, key in enumerate(orden):
        val = vals_norm[key]
        fill_h = max(int(bar_h * val), 1)

        x2 = w - esp - i * (bar_w + esp)      # empezando desde la derecha
        x1 = x2 - bar_w
        y2 = base_y
        y1 = y2 - fill_h

        # gradiente vertical en el color propio
        col = _BAR_COLORS[key]
        grad = _vertical_gradient(fill_h, tuple(int(c*0.7) for c in col), col)
        grad = np.tile(grad, (1, bar_w, 1))

        roi = frame[y1:y2, x1:x2]
        if roi.size:
            cv2.addWeighted(grad, alpha, roi, 1-alpha, 0, dst=roi)

        # contorno + etiquetas
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        _shadow_text(frame, key.upper(), (x1+2, y2+14),
                     cv2.FONT_HERSHEY_PLAIN, 0.9, col)
        _shadow_text(frame, f"{val*100:3.0f}%", (x1+2, y1-4),
                     cv2.FONT_HERSHEY_PLAIN, 0.8, col)

        # glow sutil si pasa 85 %
        if val > 0.85:
            _glow_rect(frame, (x1, y1), (x2, y2), col)

# ─────────────────────────────────────────────────────────────
#  Barra de lie-score con soportes de anclaje (TL, TR, BL, BR)
# ─────────────────────────────────────────────────────────────
def draw_liescore(
    frame: np.ndarray,
    score: float,
    *,
    anchor: str = "tl",            #  'tl' | 'tr' | 'bl' | 'br'
    offset: Tuple[int, int] = (12, 12),
    w_frac: float = 0.35,
    h_px: int = 20,
    color_fg: Tuple[int, int, int] = FLUO_GREEN,
    color_bg: Tuple[int, int, int] = (30, 30, 30),
    alpha_bg: float = 0.35
):
    """
    Dibuja la barra de lie-score anclada a la esquina indicada.
    • `score` debe estar ∈ [0,1].  Se recorta con np.clip.
    • `anchor` :
        - 'tl'  → top-left      (por defecto, como antes)
        - 'tr'  → top-right
        - 'bl'  → bottom-left   ← la que quieres
        - 'br'  → bottom-right
    • `offset` desplaza la barra desde esa esquina.
    """
    h,  w   = frame.shape[:2]
    ox, oy  = offset
    bar_w   = int(w * w_frac)
    pct     = float(np.clip(score, 0, 1))

    # ── coordenadas según ancla ────────────────────────────
    if   anchor == "tl":
        x, y = ox, oy
    elif anchor == "tr":
        x, y = w - bar_w - ox,            oy
    elif anchor == "bl":
        x, y = ox,             h - h_px - oy
    elif anchor == "br":
        x, y = w - bar_w - ox,  h - h_px - oy
    else:
        raise ValueError(f"anchor desconocido: {anchor}")

    p1, p2 = (x, y), (x + bar_w, y + h_px)

    # ── capa de fondo + barra foreground ───────────────────
    _blend_rect(frame, p1, p2, color_bg, alpha_bg)
    _blend_rect(frame, p1, (x + int(bar_w * pct), y + h_px),
                color_fg, 0.9, glow=True)

    cv2.rectangle(frame, p1, p2, color_fg, 1, cv2.LINE_AA)
    _shadow_text(frame,
                 f"{score*100:4.0f}%",
                 (p2[0] + 12, p2[1] - 2),
                 cv2.FONT_HERSHEY_DUPLEX, 0.6,
                 color=color_fg)


# ═══════════════════ Malla Facial / Iris ══════════════════════════════
def draw_mesh(frame: np.ndarray, lm, *, highlight=False):
    """
    Dibuja la malla facial MediaPipe:
    - Tesselación
    - Contornos (ojos, labios, cejas, mandíbula)
    - Iris
    - Puntos individuales opcionales (highlight)
    """
    if lm is None:
        return

    # 🔷 Malla triangular completa
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=lm,
        connections=mp_face.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
    )

    # 🔶 Contornos: ojos, labios, nariz, cara
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=lm,
        connections=mp_face.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
    )

    # 🔵 Iris (pupilas)
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=lm,
        connections=mp_face.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
    )

    # ✅ Si se desea, destacar los puntos 468–477 manualmente
    if highlight:
        h, w = frame.shape[:2]
        for i in range(468, 478):
            pt = lm.landmark[i]
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (cx, cy), 2, FLUO_PINK, -1, cv2.LINE_AA)

        # Dibujar ID para debug (opcional)
        for idx in [33, 133, 61, 291, 199, 1]:
            pt = lm.landmark[idx]
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, FLUO_PURPLE, 1, cv2.LINE_AA)

# ═══════════════════ HUD (Datos en vivo) ══════════════════════════════
def draw_hud(frame: np.ndarray, *, fps: float, blinks: int, hr: float,
             text: Optional[str] = None):
    h, w = frame.shape[:2]
    pad    = 10
    hud_w  = min(HUD_W_DEF, w - 24)
    x2     = w - 12
    x1     = x2 - hud_w
    y1, y2 = 12, 110
    _blend_rect(frame, (x1, y1), (x2, y2), (30, 30, 30), 0.45)

    filas = [
        f"FPS     : {fps:4.1f}",
        f"Parpadeos: {blinks:3d}",
        f"FC (lpm): {hr:5.1f}",
    ]
    if text:
        filas.append(text)
    for i, fila in enumerate(filas):
        _shadow_text(frame, fila, (x1 + pad, y1 + 24 + i * 22),
                     cv2.FONT_HERSHEY_PLAIN, 1.0)

def _norm_and_update(key: str, val: float) -> float:
    """Normaliza val a [0-1] usando rango histórico min-max adaptativo."""
    s = _BAR_STATS[key]
    # Actualiza extremos
    s["min"] = min(s["min"], val)
    s["max"] = max(s["max"], val)
    rng = max(s["max"] - s["min"], 1e-3)
    return np.clip((val - s["min"]) / rng, 0, 1)

# ➊  👇  ESTA FUNCIÓN DEBE ESTAR DEFINIDA A NIVEL DE MÓDULO
def _shadow_text(img: np.ndarray,
                 text: str,
                 org: Tuple[int, int],
                 font: int       = cv2.FONT_HERSHEY_DUPLEX,
                 scale: float    = 0.6,
                 color=(0,255,0),
                 thickness: int  = 1):
    x, y = org
    cv2.putText(img, text, (x+1, y+1), font, scale,
                (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), font, scale,
                color,     thickness, cv2.LINE_AA)

# ═══════════════════ Leyenda y Marcadores ═════════════════════════════
def draw_technologies(frame: np.ndarray,
                      pts68=None, mesh=None, face_rect=None):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_PLAIN

    if pts68 is not None:
        p = np.array(pts68, np.int32)
        cv2.polylines(frame, [p[36:42]], True, FLUO_GREEN, 1)
        cv2.polylines(frame, [p[42:48]], True, FLUO_GREEN, 1)
        _shadow_text(frame, "EAR", tuple(p[36] + (0, -10)),
                     font, 0.8)
        cv2.polylines(frame, [p[48:60]], True, FLUO_GREEN, 1)
        _shadow_text(frame, "MICRO", tuple(p[54] + (0, 22)),
                     font, 0.8)

    if face_rect is not None:
        x1, y1, x2, y2 = face_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), FLUO_GREEN, 1)
        _shadow_text(frame, "rPPG", (x1, y1 - 8), font, 0.8)

    if mesh is not None and hasattr(mesh, "landmark"):
        for idx in (474, 475, 476, 477):
            pt = mesh.landmark[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)),
                       2, FLUO_GREEN, -1)

    # Leyenda en la esquina inferior-derecha (misma que tu versión)
    ley_w, ley_h = 214, 84
    lx2, ly2 = w - 12, h - 12
    lx1, ly1 = lx2 - ley_w, ly2 - ley_h
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
    _shadow_text(frame, "Leyenda:", (lx1 + 8, ly1 + 22), font, 1.0)
    _shadow_text(frame, "Todo – VERDE", (lx1 + 8, ly1 + 46), font, 0.9)
