# optical_flow.py
import cv2, numpy as np

_prev = None                       # ← buffer de frame anterior

def reset():
    """Llama a esto cuando cambies de vídeo o reinicies cámara."""
    global _prev
    _prev = None

def flow_overlay(bgr, *, step=20, color=(0,255,255), mag_thr=0.2):
    """
    Dibuja flechas sobre el campo de flujo óptico (Farneback).

    • step      = separación (px) entre vectores.
    • color     = BGR de las flechas.
    • mag_thr   = umbral mínimo de módulo (para no saturar con ruido).
    """
    global _prev
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if _prev is None or _prev.shape != gray.shape:
        _prev = gray.copy()
        return bgr                                  # aún no hay flujo

    flow = cv2.calcOpticalFlowFarneback(_prev, gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = gray.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            if abs(dx)+abs(dy) < mag_thr:           # filtra flechas pequeñas
                continue
            cv2.arrowedLine(bgr,
                            (x, y),
                            (int(x+dx), int(y+dy)),
                            color, 1, tipLength=.3,
                            line_type=cv2.LINE_AA)
    _prev = gray
    return bgr
