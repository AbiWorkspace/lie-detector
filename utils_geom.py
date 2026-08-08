# utils_geom.py
import math, logging
from typing import Tuple, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────── EAR clásico ──────────────────────────
def compute_ear(pts: np.ndarray) -> float:
    """Eye-Aspect-Ratio de 6 puntos (párpado)"""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


# ────────────────────────── Pose por EPnP ─────────────────────────
_LAND_IDXS = [1, 33, 263, 61, 291, 199]          # nariz, ojos, boca, mentón
_MODEL_PTS = np.float32(
    [[  0,   0,   0],      # nose tip
     [-30, -65, -50],      # left eye
     [ 30, -65, -50],      # right eye
     [-30,  40, -50],      # left mouth
     [ 30,  40, -50],      # right mouth
     [  0, -20,-100]]      # chin
)

def solve_pose(lm2d, w: int, h: int) -> Tuple[float, float, float]:
    """
    EPnP sobre 6 puntos de MediaPipe → (yaw, pitch, roll) en °.  
    Si algo falla se devuelve (0,0,0) y se avisa por log.
    """
    try:
        image_pts = np.float32(
            [[lm2d.landmark[i].x * w,
              lm2d.landmark[i].y * h] for i in _LAND_IDXS]
        )

        cam = np.array([[w, 0, w / 2],
                        [0, w, h / 2],
                        [0, 0,   1  ]], dtype=np.float32)

        ok, rvec, _ = cv2.solvePnP(
            _MODEL_PTS, image_pts, cam, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError("solvePnP no converge")

        rmat, _ = cv2.Rodrigues(rvec)
        sy  = math.hypot(rmat[0, 0], rmat[1, 0])  # √(r00²+r10²)

        yaw   =  math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        pitch = -math.degrees(math.atan2(rmat[2, 0], sy))
        roll  =  math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        return yaw, pitch, roll

    except Exception as e:
        logger.warning(f"solve_pose error: {e}")
        return 0.0, 0.0, 0.0


# ────────────────────────── Pupila (MediaPipe iris) ──────────────
def pupil_diameter(lm, ids: Sequence[int], w: int, h: int) -> float:
    """Diámetro pupilar (px) normalizado al tamaño del frame."""
    xs = [lm.landmark[i].x for i in ids]
    ys = [lm.landmark[i].y for i in ids]
    return max(max(xs) - min(xs), max(ys) - min(ys)) * max(w, h)
