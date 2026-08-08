import cv2, numpy as np, scipy.signal as sig

class PPGChrom:
    def __init__(self, fps, buf_size=300):
        self.fps = fps
        self.buf_size = buf_size
        self.buf = []

    def update(self, frame, face_roi):
        """
        Devuelve BPM estimado con CHROM usando 60 % central de la cara.
        """
        x, y, w, h = face_roi
        cx, cy = x + w // 2, y + h // 2
        rw, rh = int(w * 0.6), int(h * 0.6)
        x1 = max(cx - rw // 2, 0);  x2 = min(cx + rw // 2, frame.shape[1] - 1)
        y1 = max(cy - rh // 2, 0);  y2 = min(cy + rh // 2, frame.shape[0] - 1)

        roi = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0

        mean_rgb = roi.mean(axis=(0, 1))
        self.buf.append(mean_rgb)
        if len(self.buf) > self.buf_size:
            self.buf.pop(0)
        if len(self.buf) < self.buf_size:
            return 0.0

        sig_rgb = np.array(self.buf, dtype=np.float32).T
        b, a = sig.butter(2, [0.6 / self.fps * 2, 3 / self.fps * 2], btype="band")
        filt  = sig.filtfilt(b, a, sig_rgb)
        Xs    = 3 * filt[0] - 2 * filt[1]
        PPG   = Xs / filt[2]

        freqs = np.fft.rfftfreq(len(PPG), 1 / self.fps)
        spec  = np.abs(np.fft.rfft(PPG - PPG.mean()))
        mask  = (freqs >= 0.7) & (freqs <= 2.5)          # 42-150 BPM
        if not np.any(mask):
            return 0.0
        bpm = freqs[mask][np.argmax(spec[mask])] * 60
        return bpm
