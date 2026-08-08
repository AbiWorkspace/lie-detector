import cv2
import threading

class CameraThread(threading.Thread):
    def __init__(self, source):
        super().__init__()    # <--- ESTA LÍNEA ES CLAVE
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.running = True

        # Intenta obtener el FPS real del video (si es archivo), sino usa 30
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps is None or self.fps <= 0 or self.fps > 240:
            self.fps = 30

        self.lock = threading.Lock()
        self._frame = None

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self._frame = frame

    def read(self):
        with self.lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
