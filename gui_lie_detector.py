import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import datetime
from visual_debug import draw_liescore, draw_emotions


from detector       import LieDetector
from camera_thread  import CameraThread
from visual_debug   import draw_liescore

MODEL_EMO   = r"C:/Users/Abigu/OneDrive/Escritorio/PULIDO/proyecto_lie_detector/modelo_emociones.h5"
LANDMARKS68 = r"C:/Users/Abigu/OneDrive/Escritorio/PULIDO/proyecto_lie_detector/shape_predictor_68_face_landmarks.dat"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lie Detector 3.0 — GUI")
        self.geometry("900x700")
        self.configure(bg="#111")

        # Detector y cámara
        print("[App] Inicializando detector y cámara...")
        self.detector = LieDetector(MODEL_EMO, LANDMARKS68)
        self.cam_thread = CameraThread(src=0, qsize=2)
        self.cam_thread.start()  # arranca la cámara
        self.running = False
        self.current_frame = None

        # Canvas para video
        self.video_label = tk.Label(self, bg="#222")
        self.video_label.pack(pady=10)

        # Botones Start / Stop
        control_frame = tk.Frame(self, bg="#111")
        control_frame.pack(pady=10)

        self.start_button = ttk.Button(control_frame, text="▶ Start", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = ttk.Button(control_frame, text="■ Stop", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10)

        self.screenshot_button = ttk.Button(control_frame, text="📸 Screenshot", command=self.save_screenshot)
        self.screenshot_button.grid(row=0, column=2, padx=10)

        # Checkboxes
        checkbox_frame = tk.Frame(self, bg="#111")
        checkbox_frame.pack(pady=5)

        self.var_mesh = tk.BooleanVar(value=True)
        self.var_bars = tk.BooleanVar(value=True)
        self.var_liebar = tk.BooleanVar(value=True)

        ttk.Checkbutton(checkbox_frame, text="Mesh", variable=self.var_mesh).grid(row=0, column=0, padx=10)
        ttk.Checkbutton(checkbox_frame, text="Bars", variable=self.var_bars).grid(row=0, column=1, padx=10)
        ttk.Checkbutton(checkbox_frame, text="Lie-bar", variable=self.var_liebar).grid(row=0, column=2, padx=10)

        # Estado
        self.status_label = tk.Label(self, text="Estado: Inactivo", fg="white", bg="#111", font=("Segoe UI", 12))
        self.status_label.pack(pady=5)

    def process_frame(self, frame):
        if not self.running:
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb.copy()

            # —— pasa el frame por el detector ————————————————
            result = self.detector.process(frame)

            # —— overlays seleccionables ————————————————
            if self.var_mesh.get():
                self.detector.draw_landmarks(frame_rgb)
            if self.var_bars.get():
                self.detector.draw_emotion_bars(frame_rgb)
            if self.var_liebar.get():
                draw_liescore(frame_rgb,
                            self.detector.lie_score,
                            anchor="bl",
                            offset=(20, 20))

            # —— NUEVO: tarjeta DeepFace / FER ————————————  # NEW
            draw_emotions(                                           # NEW
                frame_rgb,                                           # NEW
                deep_emo=result.get("deepface_emotion"),             # NEW
                fer_emo =result.get("fer_emotion"),                  # NEW
            )                                                        # NEW

            # —— pinta en el widget Tkinter ————————————————
            image  = Image.fromarray(frame_rgb).resize((720, 540))
            imgtk  = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        except Exception as e:
            print(f"[App] ❌ Error en process_frame: {e}")

    
    def start_camera(self):
        if self.running:
            messagebox.showinfo("Aviso", "La cámara ya está activa.")
            print("[App] Cámara ya corriendo.")
            return
        print("[App] Iniciando cámara...")
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Estado: Cámara activa")
        self.cam_thread.start()

    def stop_camera(self):
        if not self.running:
            messagebox.showinfo("Aviso", "La cámara ya está detenida.")
            print("[App] Cámara ya estaba detenida.")
            return
        print("[App] Deteniendo cámara...")
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Estado: Detenido")
        self.cam_thread.stop()

    def process_frame(self, frame):
        if not self.running:
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb.copy()
            result = self.detector.process(frame)

            print("DeepFace:", result.get("deepface_emotion"))
            print("FER     :", result.get("fer_emotion"))

            
            if self.var_mesh.get():
                self.detector.draw_landmarks(frame_rgb)
            if self.var_bars.get():
                self.detector.draw_emotion_bars(frame_rgb)
            if self.var_liebar.get():
                draw_liescore(frame_rgb,
                self.detector.lie_score,
                anchor="bl",
                offset=(20, 20))    # 20 px a la derecha y 20 px arriba
                # bottom-left


            image = Image.fromarray(frame_rgb).resize((720, 540))
            imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        except Exception as e:
            print(f"[App] ❌ Error en process_frame: {e}")

    def save_screenshot(self):
        if self.current_frame is None:
            messagebox.showwarning("Sin imagen", "Aún no hay frame para guardar.")
            return

        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_frame)
        messagebox.showinfo("Captura guardada", f"Se guardó como {filename}")
        print(f"[App] ✅ Screenshot guardada: {filename}")

    def on_closing(self):
        self.running = False
        if self.cam_thread is not None:
            self.cam_thread.stop()
        self.destroy()  # <— esto detiene correctamente el ciclo after
    
    def _loop(self):
        if not self.running or not self.winfo_exists():
            return
        self.process_frame()
        self.after(10, self._loop)



if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
