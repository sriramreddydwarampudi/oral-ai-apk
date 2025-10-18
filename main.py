import cv2
import numpy as np
import os
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

# Try requesting Android permissions (runs harmlessly on PC)
try:
    from android.permissions import request_permissions, Permission
    ANDROID = True
except ImportError:
    ANDROID = False

# ==========================
# 🔹 TensorFlow Lite Model
# ==========================
MODEL_PATH = "yolov8_640_float32.tflite"
TFLITE_AVAILABLE = False
interpreter = None
input_details = output_details = None

try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        TFLITE_AVAILABLE = True
        print("✅ TensorFlow Lite model loaded successfully")
    else:
        print(f"⚠️ Model file not found: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ TensorFlow Lite not available: {e}")

# ==========================
# 🔹 Class Names
# ==========================
CLASS_NAMES = [
    "Normal",
    "Initial Caries",
    "Moderate Caries",
    "Severe Caries",
    "Tooth Stain",
    "Dental Calculus",
    "Other Lesions"
]

# ==========================
# 🔹 Camera Widget
# ==========================
class KivyCamera(Image):
    def __init__(self, capture, fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        self.frame_count = 0
        Clock.schedule_interval(self.update, 1.0 / fps)

    def preprocess(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32)
        input_tensor = np.expand_dims(resized / 255.0, axis=0)
        return input_tensor

    def postprocess(self, output, frame):
        """
        Simplified placeholder postprocessing.
        NOTE: Real YOLOv8 TFLite output requires decoding (NMS, etc.).
        This just demonstrates bounding boxes visually.
        """
        h, w, _ = frame.shape
        try:
            # Example: assume (N,6) [x1,y1,x2,y2,score,class]
            for det in output:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, score, cls_id = det
                if score < 0.25:
                    continue
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = CLASS_NAMES[int(cls_id) % len(CLASS_NAMES)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}",
                            (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(frame, f"Postprocess error: {str(e)[:40]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

    def detect_tflite(self, frame):
        """Run TFLite detection if available"""
        if not TFLITE_AVAILABLE or interpreter is None:
            cv2.putText(frame, "Camera only - TFLite not loaded", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            return frame

        try:
            input_tensor = self.preprocess(frame)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output = np.squeeze(output_data)
            self.postprocess(output, frame)
        except Exception as e:
            cv2.putText(frame, f"TFLite error: {str(e)[:40]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret or frame is None or frame.size == 0:
            return

        self.frame_count += 1
        frame = self.detect_tflite(frame)

        # Convert to Kivy texture (use RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)
        buf = frame_rgb.tobytes()
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.texture = texture

# ==========================
# 🔹 Main App
# ==========================
class DentalDetectionApp(App):
    def build(self):
        if ANDROID:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])

        # Try to open a camera
        self.capture = None
        for cam_idx in [0, 1, 2]:
            cap = cv2.VideoCapture(cam_idx)
            if cap.isOpened():
                print(f"✅ Camera {cam_idx} opened successfully")
                self.capture = cap
                break
            cap.release()

        if self.capture is None:
            print("❌ Could not open any camera, exiting")
            raise SystemExit("Camera not found")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # UI Layout
        layout = BoxLayout(orientation='vertical')

        # Camera widget
        self.camera_view = KivyCamera(capture=self.capture)
        layout.add_widget(self.camera_view)

        # Info label
        status = "🦷 Dental Detection"
        status += " (AI Active)" if TFLITE_AVAILABLE else " (Camera Only)"
        label = Label(text=status, size_hint=(1, 0.12), font_size='18sp', color=(1, 1, 1, 1))
        layout.add_widget(label)

        # Exit button
        btn_box = BoxLayout(size_hint=(1, 0.08), padding=5, spacing=5)
        exit_btn = Button(text="Exit", background_color=(0.8, 0.2, 0.2, 1), font_size='16sp')
        exit_btn.bind(on_press=self.stop_app)
        btn_box.add_widget(exit_btn)
        layout.add_widget(btn_box)

        return layout

    def stop_app(self, *args):
        self.stop()

    def on_stop(self):
        if hasattr(self, 'capture') and self.capture and self.capture.isOpened():
            self.capture.release()
            print("✅ Camera released")

# ==========================
# 🔹 Run App
# ==========================
if __name__ == '__main__':
    DentalDetectionApp().run()
