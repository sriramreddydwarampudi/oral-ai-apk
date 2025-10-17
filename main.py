import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.popup import Popup

# ==========================
# 🔹 Load TensorFlow Lite Model
# ==========================
MODEL_PATH = "yolov8_640_float32.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
# 🔹 Main Camera + Detection Widget
# ==========================
class KivyCamera(Image):
    def __init__(self, capture, fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def preprocess(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32)
        input_tensor = np.expand_dims(resized, axis=0) / 255.0
        return input_tensor

    def detect_tflite(self, frame):
        input_tensor = self.preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        h, w, _ = frame.shape
        for det in output:
            x1, y1, x2, y2, score, class_id = det
            if score < 0.25:
                continue
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            label = f"{CLASS_NAMES[int(class_id)]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
        return frame

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 0)
            frame = self.detect_tflite(frame)
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture


# ==========================
# 🔹 Main App Layout
# ==========================
class DentalDetectionApp(App):
    def build(self):
        # Camera selection popup
        def select_camera(instance):
            dropdown = DropDown()
            for i in range(2):
                btn = Button(text=f"Camera {i}", size_hint_y=None, height=44)
                btn.bind(on_release=lambda btn: (dropdown.select(btn.text)))
                dropdown.add_widget(btn)

            mainbutton = Button(text='Select Camera', size_hint=(1, 0.1))
            dropdown.bind(on_select=lambda instance, x: setattr(mainbutton, 'text', x))
            dropdown.open(mainbutton)

            popup = Popup(title="Select Camera", content=mainbutton, size_hint=(0.8, 0.3))
            popup.open()
            return popup

        # Open back camera (0) by default
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        layout = BoxLayout(orientation='vertical')
        self.my_camera = KivyCamera(capture=self.capture)
        layout.add_widget(self.my_camera)
        layout.add_widget(Label(text="🦷 Dental Detection (TFLite)", size_hint=(1, 0.1)))
        return layout

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()


if __name__ == '__main__':
    DentalDetectionApp().run()