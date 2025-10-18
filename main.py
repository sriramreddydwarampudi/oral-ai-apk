import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import os

# Request Android permissions
try:
    from android.permissions import request_permissions, Permission
    request_permissions([
        Permission.CAMERA,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE
    ])
    ANDROID = True
    print("✅ Running on Android - Permissions requested")
except ImportError:
    ANDROID = False
    print("ℹ️ Not running on Android")

# Try to load TensorFlow Lite (optional - won't break if not available)
MODEL_PATH = "yolov8_640_float32.tflite"
TFLITE_AVAILABLE = False
interpreter = None

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
    print(f"⚠️ TFLite not available: {e}")

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
# 🔹 Camera Widget with Detection
# ==========================
class KivyCamera(Image):
    def __init__(self, capture, fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        self.frame_count = 0
        self.fps = fps
        Clock.schedule_interval(self.update, 1.0 / fps)

    def preprocess(self, frame):
        """Preprocess frame for TFLite model"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32)
        input_tensor = np.expand_dims(resized, axis=0) / 255.0
        return input_tensor

    def detect_tflite(self, frame):
        """Run detection if TFLite is available"""
        if not TFLITE_AVAILABLE or interpreter is None:
            # Draw status message
            cv2.putText(frame, "TFLite not loaded - Camera only", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame

        try:
            input_tensor = self.preprocess(frame)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            h, w, _ = frame.shape
            detection_count = 0
            
            for det in output:
                x1, y1, x2, y2, score, class_id = det
                if score < 0.25:
                    continue
                    
                detection_count += 1
                x1 = int(x1 * w / 640)
                y1 = int(y1 * h / 640)
                x2 = int(x2 * w / 640)
                y2 = int(y2 * h / 640)
                
                class_idx = int(class_id)
                if class_idx < len(CLASS_NAMES):
                    label = f"{CLASS_NAMES[class_idx]}: {score:.2f}"
                else:
                    label = f"Unknown: {score:.2f}"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(frame, (x1, y1 - text_height - 5),
                            (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Show detection count
            cv2.putText(frame, f"Detections: {detection_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
        except Exception as e:
            print(f"Detection error: {e}")
            cv2.putText(frame, f"Error: {str(e)[:40]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def update(self, dt):
        """Update frame from camera"""
        ret, frame = self.capture.read()
        if ret:
            self.frame_count += 1
            
            # Apply detection
            frame = self.detect_tflite(frame)
            
            # Convert to Kivy texture
            frame = cv2.flip(frame, 0)
            buf = frame.tobytes()
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]),
                colorfmt='bgr'
            )
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture


# ==========================
# 🔹 Main App
# ==========================
class DentalDetectionApp(App):
    def build(self):
        # Verify model file
        if os.path.exists(MODEL_PATH):
            print(f"✅ Model file found: {MODEL_PATH}")
        else:
            print(f"⚠️ Model file not found: {MODEL_PATH}")
        
        # Try to open camera
        self.capture = None
        camera_indices = [0, 1, 2] if ANDROID else [0]
        
        for cam_idx in camera_indices:
            try:
                cap = cv2.VideoCapture(cam_idx)
                if cap.isOpened():
                    print(f"✅ Camera {cam_idx} opened successfully")
                    self.capture = cap
                    break
                cap.release()
            except Exception as e:
                print(f"Failed to open camera {cam_idx}: {e}")
        
        if self.capture is None:
            print("❌ Could not open any camera, using fallback")
            self.capture = cv2.VideoCapture(0)
        
        # Configure camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get actual camera resolution
        actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution: {actual_width}x{actual_height}")

        # Build UI
        layout = BoxLayout(orientation='vertical')
        
        # Camera view
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        layout.add_widget(self.my_camera)
        
        # Status label
        status_text = "🦷 Dental Detection"
        if TFLITE_AVAILABLE:
            status_text += " (AI Active)"
        else:
            status_text += " (Camera Only)"
        
        info_label = Label(
            text=status_text,
            size_hint=(1, 0.12),
            font_size='18sp',
            color=(1, 1, 1, 1)
        )
        layout.add_widget(info_label)
        
        # Control buttons
        button_layout = BoxLayout(
            orientation='horizontal',
            size_hint=(1, 0.08),
            spacing=5,
            padding=5
        )
        
        # Exit button
        exit_btn = Button(
            text="Exit",
            background_color=(0.8, 0.2, 0.2, 1),
            font_size='16sp'
        )
        exit_btn.bind(on_press=self.stop_app)
        button_layout.add_widget(exit_btn)
        
        layout.add_widget(button_layout)
        
        return layout
    
    def stop_app(self, *args):
        """Stop the application"""
        self.stop()
    
    def on_stop(self):
        """Cleanup on app stop"""
        if hasattr(self, 'capture') and self.capture and self.capture.isOpened():
            self.capture.release()
            print("✅ Camera released")


if __name__ == '__main__':
    DentalDetectionApp().run()
