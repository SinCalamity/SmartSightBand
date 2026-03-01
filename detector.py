import cv2
import torch
from ultralytics import YOLO
from collections import deque


class ObjectDetector:
    def __init__(self, model_path="smartsight_ph.pt", confidence_threshold=0.5):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = confidence_threshold

        # Focal length (calibrated)
        self.focal_length = 1028

        # 🔧 Minor calibration factor (fine-tuning)
        self.calibration_factor = 0.97

        # Known object widths (cm)
        self.known_widths = {
            "person": 55,
            "chair": 45,
            "table": 120,
            "door": 90, 
            "stairs": 120
        }

        self.distance_buffers = {}

    # ------------------------------

    def semantic_distance(self, distance):

        if distance <= 40:
            return "very close"
        elif distance <= 80:
            return "close"
        elif distance <= 150:
            return "nearby"
        else:
            return "far"

    # ------------------------------

    def get_direction(self, x1, x2, frame_width):

        center_x = (x1 + x2) // 2

        if center_x < frame_width * 0.33:
            return "on your left"
        elif center_x > frame_width * 0.66:
            return "on your right"
        else:
            return "in front"

    # ------------------------------

    def estimate_distance(self, class_name, box_width):

        if class_name not in self.known_widths or box_width <= 0:
            return None

        real_width = self.known_widths[class_name]
        raw_distance = (real_width * self.focal_length) / box_width

        # 🔧 Apply calibration factor
        raw_distance *= self.calibration_factor

        # Close-range correction
        if raw_distance < 70:
            raw_distance *= 0.85

        # Small-person clamp
        if class_name == "person" and raw_distance < 30:
            raw_distance = 30

        if class_name not in self.distance_buffers:
            self.distance_buffers[class_name] = deque(maxlen=5)

        self.distance_buffers[class_name].append(raw_distance)

        smoothed = sum(self.distance_buffers[class_name]) / len(self.distance_buffers[class_name])

        return round(smoothed, 1)

    # ------------------------------

    def detect(self, frame):

        results = self.model(frame, imgsz=256, verbose=False)
        detections = []

        frame_width = frame.shape[1]

        for r in results:
            for box in r.boxes:

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id].lower()

                # -------- Class-specific confidence filters --------
                if class_name == "chair" and confidence < 0.7:
                    continue

                if class_name == "door" and confidence < 0.4:
                    continue

                if confidence < self.conf_threshold:
                    continue

                # -------- Geometry rule: table vs chair --------
                width = x2 - x1
                height = y2 - y1

                if class_name == "chair" and width > height:
                    class_name = "table"

                # -------- Distance estimation --------
                distance = self.estimate_distance(class_name, width)

                if distance is not None:
                    semantic = self.semantic_distance(distance)
                    direction = self.get_direction(x1, x2, frame_width)
                else:
                    semantic = None
                    direction = None

                detections.append({
                    "class": class_name,
                    "box": (x1, y1, x2, y2),
                    "distance": distance,
                    "semantic": semantic,
                    "direction": direction
                })

        return detections

    # ------------------------------

    @staticmethod
    def draw_detections(frame, detections):

        for d in detections:
            x1, y1, x2, y2 = d["box"]

            label = d["class"]

            if d["distance"] is not None:
                label += f" {d['distance']}cm"

            if d["semantic"]:
                label += f" {d['semantic']}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

        return frame