import cv2
import time
from collections import Counter

from camera import get_camera
from detector import ObjectDetector
from tts import speak_async


TARGET_FPS = 30
WAIT_TIME = int(1000 / TARGET_FPS)

SPEAK_DISTANCE_LIMIT = 120
REPEAT_LIMIT = 3
REPEAT_DELAY = 1.5


PRIORITY_ORDER = ["person", "stairs", "chair", "table", "door"]


def select_priority_group(detections):
    valid = [d for d in detections if d["distance"] is not None and d["distance"] <= SPEAK_DISTANCE_LIMIT]

    if not valid:
        return None, 0

    class_counts = Counter([d["class"] for d in valid])

    for p in PRIORITY_ORDER:
        if p in class_counts:
            group = [d for d in valid if d["class"] == p]
            return group, class_counts[p]

    return None, 0


def get_group_direction(group, frame_width):
    centers = [ (d["box"][0] + d["box"][2]) / 2 for d in group ]
    avg_center = sum(centers) / len(centers)

    if avg_center < frame_width * 0.33:
        return "on your left"
    elif avg_center > frame_width * 0.66:
        return "on your right"
    else:
        return "in front"


def main():

    cap = get_camera(source=1, width=1280, height=720)

    if not cap.isOpened():
        print("[ERROR] Camera failed to open.")
        return

    detector = ObjectDetector("smartsight_ph3.pt", confidence_threshold=0.65)

    print("SmartSight Running (Multi-Object Mode)")
    print("Press S to scan | Press Q to quit")

    frame_count = 0
    DETECTION_INTERVAL = 2
    detections = []

    last_label = None
    repeat_count = 0
    last_speech_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        if frame_count % DETECTION_INTERVAL == 0:
            detections = detector.detect(frame)

        display = detector.draw_detections(frame, detections)

        group, count = select_priority_group(detections)

        current_time = time.time()

        if group:
            obj_class = group[0]["class"]
            frame_width = frame.shape[1]
            direction = get_group_direction(group, frame_width)

            if count > 1:
                label = f"{count} {obj_class}s {direction}"
            else:
                semantic = group[0]["semantic"]
                label = f"{obj_class} {semantic} {direction}"

            if label != last_label:
                repeat_count = 0

            if repeat_count < REPEAT_LIMIT and current_time - last_speech_time > REPEAT_DELAY:
                speak_async(label)
                repeat_count += 1
                last_speech_time = current_time
                last_label = label

        else:
            last_label = None
            repeat_count = 0

        cv2.imshow("SmartSight Assistive Detection", display)

        key = cv2.waitKey(WAIT_TIME) & 0xFF

        if key == ord('s') and group:
            speak_async(label)
            last_label = label
            repeat_count = 1
            last_speech_time = current_time

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()