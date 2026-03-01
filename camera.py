import cv2

def get_camera(source=0, width=1280, height=720):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Force MJPG for USB cam performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Optional: try forcing 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap