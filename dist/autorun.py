import cv2
import numpy as np
import requests
import json
import time
import threading

CAM_URL = "http://192.168.43.112:81/stream"
ESP32_SENSOR_URL = "http://192.168.43.201/data"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

latest_frame = None
frame_lock = threading.Lock()

def get_frame_from_stream():
    """Ambil frame dari stream ESP32-CAM menggunakan OpenCV."""
    cap = cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():
        print("ESP32-CAM tidak terbuka, coba fallback...")
        return None

    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

def get_frame_from_local():
    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        print("Kamera laptop tidak dapat dibuka.")
        return None
    ret, frame = cap_local.read()
    cap_local.release()
    if ret:
        return frame
    return None

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 250, 0), 2)

    cv2.putText(frame, f"deteksi: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def camera_loop():
    global latest_frame
    while True:
        frame = get_frame_from_stream()
        if frame is None:
            print("Fallback ke kamera lokal...")
            frame = get_frame_from_local()

        if frame is not None:
            processed = process_frame(frame)
            with frame_lock:
                latest_frame = processed.copy()
        else:
            print("Tidak dapat memperoleh frame dari kamera manapun.")

        time.sleep(0.1)  

def snapshot_loop():
    while True:
        with frame_lock:
            if latest_frame is not None:
                cv2.imwrite("detected.jpg", latest_frame)
        time.sleep(1)

def sensor_loop():
    while True:
        try:
            response = requests.get(ESP32_SENSOR_URL, timeout=3)
            if response.status_code == 200:
                data = response.json()
                with open("sensor.json", "w") as f:
                    json.dump(data, f)
                print("Sensor:", data)
            else:
                print("Gagal ambil data sensor:", response.status_code)
        except Exception as e:
            print("Ambil sensor:", e)
        time.sleep(2)

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=snapshot_loop, daemon=True).start()
    threading.Thread(target=sensor_loop, daemon=True).start()

    while True:
        time.sleep(1)
