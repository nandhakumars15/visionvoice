import cv2
import time
import threading
import pyttsx3
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

engine = pyttsx3.init()
last_spoken = {}
speaking = False

def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < 2 * frame_width / 3:
        return "center"
    return "right"

def _say(text):
    global speaking
    speaking = True
    engine.say(text)
    engine.runAndWait()
    speaking = False

def speak(text, cooldown=4):
    now = time.time()
    if speaking:
        return
    if now - last_spoken.get(text, 0) > cooldown:
        last_spoken[text] = now
        threading.Thread(target=_say, args=(text,), daemon=True).start()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w, _ = frame.shape
    results = model(frame, conf=0.5, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = model.names[int(box.cls[0])]
        direction = get_direction((x1 + x2) / 2, w)
        speak(f"{name} on your {direction}")

    cv2.imshow("VisionVoice", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()