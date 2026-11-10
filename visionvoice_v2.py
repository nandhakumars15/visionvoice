import cv2
import time
import queue
import threading
import itertools
import pyttsx3
from ultralytics import YOLO

CONF = 0.5
STABLE_FRAMES = 2
DEBUG = False
NORMAL_COOLDOWN = 6
URGENT_COOLDOWN = 3

VERY_CLOSE = 0.55
CLOSE = 0.25
NEARBY = 0.06

SMALL_OBJECTS = {"cell phone", "bottle", "cup", "remote", "mouse", "book", "scissors", "toothbrush", "fork", "knife", "spoon"}
SMALL_BOOST = 3

IGNORE_CLASSES = {"person"}

speech_q = queue.PriorityQueue()
counter = itertools.count()


def speech_worker():
    while True:
        _, _, text = speech_q.get()
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("SPEECH ERROR:", e)


threading.Thread(target=speech_worker, daemon=True).start()


def speak(text, urgent=False):
    if not urgent and speech_q.qsize() >= 1:
        return
    if DEBUG:
        print("SPEAK:", text)
    speech_q.put((0 if urgent else 1, next(counter), text))


def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < 2 * frame_width / 3:
        return "center"
    return "right"


def get_distance(area_ratio):
    if area_ratio >= VERY_CLOSE:
        return "very close"
    if area_ratio >= CLOSE:
        return "close"
    if area_ratio >= NEARBY:
        return "nearby"
    return "far"


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam. Try changing 0 to 1.")
    raise SystemExit

streaks = {}
last_spoken = {}

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        results = model(frame, conf=CONF, verbose=False)
        if DEBUG:
            raw = [f"{model.names[int(b.cls[0])]} {float(b.conf[0]):.2f}" for b in results[0].boxes]
            print("RAW:", raw)

        seen = {}
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = model.names[int(box.cls[0])]
            if name != "person" and float(box.conf[0]) < 0.65:
                continue
            direction = get_direction((x1 + x2) / 2, w)
            area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
            if name in IGNORE_CLASSES:
                continue
            if name in SMALL_OBJECTS:
                area_ratio = min(area_ratio * SMALL_BOOST, 1.0)
            key = (name, direction)
            seen[key] = max(seen.get(key, 0), area_ratio)

        new_streaks = {k: streaks.get(k, 0) + 1 for k in seen}
        for k, v in streaks.items():
            if k not in seen and v > 1:
                new_streaks[k] = v - 1
        streaks = new_streaks

        stable = [(k, seen[k]) for k in seen if streaks[k] >= STABLE_FRAMES]
        stable.sort(key=lambda item: item[1], reverse=True)

        spoke_normal = False
        now = time.time()
        for (name, direction), ratio in stable:
            distance = get_distance(ratio)
            if DEBUG and name != "person":
                print(f"{name} {direction} ratio={ratio:.3f} {distance} streak={streaks[(name, direction)]}")
            if distance == "far":
                continue

            urgent = direction == "center" and distance in ("very close", "close")
            if urgent:
                message = f"Warning! {name} ahead, {distance}"
                cooldown = URGENT_COOLDOWN
            else:
                message = f"{name} on your {direction}, {distance}"
                cooldown = NORMAL_COOLDOWN

            cooldown_key = (name, "urgent" if urgent else direction)
            if now - last_spoken.get(cooldown_key, 0) < cooldown:
                continue

            if urgent:
                if speech_q.qsize() < 2:
                    speak(message, urgent=True)
                    last_spoken[cooldown_key] = now
            elif not spoke_normal:
                speak(message)
                last_spoken[cooldown_key] = now
                spoke_normal = True

        annotated = results[0].plot()
        for i, ((name, direction), ratio) in enumerate(stable):
            text = f"{name} {direction} {ratio:.2f} {get_distance(ratio)}"
            cv2.putText(annotated, text, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("VisionVoice", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()