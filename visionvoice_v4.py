import cv2
import os
import time
import asyncio
import ctypes
import tempfile
import queue
import threading
import itertools
import pyttsx3
import torch
import easyocr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
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

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TAMIL_VOICE = "ta-IN-PallaviNeural"

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    TESS_OK = os.path.exists(TESSERACT_PATH)
except ImportError:
    TESS_OK = False

speech_q = queue.PriorityQueue()
counter = itertools.count()


def has_tamil(text):
    return any("\u0b80" <= c <= "\u0bff" for c in text)


def speak_tamil_online(text):
    import edge_tts
    path = os.path.join(tempfile.gettempdir(), "visionvoice_tamil.mp3")
    asyncio.run(edge_tts.Communicate(text, TAMIL_VOICE).save(path))
    mci = ctypes.windll.winmm.mciSendStringW
    mci("close vv", None, 0, None)
    mci(f'open "{path}" type mpegvideo alias vv', None, 0, None)
    mci("play vv wait", None, 0, None)
    mci("close vv", None, 0, None)


def speech_worker():
    while True:
        _, _, text = speech_q.get()
        if has_tamil(text):
            try:
                speak_tamil_online(text)
                continue
            except Exception as e:
                print("TAMIL SPEECH ERROR:", e)
                text = "Tamil text detected, but Tamil speech needs an internet connection"
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

BLIP_NAME = "Salesforce/blip-image-captioning-base"
blip_state = {"ready": False, "busy": False, "caption": ""}
blip_objs = {}


def load_blip():
    blip_objs["processor"] = BlipProcessor.from_pretrained(BLIP_NAME)
    blip_objs["model"] = BlipForConditionalGeneration.from_pretrained(BLIP_NAME)
    blip_objs["model"].eval()
    blip_state["ready"] = True
    speak("Scene description is ready. Press D to describe.", urgent=True)


def describe(frame):
    blip_state["busy"] = True
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = blip_objs["processor"](images=image, return_tensors="pt")
        with torch.no_grad():
            out = blip_objs["model"].generate(
                **inputs,
                max_new_tokens=40,
                min_new_tokens=12,
                num_beams=5,
                repetition_penalty=1.5,
            )
        caption = blip_objs["processor"].decode(out[0], skip_special_tokens=True)
        blip_state["caption"] = caption
        speak(f"I see {caption}", urgent=True)
    except Exception as e:
        print("DESCRIBE ERROR:", e)
    finally:
        blip_state["busy"] = False


def request_describe(frame):
    if not blip_state["ready"]:
        speak("Scene description is still loading", urgent=True)
        return
    if blip_state["busy"]:
        return
    speak("Describing", urgent=True)
    threading.Thread(target=describe, args=(frame,), daemon=True).start()

OCR_MIN_CONF = 0.6
TESS_MIN_CONF = 60
ROI_MARGIN = 0.15
OCR_MAX_CHARS = 300
ocr_state = {"ready": False, "busy": False, "text": ""}
ocr_objs = {}


def load_ocr():
    ocr_objs["reader"] = easyocr.Reader(["en"], gpu=False)
    ocr_state["ready"] = True
    speak("Text reading is ready. Press R to read.", urgent=True)


def crop_center(frame):
    h, w = frame.shape[:2]
    y1, y2 = int(h * ROI_MARGIN), int(h * (1 - ROI_MARGIN))
    x1, x2 = int(w * ROI_MARGIN), int(w * (1 - ROI_MARGIN))
    return frame[y1:y2, x1:x2]


def read_tamil(frame):
    if not TESS_OK:
        return ""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(
        gray, lang="tam", config="--psm 11", output_type=pytesseract.Output.DICT
    )
    words = []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.replace("\u200c", "").replace("\u200d", "").strip()
        if not word or float(conf) < TESS_MIN_CONF:
            continue
        if any("\u0b80" <= c <= "\u0bff" for c in word):
            words.append(word)
    return " ".join(words)


def read_text(frame):
    ocr_state["busy"] = True
    try:
        frame = crop_center(frame)
        text = read_tamil(frame)
        if not text:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            found = ocr_objs["reader"].readtext(rgb)
            found = [f for f in found if f[2] >= OCR_MIN_CONF and len(f[1].strip()) >= 2]
            found.sort(key=lambda f: (round(f[0][0][1] / 40), f[0][0][0]))
            text = " ".join(f[1] for f in found).strip()
        text = text[:OCR_MAX_CHARS]
        if text:
            ocr_state["text"] = text
            print("OCR:", text)
            speak(text, urgent=True)
        else:
            ocr_state["text"] = ""
            speak("No text found. Move closer or hold steady.", urgent=True)
    except Exception as e:
        print("OCR ERROR:", e)
    finally:
        ocr_state["busy"] = False


def request_read(frame):
    if not ocr_state["ready"]:
        speak("Text reading is still loading", urgent=True)
        return
    if ocr_state["busy"]:
        return
    speak("Reading", urgent=True)
    threading.Thread(target=read_text, args=(frame,), daemon=True).start()


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

threading.Thread(target=load_blip, daemon=True).start()
threading.Thread(target=load_ocr, daemon=True).start()

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
        if blip_state["caption"]:
            cv2.putText(annotated, blip_state["caption"], (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if ocr_state["text"] and ocr_state["text"].isascii():
            cv2.putText(annotated, ocr_state["text"][:60], (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.rectangle(annotated,
                      (int(w * ROI_MARGIN), int(h * ROI_MARGIN)),
                      (int(w * (1 - ROI_MARGIN)), int(h * (1 - ROI_MARGIN))),
                      (255, 255, 255), 1)
        cv2.imshow("VisionVoice", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            request_describe(frame.copy())
        if key == ord("r"):
            request_read(frame.copy())
finally:
    cap.release()
    cv2.destroyAllWindows()