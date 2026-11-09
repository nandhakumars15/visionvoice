import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)   # 0 = default webcam

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()          # draws boxes on the frame

    cv2.imshow("VisionVoice", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()