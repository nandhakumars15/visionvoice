from ultralytics import YOLO

model = YOLO("yolov8n.pt")   
results = model("test.jpg.jpeg")

for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        conf = float(box.conf[0])
        print(f"{name}: {conf:.2f}")

results[0].show()  