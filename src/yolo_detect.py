from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_and_save(image_path, save_path):
    results = model(image_path)

    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_objects.append(label)

        # 🔥 Save PNG
        r.save(filename=save_path)

    return list(set(detected_objects))