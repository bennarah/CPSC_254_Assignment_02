import torch
import cv2
import os

# ─── Configuration ────────────────────────────────────────────────────────────

IMAGE_PATH = "datasets/objects/objects.jpg"

ACTUAL_OBJECTS = ["cat", "dog", "person"]

# ─── Load YOLOv5 pretrained model from torch.hub ─────────────────────────────

print("Loading YOLOv5 model (this may download on first run)...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
print("Model loaded successfully.\n")

# ─── Run object detection ─────────────────────────────────────────────────────

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found at: {IMAGE_PATH}\n"
        "Please make sure objects.jpg is in the datasets/objects/ folder."
    )

print(f"Running detection on: {IMAGE_PATH}\n")
results = model(IMAGE_PATH)

# ─── Extract detections ───────────────────────────────────────────────────────

detections = results.pandas().xyxy[0]  # columns: xmin, ymin, xmax, ymax, confidence, class, name

predicted_objects = []
bounding_boxes = []

for _, row in detections.iterrows():
    label      = row['name']
    confidence = row['confidence']
    x1 = int(row['xmin'])
    y1 = int(row['ymin'])
    x2 = int(row['xmax'])
    y2 = int(row['ymax'])

    predicted_objects.append(label)
    bounding_boxes.append((label, confidence, x1, y1, x2, y2))

# ─── Print required output ────────────────────────────────────────────────────

print("Actual objects in image:")
print(ACTUAL_OBJECTS)

print("\nPredicted objects:")
print([obj for obj, *_ in bounding_boxes])

print("\nDetected objects with bounding boxes:")
if len(bounding_boxes) == 0:
    print("  No objects detected. Try a different image or lower the confidence threshold.")
else:
    for label, conf, x1, y1, x2, y2 in bounding_boxes:
        print(f"  Object: {label:15s} | Confidence: {conf:.2f} | "
              f"BBox: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")

# ─── Draw bounding boxes and save annotated image ─────────────────────────────

img = cv2.imread(IMAGE_PATH)

for label, conf, x1, y1, x2, y2 in bounding_boxes:
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=(0, 255, 0), thickness=2)

output_path = "datasets/objects/objects_detected.jpg"
cv2.imwrite(output_path, img)
print(f"\nAnnotated image saved to: {output_path}")
