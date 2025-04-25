from ultralytics import YOLO

# Load the trained model
model = YOLO("yolo11n.pt")

# Evaluate the model
metrics = model.val(data="brain-tumor.yaml")

# Print metrics
print("📊 Evaluation Results:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")