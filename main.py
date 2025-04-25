import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set page title
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection using YOLOv11n")
st.write("Upload an MRI or CT image to detect presence of a brain tumor.")

# Load trained model
model = YOLO("yolo11n.pt")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    with st.spinner("Analyzing..."):
        results = model.predict(image, imgsz=640)
        boxes = results[0].boxes
        annotated_img = results[0].plot()

    # Show annotated image
    st.image(annotated_img, caption="Prediction", use_column_width=True)

    # Show results table if boxes are detected
    if boxes is not None and len(boxes) > 0:
        st.subheader("Detected Results:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            st.write(f"🧠 **{label.capitalize()}** — Confidence: {conf:.2f}")
    else:
        st.write("No tumors detected.")
