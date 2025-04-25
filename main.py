import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime
import os

# Set Streamlit page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Title
st.title("🧠 Brain Tumor Detection using YOLOv11n")
st.write("Upload an MRI or CT image to detect presence of a brain tumor.")

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq_api(prompt):
    """Call Groq API to get AI-powered explanation or suggestions."""
    if not GROQ_API_KEY:
        return "❌ Error: Groq API key not found."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error calling Groq API: {e}"

def generate_pdf_report(uploaded_image, annotated_image, results_text, explanation, suggestions):
    """Generate a PDF report from detection output."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "🧠 Brain Tumor Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Uploaded Image
    c.drawString(50, height - 100, "Uploaded Image:")
    img_data = BytesIO()
    uploaded_image.save(img_data, format="PNG")
    img = ImageReader(img_data)
    c.drawImage(img, 50, height - 300, width=200, height=200)

    # Annotated Image
    c.drawString(50, height - 330, "Detection Result:")
    annotated_img_data = BytesIO()
    Image.fromarray(annotated_image).save(annotated_img_data, format="PNG")
    annotated_img = ImageReader(annotated_img_data)
    c.drawImage(annotated_img, 50, height - 530, width=200, height=200)

    # Results Text
    c.drawString(50, height - 560, "Detection Summary:")
    text_object = c.beginText(50, height - 580)
    text_object.setFont("Helvetica", 10)
    for line in results_text.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)

    # Explanation
    c.drawString(50, height - 640, "Explanation:")
    text_object = c.beginText(50, height - 660)
    text_object.setFont("Helvetica", 10)
    for line in explanation.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)

    # Suggestions
    c.drawString(50, height - 740, "Recommendations:")
    text_object = c.beginText(50, height - 760)
    text_object.setFont("Helvetica", 10)
    for line in suggestions.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# File uploader
uploaded_file = st.file_uploader("Upload an MRI or CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📤 Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Analyzing the scan..."):
        results = model.predict(image, imgsz=640)
        boxes = results[0].boxes
        annotated_img = results[0].plot()

    # Show result image
    st.image(annotated_img, caption="📊 Model Detection Output", use_column_width=True)

    # Result Text Summary
    results_text = ""
    if boxes is not None and len(boxes) > 0:
        st.subheader("🔍 Detected Tumors")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            line = f"🧠 {label.capitalize()} — Confidence: {conf:.2f}"
            st.write(line)
            results_text += line + "\n"
    else:
        results_text = "✅ No tumors detected."
        st.success(results_text)

    # Groq Explanation
    st.subheader("📘 Explanation")
    explanation_prompt = f"""
    You are a medical AI assistant. Explain the following brain tumor detection results in simple language suitable for patients:
    {results_text}
    Briefly describe what the model found and its relevance.
    """
    explanation = call_groq_api(explanation_prompt)
    st.write(explanation)

    # Groq Suggestions
    st.subheader("📌 Recommendations")
    suggestions_prompt = f"""
    Based on the brain tumor detection results below, suggest clear, patient-friendly next steps.
    {results_text}
    Include things like consulting a neurologist, MRI follow-up, and avoiding self-diagnosis.
    """
    suggestions = call_groq_api(suggestions_prompt)
    st.write(suggestions)

    # PDF Download
    st.subheader("📄 Download Full Report")
    pdf_buffer = generate_pdf_report(image, annotated_img, results_text, explanation, suggestions)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="tumor_detection_report.pdf",
        mime="application/pdf"
    )
