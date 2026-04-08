import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# --- Load models once ---
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
yolo_model = YOLO("yolov8n.pt")

# --- Page configuration ---
st.set_page_config(
    page_title="Image Caption & Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar for options ---
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "YOLO Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

st.title("🖼️ Image Captioning & Object Detection")
st.markdown("Upload an image and automatically get a caption and detected objects!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")

    # --- Caption Generation ---
    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    # --- Object Detection ---
    with st.spinner("Detecting objects..."):
        results = yolo_model(image)
        detected_objects = [
            results[0].names[int(cls)]
            for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf)
            if conf > confidence_threshold
        ]
        detected_objects = sorted(list(set(detected_objects)))  # remove duplicates & sort

    # --- Layout: Side-by-side images ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected & Annotated Image")
        annotated_image = results[0].plot()
        st.image(annotated_image, use_column_width=True)

    # --- Captions & Detected Objects in a container ---
    with st.container():
        st.markdown("### Caption")
        st.info(caption)

        st.markdown("### Detected Objects")
        if detected_objects:
            st.success(", ".join(detected_objects))
        else:
            st.warning("No objects detected above confidence threshold.")