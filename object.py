import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # YOLOv8 model

# Load YOLO model (pre-trained on COCO dataset)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Using the nano version for speed

model = load_model()

# Streamlit UI
st.title("Object Identification using YOLOv8")
st.write("Upload an image, and the model will identify objects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Perform object detection
    results = model(image)

    # Convert results to OpenCV format
    for result in results:
        image_with_boxes = result.plot()  # Draw boxes on image

    # Convert back to PIL for displaying
    st.image(image_with_boxes, caption="Detected Objects", use_column_width=True)

    # Show detected objects with confidence scores
    st.subheader("Detected Objects:")
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            confidence = round(float(box.conf), 2)
            st.write(f"🔹 {class_name}: {confidence * 100:.2f}%")

st.write("Built with ❤️ using Streamlit and YOLOv8")
