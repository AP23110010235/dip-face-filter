import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("DIP Project: Face Filter")
st.write("Upload a photo to apply the 'DIP' face filter.")

# 1. Load the DIP Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. File Uploader (Bypasses the connection error)
img_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(img_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # DIP PROCESS:
    # A. Grayscale Conversion (Standard DIP preprocessing)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # B. Feature Extraction (Detecting the Face)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # C. Spatial Transformation (Placing text on specific coordinates)
    for (x, y, w, h) in faces:
        # Draw "DIP" above the face
        cv2.putText(img_bgr, "DIP", (x, y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # Draw bounding box
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Display the result
    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption='Processed Image', use_column_width=True)
