import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("DIP Project: Face Filter")
st.write("Upload a photo to see 'DIP' on your forehead!")

# 1. Load the Haar Cascade Face Detector
# This uses feature-based object detection to find the face coordinates
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. File Uploader
img_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(img_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # DIP PROCESSING PIPELINE:
    # Step A: Grayscale Conversion (Reduces noise/processing time)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step B: Detect Faces (Returns x, y, width, height)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Step C: Spatial Mapping for Text Overlay
    for (x, y, w, h) in faces:
        # We calculate the forehead position:
        # X: Start at x, then move 20% of the width in to center it better
        # Y: Start at y (top of head), then move 25% of the height DOWN
        text_x = x + int(w * 0.2)
        text_y = y + int(h * 0.25)

        # Draw "DIP" on the Forehead
        cv2.putText(img_bgr, "DIP", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Optional: Draw the bounding box for your project report
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert back to RGB to display in Streamlit
    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption='Processed Image with Forehead Filter', use_column_width=True)
    
    # Add a download button for your project submission
    st.download_button(label="Download Processed Image", 
                       data=cv2.imencode('.jpg', img_bgr)[1].tobytes(), 
                       file_name="dip_output.jpg", 
                       mime="image/jpeg")
