import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("DIP Project: Face Filter")
st.write("Upload your photo to see 'DIP' centered on your forehead.")

# 1. Load the Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. File Uploader
img_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(img_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # --- DIP PROCESSING PIPELINE ---
    
    # Step A: Grayscale Conversion (Reduces data complexity for detection)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step B: Face Detection (Returns x, y, width, height)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Step C: Spatial Mapping for Text Placement
    for (x, y, w, h) in faces:
        # ADJUSTING COORDINATES FOR FOREHEAD:
        # Move X to the right by 25% of face width to center "DIP"
        # Move Y down by 20% of face height to move from top-edge to forehead
        text_x = x + int(w * 0.25)
        text_y = y + int(h * 0.20)

        # Draw the "DIP" text on the calculated forehead position
        cv2.putText(img_bgr, "DIP", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Draw the bounding box (Blue color)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # --- OUTPUT ---
    
    # Convert back to RGB for Streamlit display
    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption='Processed Image: Text on Forehead', use_column_width=True)
    
    # Add a download button for your project submission
    st.download_button(label="Download Processed Image", 
                       data=cv2.imencode('.jpg', img_bgr)[1].tobytes(), 
                       file_name="dip_forehead_filter.jpg", 
                       mime="image/jpeg")
