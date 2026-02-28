import cv2
import streamlit as st
import numpy as np
import tempfile

st.title("DIP Project: Video Face Filter")
st.write("Upload a video to apply the 'DIP' forehead filter.")

# Load the DIP Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 1. Video File Uploader
video_file = st.file_uploader("Upload a video of your face", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    # Open the video using OpenCV
    cap = cv2.VideoCapture(tfile.name)
    st.write("Processing video... please wait.")

    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Placeholder for the processed video
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # DIP PROCESS:
        # A. Grayscale Conversion (Standard DIP preprocessing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # B. Feature Extraction (Detecting the Face)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # C. Spatial Transformation (Placing text on forehead)
        for (x, y, w, h) in faces:
            # Position calculation for forehead
            text_x = x + int(w * 0.2)
            text_y = y + int(h * 0.2)
            
            # Overlay "DIP" text
            cv2.putText(frame, "DIP", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame in the app
        st_frame.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    st.success("Video processing complete!")
