import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

st.title("DIP Project: Face Filter")

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # 1. Convert to grayscale (DIP Pre-processing)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # 3. Draw "DIP" on the frame
    for (x, y, w, h) in faces:
        cv2.putText(img, "DIP", (x, y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="filter", video_frame_callback=video_frame_callback)
