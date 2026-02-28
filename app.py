import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 1. Page Setup
st.title("DIP Live Face Filter")
st.write("Click 'Start' to see the DIP filter on your forehead in real-time.")

# 2. Define the Processing Logic
class FaceFilterTransformer(VideoTransformerBase):
    def __init__(self):
        # Load the Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        # Convert the video frame into an OpenCV image (numpy array)
        img = frame.to_ndarray(format="bgr24")

        # DIP STEP: Grayscale conversion (Standard preprocessing)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # DIP STEP: Detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            # CALCULATE FOREHEAD POSITION
            # x + 20% of width to center it
            # y + 20% of height to bring it down from the top of the box
            text_pos = (x + int(w * 0.2), y + int(h * 0.2))

            # DRAW THE FILTER
            cv2.putText(img, "DIP", text_pos, 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
            
            # Draw the box around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return img

# 3. Launch the WebRTC Streamer
webrtc_streamer(key="face-filter", video_transformer_factory=FaceFilterTransformer)
