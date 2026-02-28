import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Define the Image Processing Class
class FaceFilterTransformer(VideoTransformerBase):
    def __init__(self):
        # Load the pre-trained Haar Cascade model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        # Convert the frame to a format OpenCV understands (BGR)
        img = frame.to_ndarray(format="bgr24")

        # DIP Step: Grayscale conversion to simplify feature detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # DIP Step: Object Detection (Face)
        # scaleFactor=1.1, minNeighbors=5 are standard DIP tuning parameters
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            # DIP Step: Spatial Overlay
            # Writing "DIP" text relative to the face coordinates (x, y)
            cv2.putText(img, 'DIP', (x + int(w/4), y - 20), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
            
            # Optional: Draw a bounding box for clarity
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img

# 2. Streamlit Web Interface
st.title("👨‍💻 DIP Real-Time Face Filter")
st.subheader("Face Detection & Dynamic Text Overlay")

webrtc_streamer(key="filter", video_transformer_factory=FaceFilterTransformer)
