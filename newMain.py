import streamlit as st
import cv2
import numpy as np
import os

# Title
st.title("Face and Features Recognition System")

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Local image files
local_images = [
    "DataSet/1.jpg",
    "DataSet/2.jpg",
    "DataSet/3.jpg",
    "DataSet/4.jpg",
    "DataSet/5.jpg",
    "DataSet/6.jpg",
    "DataSet/7.jpg",
    "DataSet/8.jpg"
]

# Function to detect face, eyes, and smiles
def detect_features(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=15)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, 'Face', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        roi_gray = imgGray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=14)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            cv2.putText(roi_color, 'Eye', (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.6, minNeighbors=22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            cv2.putText(roi_color, 'Smile', (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return img

# File uploader
uploaded_file = st.file_uploader("Upload an image for face detection", type=["jpg", "jpeg", "png"])

# Process and display the uploaded image
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_img = cv2.imdecode(file_bytes, 1)

    if uploaded_img is not None:
        processed_uploaded_img = detect_features(uploaded_img.copy())

        # Display the uploaded image first
        st.subheader("Uploaded Image with Detected Features")
        st.image(cv2.cvtColor(processed_uploaded_img, cv2.COLOR_BGR2RGB), use_container_width=True)

# Divider
st.markdown("---")

# Display local sample images
st.subheader("Sample Images with Detected Features")

for file in local_images:
    if os.path.exists(file):
        img = cv2.imread(file)
        if img is not None:
            processed_img = detect_features(img.copy())
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=file.split("/")[-1], use_container_width=True)
    else:
        st.error(f"Image file not found: {file}")
