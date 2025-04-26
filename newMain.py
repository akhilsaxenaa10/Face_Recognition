import cv2
import streamlit as st
import numpy as np


image_files = [
    "smiling_babies/baby1.jpg",
    "smiling_babies/baby2.jpg",
    "smiling_babies/baby3.jpg",
    "smiling_babies/baby4.jpg",
    "smiling_babies/baby5.jpg",
    "smiling_babies/baby6.jpg",
    "smiling_babies/baby7.jpg"
]

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Loop through each image file
for file in image_files:
    img = cv2.imread(file)
    if img is None:
        st.error(f"Couldn't load image: {file}")
        continue

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(imgGray, 1.1, 15)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_GrayImg = imgGray[y:y + h, x:x + w]
        roi_ColorImg = img[y:y + h, x:x + w]

        eyes = eyes_cascade.detectMultiScale(roi_GrayImg, 1.8, 9)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_ColorImg, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        smiles = smile_cascade.detectMultiScale(roi_GrayImg, 1.8, 8)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_ColorImg, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show image in Streamlit
    st.image(img_rgb, caption=f"Detected - {file.split('/')[-1]}", use_column_width=True)
