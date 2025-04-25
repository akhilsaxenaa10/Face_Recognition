import cv2

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
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Loop through each image file
for file in image_files:
    img = cv2.imread(file)
    if img is None:
        print(f"Couldn't load image: {file}")
        continue

    # Convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(imgGray, 1.1, 15)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Define region of interest for eye and smile detection
        roi_GrayImg = imgGray[y:y + h, x:x + w]
        roi_ColorImg = img[y:y + h, x:x + w]

        # Detect eyes
        eyes = eyes_cascade.detectMultiScale(roi_GrayImg, 1.8,9)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_ColorImg, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_GrayImg, 1.8, 8)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_ColorImg, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)

    # Show the final images
    cv2.imshow(f"Detected - {file.split('/')[-1]}", img)
    cv2.waitKey(0)
     
# Close all windows after processing
cv2.destroyAllWindows()
