import cv2
import os

# Folder containing your images
image_folder = 'smiling_babies'

# Get all JPG files from the folder
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

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

    # Show the final image
    cv2.imshow(f"Detected - {os.path.basename(file)}", img)
    cv2.waitKey(0)

# Close all windows after processing
cv2.destroyAllWindows()
