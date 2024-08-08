import cv2
import numpy as np
import os
import threading
from tensorflow.keras.models import load_model

# Global variables
lock = threading.Lock()
shared_emotion = -1
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to find the mode of an array
def find_mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    mode_index = np.argmax(counts)
    return int(values[mode_index])

# Initialize parameters
queue_size = 20
data_queue = np.array([])
dq2 = np.array([])
q2size = 60
mode_final = -1
mode2 = -1
ms = -1  # Initialize the ms variable

# Load pre-trained models
model1 = load_model('facial_expression_model.h5')
model3 = load_model('fem.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    ms = (ms + 1) % 10

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        # Use alternating models for prediction
        if ms % 3 == 0 and ms < 7:
            emotion_prediction = model1.predict(roi_gray)
        else:
            emotion_prediction = model3.predict(roi_gray)
        emotion_index = np.argmax(emotion_prediction)

        # Update data queue and compute mode
        data_queue = np.append(data_queue, emotion_index)
        if len(data_queue) > queue_size:
            data_queue = data_queue[1:]
        mode_final = find_mode(data_queue)

        # Update second queue for smoothing
        dq2 = np.append(dq2, mode_final)
        if len(dq2) > q2size:
            dq2 = dq2[1:]
        mode2 = find_mode(dq2)

        # Display results
        if mode_final != -1:
            emotion = emotion_labels[mode_final]
            color = (0, 255, 0) if emotion == "Happy" else (255, 255, 0) if emotion == "Neutral" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame
    cv2.imshow("Webcam Capture", frame)

    # Update shared emotion variable
    if mode2 != -1:
        with lock:
            shared_emotion = mode2

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()