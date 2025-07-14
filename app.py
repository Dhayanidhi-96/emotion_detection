import cv2
import numpy as np
from tensorflow.keras.models import load_model

#Load model and Haar Cascade
model = load_model(r"D:\emotion_detection\models\emotion_model.h5")

face_cascade = cv2.CascadeClassifier(r"D:\emotion_detection\haarcascade\haarcascade_frontalface_default.xml")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad","Surprise", "Neutral"]

#start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret :
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis = -1)
        roi  = np.expand_dims(roi, axis = 0)

        #Predict emotion
        preds = model.predict(roi)
        label = emotions[np.argmax(preds)]

        #Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12),2)
    
    cv2.imshow("Real - time Emotion Deteection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
