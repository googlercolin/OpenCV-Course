import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('/Users/colinhong/OneDrive - Nanyang Technological University/OpenCV-Course/myTutorial/Faces/1. FaceDetection/haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfeild', 'Madonna', 'Mindy Kaling']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'/Users/colinhong/OneDrive - Nanyang Technological University/OpenCV-Course/Resources/Faces/val/elton_john/4.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with confidence {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)