import os
import cv2 as cv
import numpy as np

# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfeild', 'Madonna', 'Mindy Kaling']

p = []
for i in os.listdir(r'Resources/Faces/train'):
    p.append(i)
# print(p)

DIR = r'/Users/colinhong/OneDrive - Nanyang Technological University/OpenCV-Course/Resources/Faces/train' # r string is a raw string so all backslashes within the single quote are left

haar_cascade = cv.CascadeClassifier('/Users/colinhong/OneDrive - Nanyang Technological University/OpenCV-Course/myTutorial/Faces/1. FaceDetection/haar_face.xml')

features = [] # image array of the faces
labels = [] # labels for the features

def create_train():
    for person in p:
        path = os.path.join(DIR, person)
        label = p.index(person) # mapping person to an index to reduce memory space

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

# print(f'Length of features = {len(features)}')
# print(f'Length of labels = {len(labels)}')

print('Training done!')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml') # To save the trained model in a yml source file 
np.save('features.npy', features)
np.save('labels.npy', labels)