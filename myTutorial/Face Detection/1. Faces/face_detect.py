import cv2 as cv

img = cv.imread('Resources/Photos/group 2.jpg')
cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

# Face detection: A classification method
# Using Haar Cascades: not the most effective in face detection, but easy and popular

haar_cascade = cv.CascadeClassifier('myTutorial/Face Detection/1. Faces/haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6) # returns the rect coordinates of that face as a list
#minNeighbors is the number of neighbors a rectangle should have to be called a face
#by reduce minNeighbors, we increase sensitivity of the haar cascades to noise (a trade-off!)

print(f'Number of faces found = {len(faces_rect)}')

# loop over image and draw a rect over detected faces
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)