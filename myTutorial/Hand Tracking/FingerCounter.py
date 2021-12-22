import cv2 as cv
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'myTutorial/Hand Tracking/FingerImages'
myList = [f for f in os.listdir(folderPath) if not f.startswith('.')] # to not include hidden files starting with '.'
myList.sort()
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

prevTime = 0

detector = htm.handDetector(detectConf=0.8)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList):
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else: 
            fingers.append(0)
        
        # Other four fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else: 
                fingers.append(0)

        print(fingers)
        
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1] # if totalFingers = 0, we will find overlayList[-1] which is the last element in the list

        cv.rectangle(img, (0, 260), (150, 425), (0, 255, 0), -1)
        cv.putText(img, str(totalFingers), (25, 400), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 25)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, f'FPS {int(fps)}', (420, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow('Image', img)
    cv.waitKey(1)