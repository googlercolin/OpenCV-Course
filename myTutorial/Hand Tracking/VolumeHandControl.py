import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript as oss

###############################
wCam, hCam = 640, 480
###############################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0

detector = htm.handDetector(detectConf=0.7)

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    vol = 0
    volBar = 400
    lmList = detector.findPosition(img, draw=False)
    if len(lmList):
        # we need the value 4 for the THUMB_TIP and 8 for INDEX_FINGER_TIP
        x_thumb, y_thumb = lmList[4][1], lmList[4][2]
        x_index, y_index = lmList[8][1], lmList[8][2]

        cv.circle(img, (x_thumb, y_thumb), 10, (255, 0, 255), -1)
        cv.circle(img, (x_index, y_index), 10, (255, 0, 255), -1)
        cv.line(img, (x_thumb, y_thumb), (x_index, y_index), (255, 150, 255), 2)
        cx, cy = (x_index+x_thumb)//2, (y_index+y_thumb)//2
        cv.circle(img, (cx, cy), 10, (255, 0, 255), -1)

        length = math.hypot(x_thumb-x_index, y_thumb-y_index)

        # Convert the range of the length ([50, 250]) to volume range ([0, 100])
        vol = np.interp(length, [50, 250], [0, 100])
        volBar = np.interp(length, [50, 250], [400, 150])
        print(vol)
        oss.osascript("set volume output volume {}".format(vol))

        if length<50:
            cv.circle(img, (cx, cy), 10, (0, 255, 0), -1)

    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), -1)
    cv.putText(img, f'{int(vol)}%', (40, 450), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, f'FPS: {int(fps)}', (40, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv.imshow('Image', img)
    
    cv.waitKey(1)