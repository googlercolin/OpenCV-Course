import cv2 as cv
import numpy as np
import time
import PoseModule as pm

cap = cv.VideoCapture('myTutorial/Pose Estimation/AITrainerMedia/bicep_curls.mp4')
detector = pm.poseDetector()
count = 0
direction = 0 # going up

prevTime = 0

while True:
    success, img = cap.read()

    # img = cv.imread('myTutorial/Pose Estimation/AITrainerMedia/pushup.jpeg')
    img = detector.findPose(img, False)

    lmList = detector.findPosition(img, False)
    if len(lmList):

        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)

        # Right Arm
        # detector.findAngle(img, 12, 14, 16)
        perc = np.interp(angle, (205, 310), (0, 100))
        bar = np.interp(angle, (205, 310), (650, 100))

        # Check for dumbbell curls
        color = (255, 0 , 255)
        if perc == 100:
            color = (0, 255, 0)
            if direction == 0: # going up
                count += 0.5
                direction = 1
        if perc == 0:
            # color = (0, 255, 0)
            if direction == 1: # going down
                count += 0.5
                direction = 0

        # Draw bar
        cv.rectangle(img, (900, 100), (975, 650), color, 3)
        cv.rectangle(img, (900, int(bar)), (975, 650), color, -1)
        cv.putText(img, f'{int(perc)}%', (900, 75), cv.FONT_HERSHEY_PLAIN, 4, color, 3)

        # Draw count
        cv.rectangle(img, (0, 450), (250, 720), (0, 255, 0), -1)
        cv.putText(img, f'{int(count)}', (50, 670), cv.FONT_HERSHEY_PLAIN, 15, (255, 0, 255), 25)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, f'{int(fps)}', (50, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 3)


    cv.imshow('Image', img)
    cv.waitKey(1)
