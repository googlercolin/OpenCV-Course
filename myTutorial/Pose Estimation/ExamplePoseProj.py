import cv2 as cv
import time
import PoseModule as pm

cap = cv.VideoCapture('myTutorial/Pose Estimation/PoseVideos/man_running.mp4')
prevTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)
    if len(lmList) == 0:
        break
    print(lmList[30]) # print only landmark 30 lists
    cv.circle(img, (lmList[30][1], lmList[30][2]), 10, (255, 0, 255), -1) # display the landmark 30 point in the video

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.imshow('Image', img)
    cv.waitKey(1)