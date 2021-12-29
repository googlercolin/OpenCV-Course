import cv2 as cv
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, modelComplexity=1, smoothLm=True, enableSeg=False, smooth=True, detectConf=0.5, trackConf=0.5) -> None:
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smoothLm = smoothLm
        self.enableSeg = enableSeg
        self.smooth = smooth
        self.detectConf = detectConf
        self.trackConf = trackConf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.modelComplexity,
            self.smoothLm,
            self.enableSeg,
            self.smooth,
            self.detectConf,
            self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []

        if self.results.pose_landmarks == None:
            return self.lmList

        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape 
            # use h, w to determine exact pixels from lm.x and lm.y
            cx, cy = int(lm.x*w), int(lm.y*h)
            self.lmList.append([id, cx, cy])
            if draw: 
                cv.circle(img, (cx, cy), 10, (255, 0, 255), -1)

        return self.lmList

    def findAngle(self, img, lm1, lm2, lm3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[lm1][1:]
        x2, y2 = self.lmList[lm2][1:]
        x3, y3 = self.lmList[lm3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2)-math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        
        # Draw
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv.circle(img, (x1, y1), 10, (255, 0, 255), -1)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), 2)
            cv.circle(img, (x2, y2), 10, (255, 0, 255), -1)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (255, 0, 255), -1)
            cv.circle(img, (x3, y3), 15, (255, 0, 255), 2)
            # cv.putText(img, str(int(angle)), (x2 + 50, y2 + 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        return angle

def main():
    cap = cv.VideoCapture('myTutorial/Pose Estimation/PoseVideos/man_running.mp4')
    prevTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, False)
        if len(lmList) == 0:
            break
        print(lmList[14]) # print only landmark 14 lists
        cv.circle(img, (lmList[14][1], lmList[14][2]), 10, (255, 0, 255), -1) # display the landmark 14 point in the video

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()