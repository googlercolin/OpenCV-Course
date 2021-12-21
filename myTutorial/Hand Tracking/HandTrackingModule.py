import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectConf = detectConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectConf, self.trackConf) # hands object only uses RGB imgs
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm) # lm shows the x, y, z ratios relative to the size of the img
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 15, (255, 0, 0), -1)
        return lmList

def main():

    cap = cv.VideoCapture(0)    
    prevTime = 0
    currTime = 0

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=True)
        if len(lmList) != 0: 
            print(lmList[4])

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
        cv.imshow("Image", img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()