import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # hands object only uses RGB imgs
mpDraw= mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                # print(id, lm) # lm shows the x, y, z ratios relative to the size of the img
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # draw circle at landmark id 0
                if (id == 0):
                    cv.circle(img, (cx,cy), 15, (255, 0, 0), -1)
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
    cv.imshow("Image", img)
    cv.waitKey(1)
