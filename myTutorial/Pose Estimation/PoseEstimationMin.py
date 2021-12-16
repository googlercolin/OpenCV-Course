import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Pose init
# def __init__(self,
#                static_image_mode=False, 
#                # if True, it will always detect based on the model, 
#                # if False, it will try to detect and only when the confidence is high then it will keep tracking
#                model_complexity=1,
#                smooth_landmarks=True,
#                enable_segmentation=False,
#                smooth_segmentation=True,
#                min_detection_confidence=0.5,
#                min_tracking_confidence=0.5)

cap = cv.VideoCapture('myTutorial/Pose Estimation/PoseVideos/man_running.mp4')
prevTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape 
            # use h, w to determine exact pixels from lm.x and lm.y
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img, (cx, cy), 10, (255, 0, 255), -1)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.imshow('Image', img)

    cv.waitKey(1)