import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("myTutorial/Faces/3. FaceDetectionAdvanced/FaceVideos/faces3.mp4")
prevTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence = 0.65)

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections: 
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection) # using mpDraw to draw the bounding boxes
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            boundingBoxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            boundingBox = int(boundingBoxC.xmin * iw), int(boundingBoxC.ymin * ih), \
                    int(boundingBoxC.width * iw), int(boundingBoxC.height * ih)
            cv.rectangle(img, boundingBox, (255, 0, 255), 2)
            cv.putText(img, f'Confidence: {int(detection.score[0]*100)}%', 
                        (boundingBox[0], boundingBox[1]-20), 
                        cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)


    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime
    cv.putText(img, f'FPS {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

    cv.imshow('Image', img)
    cv.waitKey(1)