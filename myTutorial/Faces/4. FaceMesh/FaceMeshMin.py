import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('myTutorial/Faces/4. FaceMesh/FaceMeshVideos/mesh5.mp4')

prevTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)

# def __init__(self,
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=False,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5)

drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_CONTOURS,
                                    drawSpec, drawSpec)
            for id, lm in enumerate(faceLandmarks.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv.imshow('Image', img)
    cv.waitKey(1)