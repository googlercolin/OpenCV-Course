import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, min_detection_confidence=0.65) -> None:
        self.minDetectionConf = min_detection_confidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConf)

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        boundingBoxes = []

        if self.results.detections: 
            for id, detection in enumerate(self.results.detections):
                boundingBoxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                boundingBox = int(boundingBoxC.xmin * iw), int(boundingBoxC.ymin * ih), \
                        int(boundingBoxC.width * iw), int(boundingBoxC.height * ih)
                boundingBoxes.append([id, boundingBox, detection.score])

                if draw: 
                    img = self.fancyDraw(img, boundingBox)
                    # cv.rectangle(img, boundingBox, (255, 0, 255), 2)
                    cv.putText(img, f'Confidence: {int(detection.score[0]*100)}%', 
                            (boundingBox[0], boundingBox[1]-20), 
                            cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        return img, boundingBoxes

    def fancyDraw(self, img, boundingBox, length = 30, thickness = 5, rectThickness = 1):
        x, y, w,h = boundingBox
        x1, y1 = x+w, y+h
        cv.rectangle(img, boundingBox, (255, 0, 255), rectThickness)

        # Top Left x,y
        cv.line(img, (x,y), (x+length, y), (255, 0, 255), thickness)
        cv.line(img, (x,y), (x, y+length), (255, 0, 255), thickness)

        # Top Right x1,y
        cv.line(img, (x1,y), (x1-length, y), (255, 0, 255), thickness)
        cv.line(img, (x1,y), (x1, y+length), (255, 0, 255), thickness)

        # Bottom Left x,y1
        cv.line(img, (x,y1), (x+length, y1), (255, 0, 255), thickness)
        cv.line(img, (x,y1), (x, y1-length), (255, 0, 255), thickness)

        # Bottom Right x1,y1
        cv.line(img, (x1,y1), (x1-length, y1), (255, 0, 255), thickness)
        cv.line(img, (x1,y1), (x1, y1-length), (255, 0, 255), thickness)
        
        return img


def main():
    cap = cv.VideoCapture("myTutorial/Faces/3. FaceDetectionAdvanced/FaceVideos/faces4.mp4")
    prevTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, boundingBoxes = detector.findFaces(img)

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime
        cv.putText(img, f'FPS {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()