import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=4, refineLm=False, minDetectConf=0.5, minTrackConf=0.5) -> None:
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLm = refineLm
        self.minDetectConf = minDetectConf
        self.minTrackConf = minTrackConf

        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.mediapipe.python.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLm, self.minDetectConf, self.minTrackConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLandmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLandmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    cv.putText(img, str(id), (x,y), cv.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 255), 1)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv.VideoCapture('myTutorial/Faces/4. FaceMesh/FaceMeshVideos/mesh6.mp4')
    prevTime = 0

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, False)
        if len(faces)!=0:
            print(len(faces))

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv.putText(img, f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()