import cv2 as cv

img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat', img)

#rescaleFrame will work for images, videos and live videos
def rescaleFrame(frame, scale = 0.2):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_img = rescaleFrame(img)
cv.imshow('Resized Cat', resized_img)

# changeRes only works for live video: external cam or webcam
def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)

# Reading Videos
capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break;

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)