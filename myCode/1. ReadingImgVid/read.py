import cv2 as cv

# Reading Images

img = cv.imread('Resources/Photos/cat_large.jpg')
cv.imshow('Cat', img)

# Reading Videos
# capture = cv.VideoCapture('Resources/Videos/dog.mp4')

# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break;

# capture.release()
# cv.destroyAllWindows()

cv.waitKey(0) # waitKey(0) waits for an infinite amount of time before a keyboard key is pressed