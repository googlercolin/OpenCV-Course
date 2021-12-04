import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8') # (height, width, number of color channels); uint8 is basically an image 
cv.imshow('Blank', blank)

# 1. Paint the image a certain color
# blank[200:300, 300:400] = 0,255,0
# cv.imshow('Green', blank)

# 2. Draw a rectangle
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 0, 255), thickness=cv.FILLED) #alternatively, can specify thickness=-1
cv.imshow('Rectangle', blank)

# 3. Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (255, 0, 0), thickness=-1)
cv.imshow('Circle', blank)

# 4. Draw a line
cv.line(blank, (blank.shape[1]//2, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)

# 5. Write text on an image
cv.putText(blank, 'Hello', (200, blank.shape[0]), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), thickness=2)
cv.imshow('Text', blank)

# img = cv.imread('Resources/Photos/cat.jpg')
# cv.imshow('Cat', img)

cv.waitKey(0)