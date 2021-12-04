import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blank = np.zeros(img.shape, dtype='uint8')

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Using canny (this is generally preferred as thresholding uses simple binarizing)
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

# Using threshold: threshold binarizes the image (simple)
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) # Anything less than the threshold set i.e. 125 will be set to black
# cv.imshow('Threshold', thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) 
# mode cv.RETR_TREE for hierarchical contours, RETR_EXTERNAL for ext contours, RETR_LIST for all contours
# method is the contour approx method; cv.CHAIN_APPROX_NONE just returns all the contours, SIMPLE compresses
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), thickness=1)
cv.imshow('Contours drawn', blank)


cv.waitKey(0)