import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian gradient method
lap = cv.Laplacian(gray, cv.CV_64F) # transitions from black to white is a positive slope, white to black is a negative slope
lap = np.uint8(np.absolute(lap)) # images can't have negative pixels so we take the absolute
cv.imshow('Laplacian', lap)

# Sobel gradient method: compute gradients in x- and y-directions
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Combined Sobel', combined_sobel)

# Canny edge detection: a multistage edge detector (one of the stages use Sobel)
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)