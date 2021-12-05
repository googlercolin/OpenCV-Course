import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats 2.jpg')
cv.imshow('Cats', img)

# the dim of the mask MUST be the same as that of the img
blank = np.zeros(img.shape[:2], dtype='uint8')

circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)
rect = cv.rectangle(blank.copy(), (30,30), (370, 370), 255, -1)

weird_shape = cv.bitwise_and(circle, rect)

masked = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Weird Shape Mask', masked)

cv.waitKey(0)