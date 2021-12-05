import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)

# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dim = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dim)

# -x --> left
# -y --> up
# x --> right
# y --> down

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None: 
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, scale=1.0)
    dim = (width, height)

    return cv.warpAffine(img, rotMat, dim)

rotated = rotate(img, 90)
cv.imshow('Rotated', rotated)

rotated_rotated = rotate(rotated, 45)
cv.imshow('Rotated Rotated', rotated_rotated)

# Resizing
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

# Flipping
flipped = cv.flip(img, -1) # 0 is flip vert, 1 is flip hori, -1 is flip both
cv.imshow('Flipped on both axes', flipped)

# Cropping
cropped = img[200:400, 30:100] # basically image array slicing
cv.imshow("Cropped", cropped)

cv.waitKey(0)