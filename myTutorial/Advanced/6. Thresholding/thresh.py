'''
Thresholding used in binarizing images. Anything above the threshold set = 1, anything below = 0
'''

import cv2 as cv

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) # threshold is the value passed i.e. 150, thresh is the thresholded (binarized) img
cv.imshow('Simple Threshold', thresh)

# Simple Inverse Thresholding
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) # threshold is the value passed i.e. 150, thresh is the thresholded (binarized) img
cv.imshow('Simple Inverse Threshold', thresh_inv)

# Adaptive Threshold: Using the blocksize (kernel), it finds the optimal thresholded value for a specific part and slides to the right and to the rest of the image
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3) # using Gaussian method for adaptive thresholding (slightly better for this case, but NOT one-size-fits-all)
# 11 is the blocksize (kernel), 
# C=3 is an integer subtracted from the mean to finetune the threshold
cv.imshow('Adaptive Threshold', adaptive_thresh)

# Adaptive Inverse Threshold: Using the blocksize (kernel), it finds the optimal thresholded value for a specific part and slides to the right and to the rest of the image
adaptive_thresh_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 9) 
# 11 is the blocksize (kernel), 
# C=3 is an integer subtracted from the mean to finetune the threshold
cv.imshow('Adaptive Inverse Threshold', adaptive_thresh_inv)

cv.waitKey(0)