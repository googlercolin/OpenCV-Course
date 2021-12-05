import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype='uint8')
print(blank.shape)

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (blank.shape[0]//2, blank.shape[1]//2), blank.shape[0]//2, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# bitwise AND: Intersection

bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('BITWISE AND', bitwise_and)

# bitwise OR: Union
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('BITWISE OR', bitwise_or)

# bitwise XOR: Non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('BITWISE XOR', bitwise_xor)

# bitwise NOT: inverts the color
bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('Rectangle NOT', bitwise_not)

cv.waitKey(0)