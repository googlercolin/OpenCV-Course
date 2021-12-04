import cv2 as cv

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur: reduce the noise
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT) # (7, 7) is the kernel size, we use an odd number
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edge Original', canny)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edge Blur', canny)

# Dilating the image
dilated = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow('Dilated', dilated)

# Eroding; opp of dilation
eroded = cv.erode(dilated, (3, 3), iterations=1)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC) # CUBIC and LINEAR is usually used for scaling up images; CUBIC is the slowest but sharper. AREA is used for shrinking.
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 40:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)