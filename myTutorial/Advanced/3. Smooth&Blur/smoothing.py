import cv2 as cv

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Averaging: compute the center value of a pixel of the image using the average value of the surrounding pixel intensities
average = cv.blur(img, (3,3)) # higher the kernal, the blurrer it is
cv.imshow('Average', average)

# Gaussian blur: assigns weights to the surrounding pixels, and the average of the products of the weights gives the value for the center
# Gives a more natural blur
gauss = cv.GaussianBlur(img, (3,3), 0) # the third argument is SigmaX (std deviation in the X direction)
cv.imshow('Gaussian', gauss)

# Median blur: more effective in reducing noise compared to averaging and Gaussian sometimes. 
# Good for remiving salt and pepper noise
median = cv.medianBlur(img, 3) # we usually use a lower kernal size for Median blur
cv.imshow('Median', median)

# Bilateral blur: retains the edges of the image
bilateral = cv.bilateralFilter(img, 15, 55, 55) 
# second argument is diameter, 
# third is sigmaColor (number of colors in the neigborhood which will be considered for blur calc, 
# forth is spaceSigma; higher values means that pixels further out from the central pixel will influence the blur calc
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)