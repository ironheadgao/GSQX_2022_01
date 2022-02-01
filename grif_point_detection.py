import cv2
import numpy as np


img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG',1)

# convert image to gray
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#threshold gray image to b and w
ret,thresh2 = cv2.threshold(im_gray,120,255,cv2.THRESH_BINARY_INV)

# dilate and erode image
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(thresh2, kernel, iterations=2)

kernel = np.ones((10,10), np.uint8)
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

# detect corners
gray = np.float32(img_erosion)
dst = cv2.cornerHarris(gray,5,19,0.07)

dst = cv2.dilate(dst,None,iterations=2)

img[dst>0.01*dst.max()]=[0,0,255]

cv2.imwrite('dst.png', dst)
cv2.imwrite('img2.png', img)

cv2.imshow('hor', dst)
cv2.waitKey()
cv2.destroyAllWindows()