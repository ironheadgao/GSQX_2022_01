from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import  numpy as  np
import cv2

img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\U528891953010102.JPG')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
r, g, b = cv2.split(img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()




#read the image
img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\U528891953010102.JPG')

#convert the BGR image to HSV colour space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#set the lower and upper bounds for the green hue
lower_green = np.array([5,50,50])
upper_green = np.array([15,255,255])

#create a mask for green colour using inRange function
mask = cv2.inRange(hsv, lower_green, upper_green)

#perform bitwise and on the original image arrays using the mask
res = cv2.bitwise_and(img, img, mask=mask)

#create resizable windows for displaying the images
cv2.namedWindow("res", cv2.WINDOW_NORMAL)
cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

#display the images
cv2.imshow("mask", mask)
cv2.imshow("hsv", hsv)
cv2.imshow("res", res)

if cv2.waitKey(0):
    cv2.destroyAllWindows()