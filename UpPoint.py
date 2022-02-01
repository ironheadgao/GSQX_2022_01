import cv2
import numpy as np
import cv2
import numpy as np

img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG')
#select ROI function
cropped_image = img[60:90, 18:45]
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cropped_image

img = cropped_image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,160,apertureSize = 3)

lines = cv2.HoughLines(edges,1.5,np.pi/180,12)
theta1 = []
rho1=[]
theta2=[]
rho2=[]
for line in lines:
    for rho,theta in line:
        if theta >=-0.15 and theta <0.15 and rho > img.shape[0]/2-5 and rho < img.shape[0]/2+5 :
            theta1.append(theta)
            rho1.append(rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            print(theta,"V",rho)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        elif theta > (np.pi)/2-0.04 and theta <(np.pi)/2+0.04  :
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            print(theta,rho)
            if (y1+y2)/2 < img.shape[0]/2+5 and  (y1+y2)/2 > img.shape[0]/2-5:
                cv2.line(img, (x1, y1), (x2, y2), (255,0,  0), 1)
                theta2.append(theta)
                rho2.append(rho)
            else:
               pass
        else:
            pass



theta1 = np.mean(theta1)
rho1 = np.mean(rho1)
theta2 = np.mean(theta2)
rho2 = np.mean(rho2)
def hough_inter(theta1, rho1, theta2, rho2):
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    return np.linalg.lstsq(A, b)[0]

point = hough_inter(theta1, rho1, theta2, rho2)
point = (int(point[0]),int(point[1]))

img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG')
#select ROI function
cropped_image = img[60:90, 18:45]
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cropped_image

cv2.circle(img, point, 2,(0,0,  255), 2)
cv2.imshow("cropped", cv2.resize(img, (0, 0), fx=10.0, fy=10.0, interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()