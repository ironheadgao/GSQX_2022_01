import netCDF4 as nc
import cv2
import  numpy as  np
file = 'F:\\GSQX\\X61_178_19_112_6_2_2_11.nc'
dataset = nc.Dataset(file)
print(dataset.variables.keys())

image = cv2.imread("F:\\GSQX\\U528891953010102.JPG")


def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
        y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
        y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)    #点的坐标必须是元组，不能是列表。
    cv2.imshow("image-lines", image)

#统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line_detect_possible_demo",image)

src = cv2.imread('F:\\GSQX\\U528891953010102.JPG')
print(src.shape)
cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('input_image', src)
line_detection(src)
src = cv2.imread('F:\\GSQX\\U528891953010102.JPG') #调用上一个函数后，会把传入的src数组改变，所以调用下一个函数时，要重新读取图片
line_detect_possible_demo(src)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np



    frame  = cv2.imread('F:\\GSQX\\TempRecord\\1953\\U528891953010102.JPG')
    cv2.imshow('frame', frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold of blue in HSV space
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    cv2.waitKey(0)

cv2.destroyAllWindows()
cap.release()



img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\U528891953010102.JPG',0)





import numpy as np
from skimage.measure import approximate_polygon, find_contours

import cv2

img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\U528891953010102.JPG',0)
contours = find_contours(img, 0)

result_contour = np.zeros(img.shape + (3, ), np.uint8)
result_polygon1 = np.zeros(img.shape + (3, ), np.uint8)
result_polygon2 = np.zeros(img.shape + (3, ), np.uint8)

for contour in contours:
    print('Contour shape:', contour.shape)

    # reduce the number of lines by approximating polygons
    polygon1 = approximate_polygon(contour, tolerance=2.5)
    print('Polygon 1 shape:', polygon1.shape)

    # increase tolerance to further reduce number of lines
    polygon2 = approximate_polygon(contour, tolerance=15)
    print('Polygon 2 shape:', polygon2.shape)

    contour = contour.astype(np.int).tolist()
    polygon1 = polygon1.astype(np.int).tolist()
    polygon2 = polygon2.astype(np.int).tolist()

    # draw contour lines
    for idx, coords in enumerate(contour[:-1]):
        y1, x1, y2, x2 = coords + contour[idx + 1]
        result_contour = cv2.line(result_contour, (x1, y1), (x2, y2),
                                  (0, 255, 0), 1)
    # draw polygon 1 lines
    for idx, coords in enumerate(polygon1[:-1]):
        y1, x1, y2, x2 = coords + polygon1[idx + 1]
        result_polygon1 = cv2.line(result_polygon1, (x1, y1), (x2, y2),
                                   (0, 255, 0), 1)
    # draw polygon 2 lines
    for idx, coords in enumerate(polygon2[:-1]):
        y1, x1, y2, x2 = coords + polygon2[idx + 1]
        result_polygon2 = cv2.line(result_polygon2, (x1, y1), (x2, y2),
                                   (0, 255, 0), 1)

cv2.imwrite('contour_lines.png', result_contour)
cv2.imwrite('polygon1_lines.png', result_polygon1)
cv2.imwrite('polygon2_lines.png', result_polygon2)