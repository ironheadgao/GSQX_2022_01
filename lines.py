import numpy as np

import cv2

img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.line(img, (144,217), (154,225), (0, 255, 0), 2)
#a = np.array([(375, 193), (364, 113), (277, 20), (271, 16), (52, 106), (133, 266), (289, 296), (372, 282)])
data_n = data.to_numpy()

cv2.drawContours(img, data_n, 0, (255,255,255), 2)
cv2.imshow('hor', img)
cv2.waitKey()
cv2.destroyAllWindows()
#cv2.imwrite('F:\\GSQX\\TempRecord\\locstion_recognize.JPG', hsv)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#HorizontalLineDetect(self):

        # 图像二值化
        ret, thresh1 = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
        # 进行两次中值滤波
        blur = cv2.medianBlur(thresh1, 3)  # 模板大小3*3
        blur = cv2.medianBlur(blur, 3)  # 模板大小3*3

        h=img.shape[0]
        w =img.shape[1]

        # 横向直线列表
        horizontal_lines = []
        for i in range(h - 1):
            # 找到两条记录的分隔线段，以相邻两行的平均像素差大于120为标准
            if abs(np.mean(blur[i, :]) - np.mean(blur[i + 1, :])) > 250:
                # 在图像上绘制线段
                horizontal_lines.append([0, i, w, i])
                cv2.line(img, (0, i), (w, i), (255,255,255), 1)

        horizontal_lines = horizontal_lines[1:]
        print(horizontal_lines)
        cv2.imshow('hor', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def VerticalLineDetect(self):
        # Canny边缘检测
        img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG')
        edges = cv2.Canny(img_gray, 30, 240)

        # Hough直线检测
        minLineLength = 1000
        maxLineGap = 5
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap).tolist()
        #lines.append([[13, 937, 13, 102]])
        #lines.append([[756, 937, 756, 102]])
        sorted_lines = sorted(lines, key=lambda x: x[0])

        # 纵向直线列表
        vertical_lines = []
        for line in sorted_lines:
            for x1, y1, x2, y2 in line:
                # 在图片上绘制纵向直线
                if x1 == x2:
                    print(line)
                    vertical_lines.append((x1, y1, x2, y2))
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('ver', img)
        cv2.waitKey()
        cv2.destroyAllWindows()


for i  in range(0,len(data)-1):
    point1 = (data.iloc[i,0], data.iloc[i,1])
    point2 = (data.iloc[i+1,0], data.iloc[i+1,1])
    cv2.line(img, (int(data.iloc[i,0]), int(data.iloc[i,1])), (int(data.iloc[i+1,0]),int(data.iloc[i+1,1])), [0, 255, 0], 2)
cv2.imshow('hor', img)
cv2.waitKey()
cv2.destroyAllWindows()
