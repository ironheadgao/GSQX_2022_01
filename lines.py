import numpy as np

import cv2

img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.line(img, (144,217), (154,225), (0, 255, 0), 2)
cv2.line(img, (154,225), (161,226), (0, 255, 0), 2)
cv2.imshow('hor', img)
cv2.waitKey()
cv2.destroyAllWindows()


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

    def VertexDetect(self):

        vertical_lines = self.VerticalLineDetect()
        horizontal_lines = self.HorizontalLineDetect()

        # 顶点列表
        vertex = []
        for v_line in vertical_lines:
            for h_line in horizontal_lines:
                vertex.append((v_line[0], h_line[1]))

        # print(vertex)

        # 绘制顶点
        for point in vertex:
            cv2.circle(self.image, point, 1, (255, 0, 0), 2)

        return vertex

    def CellDetect(self):
        vertical_lines = self.VerticalLineDetect()
        horizontal_lines = self.HorizontalLineDetect()

        # 顶点列表
        rects = []
        for i in range(0, len(vertical_lines) - 1, 2):
            for j in range(len(horizontal_lines) - 1):
                rects.append((vertical_lines[i][0], horizontal_lines[j][1], \
                              vertical_lines[i + 1][0], horizontal_lines[j + 1][1]))

        # print(rects)
        return rects