import matplotlib.pyplot as plt
from sympy import symbols, solve
import numpy as np
from pandas import DataFrame
import cv2

data = DataFrame(np.zeros((600, 2)))
data.columns=['x','y']
x = np.arange(0, 600, 0.1)
def func(x):
    return 100*np.sqrt(x)

it=0
for i in range(0,300):
    data.iloc[it,0]=x[it]
    data.iloc[it, 1]=0.45*func(x[it])+390
    data.iloc[it+1, 0] = x[it]
    data.iloc[it+1, 1] = -0.45*func(x[it])+390
    it+=2

data = data.sort_values('y',ascending=False).reset_index(drop=True)
x= data.iloc[:,0]
y= data.iloc[:,1]
img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG')
for x_shift in [29,50,74,97,98,99,96,95,94,119,142,167,189,213,235,
                259,282,305,327,351,373,396,420,444,
                467,489,513,537,559,581,605,629,651,
                673,698,721,745,768,769,766,793,817,841,865,
                887,909,936,957,981,1005,1028,1051,
                1073,1097,1119,1143,1167,1191,1215,
                1239,1263,1287,1311,1332,1357,1381,
                1405,1427,1425,1429,1431,1432,1435,
                1450,1475,1498,1521,1545,1569,1591,
                1615,1639,1661,1685,1709,1733,1755,
                1778,1803,1827,1850,1873,1897,1921,
                1945,1968,1991,2016,2038,2063,2082,2080,2085,2086,2087,2088,2090,2091,
                2109,2132,2156,2178,2202,2225,2250,
                2273,2296,2320,2343,2371,2397,2420]:
    for i  in range(50,len(data)-55):
        cv2.line(img, (int(data.iloc[i, 0]) + x_shift, int(data.iloc[i, 1])),
                (int(data.iloc[i + 1, 0]) + x_shift, int(data.iloc[i + 1, 1])), [0, 255,0], 1)

cv2.imshow('hor', img)
cv2.waitKey()
cv2.destroyAllWindows()



cv2.imwrite('F:\\GSQX\\TempRecord\\curveling_recognize.JPG', img)