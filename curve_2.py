import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, solve
import math
import pandas as pd
import numpy as np
import itertools as it
from pandas import DataFrame
import datetime
from datetime import timedelta
names = locals()
import cv2

data = DataFrame(np.zeros((600, 2)))
data.columns=['x','y']
x = np.arange(0, 60, 0.1)
def func(x):
    return 100*np.sqrt(x)

it=0
for i in range(0,300):
    data.iloc[it,0]=x[it]
    data.iloc[it, 1]=func(x[it])+400
    data.iloc[it+1, 0] = x[it]
    data.iloc[it+1, 1] = -func(x[it])+400
    it+=2

data = data.sort_values('y',ascending=False)
x= data.iloc[:,0]
y= data.iloc[:,1]
plt.plot(x, y)
plt.show()

img = cv2.imread('F:\\GSQX\\TempRecord\\1953\\jpg\\U528891953020102.JPG')
for i  in range(0,len(data)-1):
    point1 = (data.iloc[i,0], data.iloc[i,1])
    point2 = (data.iloc[i+1,0], data.iloc[i+1,1])
    cv2.line(img, (int(data.iloc[i,0]), int(data.iloc[i,1])), (int(data.iloc[i+1,0]),int(data.iloc[i+1,1])), [0, 255, 0], 2)
cv2.imshow('hor', img)
cv2.waitKey()
cv2.destroyAllWindows()




data = data.sort_values('y',ascending=False)
x= data.iloc[:,0]
y= data.iloc[:,1]
plt.plot(x, y)
plt.show()