import math
import pandas as pd
import numpy as np
import itertools as it
from pandas import DataFrame
import datetime
from datetime import timedelta
names = locals()

filter_data = raw_data = pd.read_csv('F:\\TempData\\0P01_20211125_AND_awsall_20211125.csv')
A=['TAIR','QAIR_r2','_RT*PRE','NRT*PRE','WD_uv','WS_uv']
B=['TEM','RHU','PRE_1h','PRE_1h','WIN_D_Avg_2mi','WIN_S_Avg_2mi']

for i in range(0,6):
    names["A" + str(i) + '_value'] = filter_data[A[i]]
    names["B" + str(i) + '_value'] = filter_data[B[i]]
    if i==0:
        names["A" + str(i) + '_value'] = list(map(float, names["A" + str(i) + '_value'] - 273.15))
        names["B" + str(i) + '_value'] = list(map(float, names["B" + str(i) + '_value']))
        Sub_value = np.array(names["B" + str(i) + '_value']) - np.array(names["A" + str(i) + '_value'])
        raw_data[A[i] + "-" + B[i]] = Sub_value
    else:
        names["A" + str(i) + '_value'] = list(map(float, names["A" + str(i) + '_value']))
        names["B" + str(i) + '_value'] = list(map(float, names["B" + str(i) + '_value']))
        Sub_value = np.array(names["B" + str(i) + '_value']) - np.array(names["A" + str(i) + '_value'])
        raw_data[A[i]+"-"+B[i]] = Sub_value
raw_data.to_csv("E:\\atmosphere\\data\\0P01_20211125_AND_awsall_20211125.csv", mode='w', index=False,
                encoding='utf-8-sig')

"""
A0_value = filter_data['TAIR']
B0_value = filter_data['TEM']

A1_value = filter_data['QAIR_r2']
B1_value = filter_data['RHU']

A2_value = filter_data['_RT*PRE']
B2_value = filter_data['PRE_1h']

A3_value = filter_data['NRT*PRE']
B3_value = filter_data['PRE_1h']

A4_value = filter_data['WD_uv']
B4_value = filter_data['WIN_D_Avg_2mi']

A5_value = filter_data['WS_uv']
B5_value = filter_data['WIN_S_Avg_2mi']



A0_value = list(map(float, A0_value - 273.15))
B0_value = list(map(float, B0_value))
Sub_value = np.array(B0_value) - np.array(A0_value)
raw_data['TAIR-TEM'] = Sub_value
raw_data.to_csv("E:\\atmosphere\\data\\0P01_20211125_AND_awsall_20211125.csv", mode='w', index=False,
                encoding='utf-8-sig')




A1_value = list(map(float, A1_value))
B1_value = list(map(float, B1_value))
Sub1_value = np.array(B1_value) - np.array(A1_value)
raw_data['QAIR_r2-RHU'] = Sub1_value
raw_data.to_csv("E:\\atmosphere\\data\\0P01_20211125_AND_awsall_20211125.csv", mode='w', index=False,
                encoding='utf-8-sig')

A2_value = list(map(float, A2_value))
B2_value = list(map(float, B2_value))
Sub2_value = np.array(B2_value) - np.array(A2_value)
raw_data['_RT*PRE-PRE_1h'] = Sub2_value
raw_data.to_csv("E:\\atmosphere\\data\\0P01_20211125_AND_awsall_20211125.csv", mode='w', index=False,
                encoding='utf-8-sig')

A3_value = list(map(float, A3_value))
B3_value = list(map(float, B3_value))
Sub3_value = np.array(B3_value) - np.array(A3_value)
raw_data['NRT*PRE-PRE_1h'] = Sub3_value
raw_data.to_csv("E:\\atmosphere\\data\\0P01_20211125_AND_awsall_20211125.csv", mode='w', index=False,
                encoding='utf-8-sig')

A4_value = list(map(float, A4_value))
B4_value = list(map(float, B4_value))
Sub4_value = np.array(B4_value) - np.array(A4_value)
raw_data['WD_uv-WIN_D_Avg_2mi'] = Sub4_value
raw_data.to_csv("E:\\atmosphere\\data\\0P01_20211125_AND_awsall_20211125.csv", mode='w', index=False,
                encoding='utf-8-sig')

A5_value = list(map(float, A5_value))
B5_value = list(map(float, B5_value))
Sub5_value = np.array(B5_value) - np.array(A5_value)
raw_data['WS_uv-WIN_S_Avg_2mi'] = Sub5_value
raw_data.to_csv("E:\\atmosphere\\data\\0P01_20211125_AND_awsall_20211125.csv", mode='w', index=False,
                encoding='utf-8-sig')
"""
