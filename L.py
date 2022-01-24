import pandas as pd
import numpy as np
from pandas import DataFrame
names = locals()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#load data
raw_data = pd.read_csv('F:\\TempData\\0P01_20211125_AND_awsall_20211125.csv')
#load var name
A=['TAIR','QAIR_r2','_RT*PRE','NRT*PRE','WD_uv','WS_uv']
B=['TEM','RHU','PRE_1h','PRE_1h','WIN_D_Avg_2mi','WIN_S_Avg_2mi']
#manage absolute zero
raw_data[A[0]] = raw_data[A[0]] - 273.15
#manage 999999 float value
for i in range(0,6):
    raw_data = raw_data.drop(raw_data[raw_data[A[i]]==999999].index)
    raw_data = raw_data.drop(raw_data[raw_data[B[i]]==999999].index)
    print(i)
#COR,RMSE,ME,MAE;
para_data = DataFrame(np.zeros((1, 0)))
for i in range(0,6):
    # COR
    names["COR" + A[i] + '_'+B[i]] = np.corrcoef(raw_data[A[i]], raw_data[B[i]])[0,1]
    para_data["COR_" + A[i] + '_'+B[i]]=names["COR" + A[i] + '_'+B[i]]
    # RMS
    names["RMSE" + A[i] + '_' + B[i]] = mean_squared_error(raw_data[A[i]], raw_data[B[i]],squared=False)
    para_data["RMSE_" + A[i] + '_' + B[i]] = names["RMSE" + A[i] + '_' + B[i]]
    # ME
    names["ME" + A[i] + '_' + B[i]] = np.mean(raw_data[A[i]]-raw_data[B[i]])
    para_data["ME_" + A[i] + '_' + B[i]] = names["ME" + A[i] + '_' + B[i]]
    # MAE
    names["MAE" + A[i] + '_' + B[i]] = mean_absolute_error(raw_data[A[i]], raw_data[B[i]])
    para_data["MAE_" + A[i] + '_' + B[i]] = names["MAE" + A[i] + '_' + B[i]]
    print(i)
para_data.to_csv('F:\\TempData\\0P01_20211125_AND_awsall_parameters.csv')









