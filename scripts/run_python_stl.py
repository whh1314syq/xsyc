# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:45:52 2019

@author: LY
"""
from matplotlib import pyplot
import pandas as pd
import numpy as np
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
from pyecharts import Line
import statsmodels.api as sm
import matplotlib.pyplot as plt
from model.stldecompose import decompose, forecast
from model.stldecompose.forecast_funcs import (naive,drift,  mean, seasonal_naive)


#df=pd.read_excel("sales.xls",index_col='日期',header=0)
df=pd.read_csv("../data/sales.csv",header=None)
#df=df.iloc[:,[0,5]]
df.columns=['timestamp','sales']
df2=df.copy()
df['timestamp']=pd.to_datetime(df['timestamp'])
df.sort_values('timestamp',inplace=True)
df.reset_index(inplace=True,drop=True)
df.set_index(["timestamp"], inplace=True)

months=23
columns_pre=df.index[-months:]
df_pre=pd.DataFrame(np.zeros((months,months)))
df_pre.columns=columns_pre
df_pre["add1"]=0
df_pre['add2']=0
df_pre["add3"]=0

for i in range(df.shape[0]-months,df.shape[0]):
    data=df[0:i]
    try:
        #result_arima=predictSales(data,4,isVisiable=True)
        short_decomp = decompose(data, period=12,lo_frac=1, lo_delta=0.01)
        fcast= forecast(short_decomp, steps=4, fc_func=drift,seasonal=True)
        result_stl=fcast.values.reshape(-1)
    except Exception as e:
        print("调用失败")
        result_stl=[0,0,0,0]

    #print(i,result_arima)
    
    begin_num=(i+months-df.shape[0])
    try:
        df_pre.iloc[begin_num,begin_num:begin_num+4]=result_stl
        #df_pre.iloc[begin_num,begin_num:begin_num+4]=[1,1,1,1]
    except Exception as e:
        print('报错')
    
    
df_plot=df_pre.iloc[:,3:-3]
list_pre1=[]
list_pre2=[]
list_pre3=[]
list_pre4=[]
for j in range(0,df_plot.shape[1]):
    list_pre1.append(df_plot.iloc[j,j])
    list_pre2.append(df_plot.iloc[j+1,j])
    list_pre3.append(df_plot.iloc[j+2,j])
    list_pre4.append(df_plot.iloc[j+3,j])

        
#date_all=df.index[df.shape[0]-months+3:]
date_all=df2.timestamp[df.shape[0]-months+3:]
sales_real= df['sales'][df.shape[0]-months+3:]

#date_pre=df_plot.columns
date_pre=df2.timestamp[df.shape[0]-months+3:]
sales1=list_pre1
sales2=list_pre2
sales3=list_pre3
sales4=list_pre4


#sales_mean=(sales_arima_r+ sales4)/2
    
date_all=list(map(str,date_all))
date_pre=list(map(str,date_pre))

line = Line("空调销售预测")
line.add("实际销量结果", date_all, sales_real,linestyle="--",line_color='black',line_width=2)

line.add("第一次预测结果", date_pre, sales1,line_color='green')
line.add("第二次预测结果", date_pre, sales2,line_color='green')
line.add("第三次预测结果", date_pre, sales3,line_color='green')
line.add("第四次预测结果", date_pre, sales4,line_color='green')
#line.add("r语言预测结果",date_pre,sales_arima_r,line_color='red')
#line.add("r与python第四次平均预测结果",date_pre,sales_mean)
#line.add("商家B", attr, v2, is_smooth=True,mark_line=["max", "average"])
line.render('../data/python_stl.html')

#print("r语言arima均差",np.mean(np.abs(sales_real-sales_arima_r)))
print("python语言stl均差",np.mean(np.abs(sales_real-sales4)))
#print("平均均差",np.mean(np.abs(sales_real-sales_mean)))
