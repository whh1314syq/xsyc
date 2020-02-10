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
pandas2ri.activate()
df=pd.read_csv("../data/sales.csv",header=None)
#df=df.iloc[:,[0,5]]
df.columns=['timestamp','sales']
df.sort_values('timestamp',inplace=True)
df.reset_index(inplace=True,drop=True)
pyplot.plot(df.timestamp, df.sales)
pyplot.show()
r_dataframe = pandas2ri.py2ri(df)
r=robjects.r
r.source("../model/ch_functions_test3.R")


months=23
columns_pre=df.timestamp.values[-months:]
df_pre=pd.DataFrame(np.zeros((months,months)))
df_pre.columns=columns_pre
df_pre["add1"]=0
df_pre['add2']=0
df_pre["add3"]=0

for i in range(df.shape[0]-months,df.shape[0]):
    train_dataframe=df.iloc[0:i,:]
    
    train_df=pandas2ri.py2ri(train_dataframe)
    result_ets=r.fun_ets(train_df)
    result_ets2=np.round(list(result_ets[0]),2)
    print(i,result_ets2)
    
    begin_num=(i+months-df.shape[0])
    try:
        df_pre.iloc[begin_num,begin_num:begin_num+4]=result_ets2
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

        
date_all=df.timestamp[df.shape[0]-months+3:]
sales_real= df.sales[df.shape[0]-months+3:]

date_pre=df_plot.columns
sales1=list_pre1
sales2=list_pre2
sales3=list_pre3
sales4=list_pre4
line = Line("空调销售预测")
line.add("实际销量结果", date_all, sales_real,linestyle="--")

line.add("第一次预测结果", date_pre, sales1)
line.add("第二次预测结果", date_pre, sales2)
line.add("第三次预测结果", date_pre, sales3)
line.add("第四次预测结果", date_pre, sales4)
#line.add("商家B", attr, v2, is_smooth=True,mark_line=["max", "average"])
line.render('../data/r_ets.html')
print(np.mean(abs(sales_real-sales4)))