# -*- coding: utf-8 -*-

from matplotlib import pyplot
import pandas as pd
import numpy as np
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
from pyecharts import Line
pandas2ri.activate()
from model.functions_xsyc import fun_python_stl
from model.functions_xsyc import fun_python_arima
from model.functions_xsyc import fun_python_ets
from model.functions_xsyc import fun_python_ets2

df=pd.read_csv("../data/sales3.csv",header=None)
#df=df.iloc[:,[0,5]]
df.columns=['timestamp','sales']
df.sort_values('timestamp',inplace=True)
df.reset_index(inplace=True,drop=True)

r=robjects.r
r.source("../model/ch_functions_test4.R")

months=15
columns_pre=df.timestamp.values[-months:]
df_pre=pd.DataFrame(np.zeros((months,months)))
df_pre.columns=columns_pre
df_pre["add1"]=0
df_pre['add2']=0
df_pre["add3"]=0

df_pre_r_arima=df_pre.copy()
df_pre_r_ets=df_pre.copy()
df_pre_r_stl=df_pre.copy()
df_pre_python_arima=df_pre.copy()
df_pre_python_ets=df_pre.copy()
df_pre_python_stl=df_pre.copy()

#df_pre_
for i in range(df.shape[0]-months,df.shape[0]):
    print("开始第i轮:",i)
    train_dataframe=df.iloc[0:i,:]
    
    train_df=pandas2ri.py2ri(train_dataframe)
    #result_arima=r.fun_arima(train_df)
    #result_arima2=np.round(list(result_arima[0]),2)
    result_r_arima=np.round(list(r.fun_arima(train_df)[0]),2)
    result_r_ets=np.round(list(r.fun_ets(train_df)[0]),2)
    result_r_stl=np.round(list(r.fun_stl(train_df)[0]),2)
    try:
        result_python_arima=fun_python_arima(train_dataframe)
    except:
        result_python_arima=[0,0,0,0]
        
    #
    result_python_ets=fun_python_ets2(train_dataframe)
    
    result_python_stl=fun_python_stl(train_dataframe)
    #result_python_stl=result_python_ets
    
    begin_num=(i+months-df.shape[0])
    try:
        df_pre_r_arima.iloc[begin_num,begin_num:begin_num+4]=result_r_arima
        df_pre_r_ets.iloc[begin_num,begin_num:begin_num+4]=result_r_ets
        df_pre_r_stl.iloc[begin_num,begin_num:begin_num+4]=result_r_stl
        
        df_pre_python_arima.iloc[begin_num,begin_num:begin_num+4]=result_python_arima
        df_pre_python_ets.iloc[begin_num,begin_num:begin_num+4]=result_python_ets
        df_pre_python_stl.iloc[begin_num,begin_num:begin_num+4]=result_python_stl
        
        #df_pre.iloc[begin_num,begin_num:begin_num+4]=[1,1,1,1]
    except Exception as e:
        print('报错')
    
#r arima数据    
df_plot_r_arima=df_pre_r_arima.iloc[:,3:-3]
#list_pre1_r_arima=[]
#list_pre2_r_arima=[]
#list_pre3_r_arima=[]
list_pre4_r_arima=[]
for j in range(0,df_plot_r_arima.shape[1]):
    #list_pre1_r_arima.append(df_plot_r_arima.iloc[j,j])
    #list_pre2_r_arima.append(df_plot_r_arima.iloc[j+1,j])
    #list_pre3_r_arima.append(df_plot_r_arima.iloc[j+2,j])
    list_pre4_r_arima.append(df_plot_r_arima.iloc[j+3,j])

        
date_all=df.timestamp[df.shape[0]-months+3:]
sales_real= df.sales[df.shape[0]-months+3:]

date_pre=df_plot_r_arima.columns
#sales1_arima=list_pre1_r_arima
#sales2_arima=list_pre2_r_arima
#sales3_arima=list_pre3_r_arima
sales4_r_arima=list_pre4_r_arima
##r stl数据
df_plot_r_stl=df_pre_r_stl.iloc[:,3:-3]
list_pre4_r_stl=[]
for j in range(0,df_plot_r_stl.shape[1]):
    
    list_pre4_r_stl.append(df_plot_r_stl.iloc[j+3,j])

sales4_r_stl=list_pre4_r_stl

##r ets
df_plot_r_ets=df_pre_r_ets.iloc[:,3:-3]
list_pre4_r_ets=[]
for j in range(0,df_plot_r_ets.shape[1]):
    
    list_pre4_r_ets.append(df_plot_r_ets.iloc[j+3,j])

sales4_r_ets=list_pre4_r_ets


##python arima 
df_plot_python_arima=df_pre_python_arima.iloc[:,3:-3]
list_pre4_python_arima=[]
for j in range(0,df_plot_python_arima.shape[1]):
    
    list_pre4_python_arima.append(df_plot_python_arima.iloc[j+3,j])

sales4_python_arima=list_pre4_python_arima
##python  ets
df_plot_python_ets=df_pre_python_ets.iloc[:,3:-3]
list_pre4_python_ets=[]
for j in range(0,df_plot_python_ets.shape[1]):
    
    list_pre4_python_ets.append(df_plot_python_ets.iloc[j+3,j])

sales4_python_ets=list_pre4_python_ets


##python stl
df_plot_python_stl=df_pre_python_stl.iloc[:,3:-3]
list_pre4_python_stl=[]
for j in range(0,df_plot_python_stl.shape[1]):
    
    list_pre4_python_stl.append(df_plot_python_stl.iloc[j+3,j])

sales4_python_stl=list_pre4_python_stl

##除去效果不好的python ets ，其他5种进行平均
sales_mean=(np.array(sales4_r_arima)+np.array(sales4_r_stl)+np.array(sales4_r_ets)+np.array(sales4_python_arima)+np.array(sales4_python_stl)+np.array(sales4_python_ets))/6

#line = Line("空调销售预测", title_color='red', title_pos='bottom')
#line = Line( title_color='red', title_pos='left')
line=Line("空调销售预测", title_pos='right', 
            width=1400, height=700, title_color='red',title_text_size=10)

line.add("实际销量结果", date_all, sales_real,line_color='black',line_width=2)

#line.add("第一次预测结果", date_pre, sales1,line_type='dashed',line_color='green')
#line.add("第二次预测结果", date_pre, sales2,line_type='dashed',line_color='green')
#line.add("第三次预测结果", date_pre, sales3,line_type='dashed',line_color='green')
line.add("r_arima第四次预测结果", date_pre, sales4_r_arima,line_type='dashed',line_color='green')
line.add("r_stl第四次预测结果", date_pre, sales4_r_stl,line_type='dashed',line_color='green')
line.add("r_ets第四次预测结果", date_pre, sales4_r_ets,line_type='dashed',line_color='green')
line.add("python_ets第四次预测结果", date_pre, sales4_python_ets,line_type='dashed',line_color='red')
line.add("python_arima第四次预测结果", date_pre, sales4_python_arima,line_type='dashed',line_color='red')
line.add("python_stl第四次预测结果", date_pre, sales4_python_stl,line_type='dashed',line_color='red')
line.add("六种算法平均预测结果", date_pre, sales_mean,line_type='dashed',line_color='blue',line_width=2)
#line.add("python版arima预测结果",date_pre,sales_arima_python,line_type='dashed',line_color='red',line_width=2)
#line.add("商家B", attr, v2, is_smooth=True,mark_line=["max", "average"])
line.render('../data/all.html')

print("r_stl精度",np.mean(abs(sales4_r_stl-sales_real)))
print("加权精度",np.mean(abs(sales_mean-sales_real)))