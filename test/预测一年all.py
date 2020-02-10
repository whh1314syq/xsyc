# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from model.stldecompose import decompose, forecast
from model.stldecompose.forecast_funcs import (naive,drift,  mean, seasonal_naive)
from pyecharts import Line
from model.functions_xsyc import fun_python_stl
from model.functions_xsyc import fun_python_arima
from model.functions_xsyc import fun_python_ets
from model.functions_xsyc import fun_python_ets2
from rpy2.robjects import r, pandas2ri

df=pd.read_csv("../data/sales3.csv",header=None)
#df=df.iloc[:,[0,5]]
df.columns=['timestamp','sales']
df2=df.copy()
#df['timestamp']=pd.to_datetime(df['timestamp'])
df.sort_values('timestamp',inplace=True)
df.reset_index(inplace=True,drop=True)

train_dataframe=df.iloc[0:(df.shape[0]-12),:]

result_python_ets=result_python_ets=fun_python_ets2(train_dataframe,12,12)
result_python_arima=fun_python_arima(train_dataframe,12)
result_python_stl=fun_python_stl(train_dataframe,12)
 

train_df=pandas2ri.py2ri(train_dataframe)
result_r_arima=np.round(list(r.fun_arima(train_df,12)[0]),2)
result_r_ets=np.round(list(r.fun_ets(train_df,12)[0]),2)
result_r_stl=np.round(list(r.fun_stl(train_df,12)[0]),2)
   
result_real=df.sales[(df.shape[0]-12):].values.reshape(-1)
line = Line("空调销售预测", title_pos='right', 
            width=1400, height=700, title_color='red',title_text_size=10)
line.add("真实销量", list(range(12)), result_real,line_color='black',line_width=2)
line.add("python_stl_year", list(range(12)), result_python_stl,line_type='dashed',line_color='red')
line.add("python_ets_year", list(range(12)), result_python_ets,line_type='dashed',line_color='red')
line.add("python_arima_year", list(range(12)), result_python_arima,line_type='dashed',line_color='red')

line.add("r_stl_year", list(range(12)), result_r_stl,line_type='dashed',line_color='green')
line.add("r_ets_year", list(range(12)), result_r_ets,line_type='dashed',line_color='green')
line.add("r_arima_year", list(range(12)), result_r_arima,line_type='dashed',line_color='green')




line.render('../data/python_r_all_year.html')

#print("r语言arima均差",np.mean(np.abs(sales_real-sales_arima_r)))
#print("python语言stl均差",np.mean(np.abs(sales_real-sales4)))
#print("平均均差",np.mean(np.abs(sales_real-sales_mean)))
