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

result_python_arima=fun_python_arima(train_dataframe,12)

line = Line("空调销售预测", title_pos='right', 
            width=1400, height=700, title_color='red',title_text_size=10)

line.add("python_arima_year", list(range(len(result_python_arima))), result_python_arima,line_type='dashed',line_color='red')





line.render('../data/python_arima_year.html')

#print("r语言arima均差",np.mean(np.abs(sales_real-sales_arima_r)))
#print("python语言stl均差",np.mean(np.abs(sales_real-sales4)))
#print("平均均差",np.mean(np.abs(sales_real-sales_mean)))
