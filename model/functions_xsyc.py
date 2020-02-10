# -*- coding: utf-8 -*-
from model.stldecompose import decompose, forecast
from model.stldecompose.forecast_funcs import (naive,drift,  mean, seasonal_naive)
import pandas as pd
from model.arima.sales import predictSales
from model.ets.hotwinters import HoltWinters
from model.ets.xl_HoltWinters import holtWinters
def fun_python_stl(df,pred=4):
    df2=df.copy()
    df2['timestamp']=pd.to_datetime(df2['timestamp'])
    df2.set_index(["timestamp"], inplace=True)
    short_decomp = decompose(df2, period=12,lo_frac=1, lo_delta=0.01)
    fcast= forecast(short_decomp, steps=pred, fc_func=drift,seasonal=True)
    result_stl=fcast.values.reshape(-1)
    return result_stl
    
    
    
def fun_python_arima(df,pred=4):
    df2=df.copy()
    df2.set_index(["timestamp"],inplace=True)
    df2.columns=["销量"]
    result=predictSales(df2,pred,isVisiable=True)
    return result

def fun_python_ets(df,pred=4):
    tsA=df.iloc[:,1]
    
    HWmodel = HoltWinters(tsA, 12, pred,mtype='additive')
    HWmodel.GridSearch()
    result=HWmodel.best_pred
    return result

def fun_python_ets2(df,t=2,pred=4):
    tsA=df.iloc[:,1]
    result= holtWinters(tsA, t, 2,pred,mtype='additive')
    return result['predicted']
    