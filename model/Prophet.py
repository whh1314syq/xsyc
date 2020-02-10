# -*- coding: utf-8 -*-
#from fbprophet import Prophet
import numpy as np
import pandas as pd
from fbprophet import Prophet
from datetime import date,timedelta
import matplotlib.pyplot as plt
from keras.layers import Dense,LSTM,Dropout
import tensorflow as tf
sales_df = pd.read_csv('../data/sales4.csv')

sales_df['y_orig'] = sales_df['y']

# log-transform y
sales_df['y'] = np.log(sales_df['y'])

sales_df['y_log']=sales_df['y'] 
sales_df['y']=sales_df['y_orig']


#make model
model = Prophet() 
model.fit(sales_df)

#future data
future_data = model.make_future_dataframe(periods=12,freq = 'm')

#predict

forecast_data = model.predict(future_data)
print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
model.plot(forecast_data)
model.plot_components(forecast_data)
print(forecast_data.columns)
#orecast_data_orig = forecast_data # make sure we save the original forecast data
#orecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
#orecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
#orecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
#odel.plot(forecast_data_orig)

def prophet_predict_fb(observed_data, x_name="ds", y_name="y", forecast_cnt=12, frep="m", file_name=""):
    """
    function that predict time series with library fbprophet
    :param observed_data: time series data(DataFrame format)
    (two columns, one is time in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format and the other is numeric data)
    :param x_name: x column name(time data), usually is DATE
    :param y_name: y column name(numeric data) e.g. HMD, MAX...
    :param forecast_cnt: how many point needed to be predicted
    :param frep: the frequency/period of prediction
    :param file_name:
    :return: None
    """

    def check_parameter_validity():
        if x_name not in observed_data.keys():
            raise KeyError("train_data doesn't have column named %s" % x_name)
        if y_name not in observed_data.keys():
            raise KeyError("train_data doesn't have column named %s" % y_name)

    try:
        check_parameter_validity()
    except KeyError as e:
        print("key error: %s" % str(e))
        return None

    observed_data = observed_data.rename(columns={x_name: "ds", y_name: "y"})

    observed_data["ds"] = pd.to_datetime(observed_data["ds"])
    observed_data["y"] = pd.to_numeric(observed_data["y"], downcast='float', errors='coerce')

    df2_pro = Prophet(changepoint_prior_scale=0.2)
    df2_pro.fit(observed_data)

    future_date = df2_pro.make_future_dataframe(periods=forecast_cnt, freq=frep)
    df2_forecast = df2_pro.predict(future_date)

    # register a datetime converter for matplotlib
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    if file_name:
        fig1 = df2_pro.plot(df2_forecast, xlabel=x_name, ylabel=y_name)
        fig1.show()
        fig1.savefig('./result/%s.png' % file_name)
        fig2 = df2_pro.plot_components(df2_forecast)
        fig2.show()
        fig2.savefig('./result/%s.png' % str(file_name + "1"))

    return df2_forecast
print(prophet_predict_fb(sales_df))