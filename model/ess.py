# -*- coding: utf-8 -*-
# imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import Holt

df = pd.read_csv('E:/xsyc/xsyc/data/xsyc.csv')
data = df.set_index(pd.DatetimeIndex(start='1/31/2016', end='1/1/2020', freq='M'))
print(data)


X = data.values
train_size = int(len(X) * 0.75)
train, test = X[0:train_size], X[train_size:len(X)]

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

# then plotting it using different colors
plt.plot(train)
plt.plot([None for i in train] + [x for x in test])
plt.title("原始数据趋势图")
plt.show()


# made train_df and test_df (the latter to be used later)
#train_df = data[0:33]
#test_df = data[33:48]

train_df, test_df = data[0:train_size], data[train_size:len(X)]

series = data
result_a = seasonal_decompose(series, model='additive')
result_a.plot()
plt.title("累加效果图")
plt.show()

series = data
result_m = seasonal_decompose(series, model='multiplicative')
result_m.plot()
plt.title("累乘效果图")
plt.show()


# what will the forecast look like? a level line 
# where it will be positioned is dependent on alpha - change alpha and see the line move up or down


y_hat_avg = test_df.copy()
fit1 = SimpleExpSmoothing(np.asarray(train_df['sales'])).fit(smoothing_level=0.3, optimized=False)
y_hat_avg['SES'] = fit1.forecast(len(test_df))
plt.figure(figsize=(18,6))
plt.plot( train_df['sales'], label='Train')
plt.plot(test_df['sales'], label='Test')
plt.plot(y_hat_avg['SES'], label='Simple_ES')
plt.legend(loc='best')
plt.title("Simple Exponential Smoothing")
plt.show()


rms = sqrt(mean_squared_error(test_df.sales, y_hat_avg.SES))
print(rms)

#Holt (level + trend)
# what will the forecast look like? a trending line
# alpha/smoothing level will place it along the y axis
# beta/smoothing slope will give its rise/decline


y_hat_avg = test_df.copy()
fit1 = Holt(np.asarray(train_df['sales']), exponential=True).fit(smoothing_level=0.3, smoothing_slope=0.69)
y_hat_avg['Holt'] = fit1.forecast(len(test_df))
plt.figure(figsize=(16,8))
plt.plot( train_df['sales'], label='Train')
plt.plot(test_df['sales'], label='Test')
plt.plot(y_hat_avg['Holt'], label='Holt')
plt.legend(loc='best')
plt.title("hot")
plt.show()


rms = sqrt(mean_squared_error(test_df.sales, y_hat_avg.Holt))
print(rms)

# now why does this do worse then just the level? 
# because we need to find the optimal parameters alpha & beta
# now try it with optimized=True


y_hat_avg = test_df.copy()
fit1 = Holt(np.asarray(train_df['sales']), exponential=True).fit(optimized=True)
y_hat_avg['Holt'] = fit1.forecast(len(test_df))
plt.figure(figsize=(16,8))
plt.plot( train_df['sales'], label='Train')
plt.plot(test_df['sales'], label='Test')
plt.plot(y_hat_avg['Holt'], label='Holt')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test_df.sales, y_hat_avg.Holt))
print(rms)

#Holt damped (level + trend, damped)


y_hat_avg = test_df.copy()
fit1 = Holt(np.asarray(train_df['sales']), exponential=True, damped=True).fit(smoothing_level=0.3, smoothing_slope=0.7, damping_slope=0.15)
y_hat_avg['Holt_damped'] = fit1.forecast(len(test_df))
plt.figure(figsize=(16,8))
plt.plot( train_df['sales'], label='Train')
plt.plot(test_df['sales'], label='Test')
plt.plot(y_hat_avg['Holt_damped'], label='Holt_damped')
plt.legend(loc='best')
plt.title("Holt damped")
plt.show()

rms = sqrt(mean_squared_error(test_df.sales, y_hat_avg.Holt_damped))
print(rms)

# Double Exponential Smoothing (level + season)
y_hat_avg = test_df.copy()
fit1 = ExponentialSmoothing(np.asarray(train_df['sales']), seasonal_periods=12, trend=None, seasonal='multiplicative',).fit(smoothing_level=0.19, smoothing_seasonal=0.1)
y_hat_avg['DES'] = fit1.forecast(len(test_df))
plt.figure(figsize=(16,8))
plt.plot( train_df['sales'], label='Train')
plt.plot(test_df['sales'], label='Test')
plt.plot(y_hat_avg['DES'], label='Double ES')
plt.legend(loc='best')
plt.title("Double Exponential Smoothing")
plt.show()

rms = sqrt(mean_squared_error(test_df.sales, y_hat_avg.DES))
print(rms)

#Holt-Winters (level + trend + season)

# here we have trend and seasonality, so we will use Holt-Winters
# the squiggly line should go up or down, depending the trend


y_hat_avg = test_df.copy()
fit1 = ExponentialSmoothing(np.asarray(train_df['sales']),\
                            seasonal_periods=12, trend='add', \
                            seasonal='add').fit(smoothing_level=0.19, smoothing_slope=0.005,smoothing_seasonal=0.1)
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test_df))
plt.figure(figsize=(16,8))
plt.plot( train_df['sales'], label='Train')
plt.plot(test_df['sales'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.title("Holt-Winters")
plt.show()

rms = sqrt(mean_squared_error(test_df.sales, y_hat_avg.Holt_Winter))
print(rms)

#Gridsearching Holt-Winters (level + trend + season)
# using optimized=True and hyperparams from modified gridsearch above

y_hat_avg = test_df.copy()
fit1 = ExponentialSmoothing(np.asarray(train_df['sales']),seasonal_periods=12, trend='add', damped=True, seasonal='add').fit(optimized=True)
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test_df))
plt.figure(figsize=(16,8))
plt.plot( train_df['sales'], label='Train')
plt.plot(test_df['sales'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.title("Holt-Winters")
plt.show()

rms = sqrt(mean_squared_error(test_df.sales, y_hat_avg.Holt_Winter))
print(rms)
print(fit1.forecast(len(test_df)))
