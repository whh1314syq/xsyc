# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing,Holt,SimpleExpSmoothing
import pandas as pd 
import numpy as np


data = pd.read_csv('E:/xsyc/xsyc/data/xsyc.csv')
data = np.asarray(data['sales'])



'''
# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(data).fit(smoothing_level=0.2,optimized=False)
# plot
l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='o')


fit2 = SimpleExpSmoothing(data).fit(smoothing_level=0.6,optimized=False)
# plot
l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(5)), marker='o')


fit3 = SimpleExpSmoothing(data).fit()
# plot
l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(5)), marker='o')

l4, = plt.plot(data, marker='o')
plt.legend(handles = [l1, l2, l3, l4], labels = ['a=0.2', 'a=0.6', 'auto', 'data'], loc = 'best', prop={'size': 7})
plt.show()

# Holt’s Method
fit1 = Holt(data).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(12)), marker='^')

fit2 = Holt(data, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(12)), marker='.')

fit3 = Holt(data, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(12)), marker='.')

l4, = plt.plot(data, marker='.')
plt.legend(handles = [l1, l2, l3, l4], labels = ["Holt's linear trend", "Exponential trend", "Additive damped trend", 'data'], loc = 'best', prop={'size': 7})
plt.show()
'''

# trend=t, damped=d, seasonal=s, seasonal_periods=p,use_boxcox=b, remove_bias=r
 
#fit1 = ExponentialSmoothing(data, seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)
#fit2 = ExponentialSmoothing(data, seasonal_periods=4, trend='add', seasonal='mul').fit(use_boxcox=True)
fit3 = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
#fit4 = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
fit5 = ExponentialSmoothing(data,seasonal_periods=12, trend='add', damped=True, seasonal='add').fit(optimized=True)
fit6 = ExponentialSmoothing(data,seasonal_periods=12, trend='add', damped=True, seasonal='add').fit(optimized=True,use_boxcox=False,remove_bias=False)
fit7 = ExponentialSmoothing(data,seasonal_periods=12, trend='add', damped=False, seasonal='add').fit(optimized=True,use_boxcox=True,remove_bias=True)
print(list(fit6.forecast(12)))
#l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='^')
#l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(5)), marker='*')
#l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(12)), marker='.')
#l4, = plt.plot(list(fit4.fittedvalues) + list(fit4.forecast(12)), marker='.')
l6, = plt.plot(list(fit6.fittedvalues) + list(fit6.forecast(12)), marker='.')

#l5, = plt.plot(data, marker='.')
plt.legend(handles = [l6],
           labels = ["data"], 
           loc = 'best', prop={'size': 7})

plt.show()