# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('E:/xsyc/xsyc/data/xsyc.csv')
data = df.set_index(pd.DatetimeIndex(start='1/31/2016', end='1/1/2020', freq='M'))

df = data.copy()
df['naive'] = df['sales'].shift() # for visualization, shift 30

def cumulative(series):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append((series[:n+1].mean()))
    return result

df['cumulative'] = cumulative(df['sales'])
df['cum2'] = df['sales'].mean()
df['moving_average'] = df['sales'].rolling(window=10).mean()
plt.style.use('fivethirtyeight')
ax = df.plot(figsize=(18, 6), fontsize=14)
plt.title("原始数据趋势图")
plt.show()


rcParams['figure.figsize'] = 18, 6
result_a = seasonal_decompose(data, model='additive')
fig = result_a.plot()
plt.title("累加效果图")
plt.show()



rcParams['figure.figsize'] = 18, 6
result_m = seasonal_decompose(data, model='multiplicative')
fig = result_m.plot()
plt.title("累乘效果图")
plt.show()


################################################################
#Simple Exponential Smoothing (level)
'''
xt=a+ϵtxt=a+ϵt  (model) 
x̂ t,t+1=αxt+(1−α)x̂ t−1,tx^t,t+1=αxt+(1−α)x^t−1,t (forecast) 
only smoothing param: α
'''
# note: additive
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

'''
Holt (level + trend)
xt=a+bt+ϵtxt=a+bt+ϵt 
x̂ t,t+τ=â t+τb̂ tx^t,t+τ=a^t+τb^t 

â t=αxt+(1−α)(â t−1+b̂ t−1)a^t=αxt+(1−α)(a^t−1+b^t−1) 

b̂ t=β(â t−â t−1)+(1−β)b̂ t−1b^t=β(a^t−a^t−1)+(1−β)b^t−1 

smoothing params: αα (level), ββ (trend)
'''
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1: # initialize
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend) # a-hat at t
        trend = beta*(level-last_level) + (1-beta)*trend # b-hat at t
        result.append(level+trend)
    return result

# note: you can dampen trends adding omega
def double_exponential_smoothing_damped(series, alpha, beta, omega):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1: # initialize
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level + omega*trend) # a-hat at t
        trend = beta*(level-last_level) + (1-beta)*omega*trend # b-hat at t
        result.append(level+trend)
    return result

'''
Double Exponential Smoothing (level + season)
xt=aFt+ϵtxt=aFt+ϵt 
x̂ t,t+τ=â t∗τF̂ t+τPx^t,t+τ=a^t∗τF^t+τP 

â t=α(xtF̂ t−P)+(1−α)â t−1a^t=α(xtF^t−P)+(1−α)a^t−1 

F̂ t=γ(xtâ t)+(1−γ)F̂ t−PF^t=γ(xta^t)+(1−γ)F^t−P 

smoothing params: αα (level), γγ (season)
'''
# note: multiplicative
# slen: season length
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals


def double_exponential_smoothing_season(series, slen, alpha, gamma):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1: # initialize
            level, seasonals = series[0], initial_seasonal_components(series, slen)
        if n >= len(series): # we are forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*(value/seasonals[i%slen]) + (1-alpha)*(level)
        seasonals[i%slen] = gamma*(value/smooth) + (1-gamma)*seasonals[i%slen]
        result.append(level*seasonals)
    return result


'''

Holt-Winters (level + trend + season)
xt=(a+bt)Ft+ϵtxt=(a+bt)Ft+ϵt 
x̂ t,t+τ=(â t+τb̂ t)F̂ t+τPx^t,t+τ=(a^t+τb^t)F^t+τP 

â t=α(xtF̂ t−P)+(1−α)(â t−1+b̂ t−1)a^t=α(xtF^t−P)+(1−α)(a^t−1+b^t−1) 

b̂ t=β(â t−â t−1)+(1−β)b̂ t−1b^t=β(a^t−a^t−1)+(1−β)b^t−1 

F̂ t=γ(xtâ t)+(1−γ)F̂ t−PF^t=γ(xta^t)+(1−γ)F^t−P 

smoothing params: αα (level),ββ (trend), γγ (season)
'''

# multiplicative seasonality & additive trend
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen


def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val/seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend # same as level+trend
            seasonals[i%slen] = gamma*(val/smooth) + (1-gamma)*seasonals[i%slen] # same as level+season
            result.append((smooth+trend)*seasonals[i%slen])
    return result















































