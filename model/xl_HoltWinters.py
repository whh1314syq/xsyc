#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Time    : 2020/1/7 11:17
# @Author  : syq
# @Email   : 1164004566@qq.com
# @tel     : 13541723445
# @File    : xl_HoltWunters.py
# @Software: PyCharm
'''
#-----------------------

import math
import numpy  as np
import pandas as pd

from sklearn        import linear_model
from scipy.optimize import fmin_l_bfgs_b
from model.arimaModel import loadData
from settings import fname
import matplotlib.pyplot as plt

# bring in the passenger data from HW4 to test the function against R output
#---------------------------------------------------------------------------

#data = list(map(int, loadData(fname).read().split('\n')))

data = open('E:/xsyc/xsyc/data/xsyc1.csv')
data = data.read().split('\n')
data = list(map(int, data))

# define main function [holtWinters] to generate retrospective smoothing/predictions
#-----------------------------------------------------------------------------------
 
def holtWinters(ts, p, sp, ahead, mtype, alpha = None, beta = None, gamma = None):
    '''HoltWinters retrospective smoothing & future period prediction algorithm 
       both the additive and multiplicative methods are implemented and the (alpha, beta, gamma)
       parameters have to be either all user chosen or all optimized via one-step-ahead prediction MSD
       initial (a, b, s) parameter values are calculated with a fixed-period seasonal decomposition and a
       simple linear regression to estimate the initial level (B0) and initial trend (B1) values
    @params:
        - ts:            时间序列(序列时间由远及近)
        - p[int]:        时间序列的周期
        - test_num[int]: 测试集长度
        - sp[int]:       计算初始化参事所需要的周期数(周期数必须大于1)
        - ahead[int]:    需要预测的滞后数
        - mtype[string]: 时间序列方法:累加法或累乘法 ['additive'/'multiplicative']
    """
    @params:
        - ts[list]:      time series of data to model
        - p[int]:        period of the time series (for the calculation of seasonal effects)
        - sp[int]:       number of starting periods to use when calculating initial parameter values
        - ahead[int]:    number of future periods for which predictions will be generated
        - mtype[string]: which method to use for smoothing/forecasts ['additive'/'multiplicative']
        - alpha[float]:  user-specified level  forgetting factor (one-step MSD optimized if None)
        - beta[float]:   user-specified slope  forgetting factor (one-step MSD optimized if None)
        - gamma[float]:  user-specified season forgetting factor (one-step MSD optimized if None)
    @return: 
        - alpha[float]:    chosen/optimal level  forgetting factor used in calculations
        - beta[float]:     chosen/optimal trend  forgetting factor used in calculations
        - gamma[float]:    chosen/optimal season forgetting factor used in calculations
        - MSD[float]:      chosen/optimal Mean Square Deviation with respect to one-step-ahead predictions
        - params[tuple]:   final (a, b, s) parameter values used for the prediction of future observations
        - smoothed[list]:  smoothed values (level + trend + seasonal) for the original time series
        - predicted[list]: predicted values for the next @ahead periods of the time series
    sample calls:
        results = holtWinters(ts, 12, 4, 24, 'additive')
        results = holtWinters(ts, 12, 4, 24, 'multiplicative', alpha = 0.1, beta = 0.2, gamma = 0.3)'''

    a, b, s = _initValues(mtype, ts, p, sp)

    if alpha == None or beta == None or gamma == None:
        ituning   = [0.1, 0.1, 0.1]
        ibounds   = [(0,1), (0,1), (0,1)]
        optimized = fmin_l_bfgs_b(_MSD, ituning, args = (mtype, ts, p, a, b, s[:]), bounds = ibounds, approx_grad = True)
        alpha, beta, gamma = optimized[0]

    MSD, params, smoothed = _expSmooth(mtype, ts, p, a, b, s[:], alpha, beta, gamma)
    predicted = _predictValues(mtype, p, ahead, params)

    return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'MSD': MSD, 'params': params, 'smoothed': smoothed, 'predicted': predicted}

def _initValues(mtype, ts, p, sp):
    '''subroutine to calculate initial parameter values (a, b, s) based on a fixed number of starting periods'''

    initSeries = pd.Series(ts[:p*sp])

    if mtype == 'additive':
        rawSeason  = initSeries - pd.Series(initSeries).rolling(window = p, min_periods = p, center = True).mean()
        #rawSeason  = initSeries - pd.rolling_mean(initSeries, window = p, min_periods = p, center = True)
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) - np.mean(initSeason)
        deSeasoned = [initSeries[v] - initSeason[v % p] for v in range(len(initSeries))]
    else:
        rawSeason  = initSeries / pd.Series(initSeries).rolling(window = p, min_periods = p, center = True).mean()
        #rawSeason  = initSeries / pd.rolling_mean(initSeries, window = p, min_periods = p, center = True)
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) / math.pow(np.prod(np.array(initSeason)), 1/p)
        deSeasoned = [initSeries[v] / initSeason[v % p] for v in range(len(initSeries))]

    lm = linear_model.LinearRegression()
    lm.fit(pd.DataFrame({'time': [t+1 for t in range(len(initSeries))]}), pd.Series(deSeasoned))
    return float(lm.intercept_), float(lm.coef_), list(initSeason)

def _MSD(tuning, *args):
    '''subroutine to pass to BFGS optimization to determine the optimal (alpha, beta, gamma) values'''

    predicted = []
    mtype     = args[0]
    ts, p     = args[1:3]
    Lt1, Tt1  = args[3:5]
    St1       = args[5][:]
    alpha, beta, gamma = tuning[:]

    for t in range(len(ts)):

        if mtype == 'additive':
            Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])
            predicted.append(Lt1 + Tt1 + St1[t % p])
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] / Lt)         + (1 - gamma) * (St1[t % p])
            predicted.append((Lt1 + Tt1) * St1[t % p])

        Lt1, Tt1, St1[t % p] = Lt, Tt, St

    return sum([(ts[t] - predicted[t])**2 for t in range(len(predicted))])/len(predicted)

def _expSmooth(mtype, ts, p, a, b, s, alpha, beta, gamma):
    '''subroutine to calculate the retrospective smoothed values and final parameter values for prediction'''

    smoothed = []
    Lt1, Tt1, St1 = a, b, s[:]

    for t in range(len(ts)):

        if mtype == 'additive':
            Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])
            smoothed.append(Lt1 + Tt1 + St1[t % p])
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] / Lt)         + (1 - gamma) * (St1[t % p])
            smoothed.append((Lt1 + Tt1) * St1[t % p])

        Lt1, Tt1, St1[t % p] = Lt, Tt, St

    MSD = sum([(ts[t] - smoothed[t])**2 for t in range(len(smoothed))])/len(smoothed)
    return MSD, (Lt1, Tt1, St1), smoothed

def _predictValues(mtype, p, ahead, params):
    '''subroutine to generate predicted values @ahead periods into the future'''

    Lt, Tt, St = params
    if mtype == 'additive':
        return [Lt + (t+1)*Tt + St[t % p] for t in range(ahead)]
    else:
        return [(Lt + (t+1)*Tt) * St[t % p] for t in range(ahead)]

# print out the results to check against R output
#------------------------------------------------

results = holtWinters(data, 12,2,12, mtype = 'additive')
#results = holtWinters(data, 12, 2,4, mtype = 'multiplicative')
#results = holtWinters(data, 12, 4, , 'additive', alpha = 0.2, beta = 0.2, gamma = 0.3)
print("TUNING: ", results['alpha'], results['beta'], results['gamma'], results['MSD'])
#print("FINAL PARAMETERS: ", results['params'])
print("PREDICTED VALUES: ", results['predicted'])
#print("orig data: ",data)
plt.plot(results['predicted'])


