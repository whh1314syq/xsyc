# -*- coding: utf-8 -*-
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


from pandas import read_table
from pandas import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	# define model
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# fit model
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
def measure_mse(actual, predicted):
	return mean_squared_error(actual, predicted)
def measure_mae(actual, predicted):
	return mean_absolute_error(actual, predicted)
def measure_mape(actual, predicted):
	return mape(actual, predicted)
def measure_smape(actual, predicted):
	return smape(actual, predicted)

# MAPE和SMAPE需要自己实现
def mape(actual, predicted):
    return np.mean(np.abs((predicted - actual) / actual)) * 100

def smape(actual, predicted):
    return 2.0 * np.mean(np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual))) * 100






# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = exp_smoothing_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul'] # removed none
	d_params = [True, False] # damped
	s_params = ['add', 'mul'] # removed none
	p_params = seasonal
	b_params = [True, False] # Box Cox
	r_params = [True, False] # remove bias
   
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:      
							   cfg = [t,d,s,p,b,r]
							   models.append(cfg)
	return models

if __name__ == '__main__':
	# load dataset
    df = pd.read_csv('E:/xsyc/xsyc/data/xsyc.csv')
    data1 = np.asarray(df['sales'])
    data = df.set_index(pd.DatetimeIndex(start='1/31/2016', end='1/1/2020', freq='M')).values
    
    # data split: here we used 127 observations
    n_test = 12
    n_test1 = 4
    # model configs
    cfg_list = exp_smoothing_configs(seasonal=[12])
    cfg_list1 = exp_smoothing_configs(seasonal=[4])
    # grid search
    scores = grid_search(data, cfg_list, n_test)
    scores1 = grid_search(data, cfg_list1, n_test1)
    print('done')
    # list top 3 configs
    for cfg, error in scores:
        print("以一年为周期：", cfg, error)
    for cfg, error in scores1:
        print("以四个月为周期：", cfg, error)
        
    fit7 = ExponentialSmoothing(data1,seasonal_periods=12, trend='add', damped=True, seasonal='add').fit(optimized=True,use_boxcox=False,remove_bias=False)
    fit8 = ExponentialSmoothing(data1,seasonal_periods=4, trend='add', damped=True, seasonal='add').fit(optimized=True,use_boxcox=False,remove_bias=True)
    print(list(fit7.forecast(12)))
    print(list(fit8.forecast(4)))
    l7, = plt.plot(list(fit7.fittedvalues) + list(fit7.forecast(12)), marker='.')
    l8, = plt.plot(list(fit8.fittedvalues) + list(fit8.forecast(4)), marker='.')
    l9, = plt.plot(data1, marker='.')
    plt.legend(handles = [l7,l8,l9],
           labels = ["sales12","sales4","data"], 
           loc = 'best', prop={'size': 7})
    plt.show()







