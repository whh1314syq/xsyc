# load and plot dataset
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
from model.settings import fname1
from model.arimaModel import loadData
# load dataset
'''
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
'''
data = read_csv('E:/xsyc/xsyc/data/sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# summarize first few rows
print(data.head())
# line plot
data.plot()
plt.show()