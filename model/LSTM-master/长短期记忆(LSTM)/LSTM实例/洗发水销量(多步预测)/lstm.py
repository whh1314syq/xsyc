import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def reorgdata(data, windowsize):
	dataX, dataY = [], []
	for i in range(len(data)+windowsize-1):
		a = data[i:(i+windowsize), 0]
		dataX.append(a)
		dataY.append(data[i - windowsize, 0])
	return np.array(dataX), np.array(dataY)

# load the data
#datasource = pd.read_csv('sdmiv.csv', usecols=[1], engine='python')
datasource = pd.read_csv('E:/xsyc/xsyc/data/sales.csv', usecols=[1], engine='python')
data = datasource.values
data = data.astype('float32')
print("原始数据：",data)

# normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
print("原始数据归一化处理：",data)

# split into train and test sets
train_size = int(len(data) * 0.75)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]
print("归一化后训练数据集：",train)
print("归一化后测试数据集：",test)



# reshape 
windowsize = 1
trainX, trainY = reorgdata(train, windowsize)
testX, testY = reorgdata(test, windowsize)
print('TestX shape: ',testX.shape)
print('TestY shape: ',testY.shape)
print('trainX.shape: ', trainX.shape)
print('trainY.shape: ', trainY.shape)
print(testX[:3])
print(testY[:3])

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('TestX shape: ', testX.shape)
print('TestY shape: ', testY.shape)
print('trainX.shape: ', trainX.shape)
print('trainY.shape: ', trainY.shape)
print(testX[:3])
print(testY[:3])

# create and load stateful LSTM
model = Sequential()
# no need double stateful LSTM

#model.add(LSTM(32,
#          input_shape=(1, windowsize)
#         ,batch_size=1
#         ,stateful=True
#         ,return_sequences=True
#          ))
#model.add(Dropout(0.3))
model.add(LSTM(32,input_shape=(1, windowsize),batch_size=1,stateful=True))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print('-----Training Beginning-----')
for i in range(3000):
    model.fit(trainX,trainY,batch_size = 1,epochs=4, verbose=1,shuffle=False)
    model.reset_states()

# make predictions with feeding batch data into model
trainPredict = model.predict(trainX,batch_size = 1)
#print(trainPredict)
testPredict = model.predict(testX,batch_size = 1)
#print(testPredict)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
print("训练预测数据:",trainPredict)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
print("测试预测数据:",testPredict)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[windowsize-1:len(trainPredict), :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)-1:len(data)-windowsize, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title('LSTM forecast comparison')
plt.ylabel('Actual/Train/Forecast number')
plt.xlabel('Monthly Sequence number')
plt.savefig('./acc_lstm', fmt = 'png', dpi = 300)
plt.show()