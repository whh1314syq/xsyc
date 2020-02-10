# -*- coding: utf-8 -*-
import time
import keras
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Activation,Dropout
import matplotlib.pyplot as plt


from pandas import read_csv
from pandas import datetime




def load_data(filename, time_step):
    '''
    filename: 
    instruction: file address, note '/'

    time_step: int
    instruction: how many previous samples are used to predict the next sample, it is the same with the time_steps of that in LSTM
    '''
    df = pd.read_csv(filename, header=None)
    data = df.values
    data = data.astype('float32')  # confirm the type as 'float32'
    data = data.reshape(data.shape[0], )
    result = []
    for index in range(len(data) - time_step):
        result.append(data[index:index + time_step + 1])
    
    # variable 'result' can be (len(data)-time_step) * (time_step + 1), the last column is predicted sample.
    return np.array(result)

data = load_data('E:/xsyc/xsyc/data/xsyc1.csv',48)

# normalize the data and split it into train and test set
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)
# define a variable to represent the ratio of train/total and split the dataset
train_count = int(0.7 * len(dataset))
x_train_set, x_test_set = dataset[:train_count, :-1], dataset[train_count:, :-1]
y_train_set, y_test_set = dataset[:train_count, -1], dataset[train_count:, -1]

# reshape the data to satisfy the input acquirement of LSTM
x_train_set = x_train_set.reshape(x_train_set.shape[0], 1, x_train_set.shape[1])
x_test_set = x_test_set.reshape(x_test_set.shape[0], 1, x_test_set.shape[1])
y_train_set = y_train_set.reshape(y_train_set.shape[0], 1)
y_test_set = y_test_set.reshape(y_test_set.shape[0], 1)

def build_model(layer):
    '''
    layer: list
    instruction: the number of neurons in each layer
    '''
    model = Sequential()
    # set the first hidden layer and set the input dimension
    model.add(LSTM(
        input_shape=(1, layer[0]), units=layer[1], return_sequences=True
    ))
    model.add(Dropout(0.2))

    # add the second layer
    model.add(LSTM(
        units=layer[2], return_sequences=False
    ))
    model.add(Dropout(0.2))

    # add the output layer with a Dense
    model.add(Dense(units=layer[3], activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model

# train the model and use the validation part to validate
model.fit(x_train_set, y_train_set, batch_size=128, epochs=20, validation_split=0.2)

# do the prediction
y_predicted = model.predict(x_test_set)

# plot the predicted curve and the original curve
# fill some zeros to get a (len, 51) array
temp = np.zeros((len(y_test_set), 50))
origin_temp = np.hstack((temp, y_test_set))
predict_temp = np.hstack((temp, y_predicted))

# tranform the data back to the original one
origin_test = scaler.inverse_transform(origin_temp)
predict_test = scaler.inverse_transform(predict_temp)

plot_curve(origin_test[:, -1], predict_test[:, -1])

def plot_curve(true_data, predicted_data):
    '''
    true_data: float32
    instruction: the true test data
    predicted_data: float32
    instruction: the predicted data from the model
    '''
    plt.plot(true_data, label='True data')
    plt.plot(predicted_data, label='Predicted data')
    plt.legend()
    plt.savefig('result.png')
    plt.show()