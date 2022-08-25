import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler

stock_data = pd.read_csv('MSFT.csv')

close_values = stock_data.filter(['Close']).values

scaler = MinMaxScaler(feature_range=(0, 1))
close_values = scaler.fit_transform(close_values)

train = close_values[0 : math.ceil(len(close_values) * 0.7)]

x_train = []
y_train = []
for i in range(90, len(train)):
    x_train.append(train[i - 90 : i, 0])
    y_train.append(train[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=5)

prediction_input = close_values[len(close_values) - 90 : len(close_values)]
prediction = model.predict(prediction_input)
prediction = scaler.inverse_transform(prediction)
prediction = prediction.reshape((90))
print(prediction)