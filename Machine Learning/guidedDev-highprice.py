#This is from the following link https://medium.com/@pennQuin/implementation-of-long-short-term-memory-lstm-81e35fa5ca54
#I will be using this as my base to improve my knowledge in ML programming

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import tensorflow as tf
tf.config.list_physical_devices('GPU')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



#Support Functions
#The first one is used for graphing a stock
#Whilst the second is used for finding out the mean squared error between the prediction and the reality

def plot_predictions(test, predicted):
    plt.plot(test, color='red', label='Real AMD Stock Price')
    plt.plot(predicted, color='blue', label='Predicted AMD Stock Price')
    plt.title('AMD Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('AMD Stock Price')
    plt.legend()
    plt.show()
    
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test,predicted))
    print("The root mean squared error is {}.".format(rmse))
    
    

#Data sets are loaded and divided into training and testing sets.
#Rows in a training set go up to a date, whilse the test sets consits of rows corresponding to after the set date
#This lets us try predicting a result and check the accuracy later


dataset = pd.read_csv("Machine Learning\\AMD5Y.csv", index_col='Date', parse_dates=['Date'])
dataset.head()
print(dataset)
training_set = dataset[:'2023'].iloc[:,1:2].values
test_set = dataset['2024':].iloc[:,1:2].values


dataset["High"][:'2023'].plot(figsize=(16,4), legend = True)
dataset["High"]['2024':].plot(figsize=(16,4), legend = True)
plt.legend(['Training set (Before 2024)','Test set (2024 and beyond)'])
plt.title('AMD Stock Price')
plt.show()

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)




#the following lines are used for iteration through the excel file for the training portion?
#the number 60 represents the amount of days preserved within the memory

x_train = []
y_train = []
for i in range(60,1119):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))



#LSTM Architecture
regressor = Sequential()


#First LSTM layer with droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Fourth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#ouput layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
#fitting to the training set
regressor.fit(x_train,y_train,epochs=10,batch_size=32)



dataset_total = pd.concat((dataset["High"][:'2023'], dataset["High"]['2024':]), axis = 0)
inputs = dataset_total[len(dataset_total)-len(test_set)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range (60, 200):
    x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)
# print(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# print(x_test)


predicted_stock_price = regressor.predict(x_test)
# print(predicted_stock_price)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# print(predicted_stock_price)

plot_predictions(test_set, predicted_stock_price)

return_rmse(test_set, predicted_stock_price)