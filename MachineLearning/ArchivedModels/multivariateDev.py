import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
#this checks if the gpu is being used for tensorflow
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the dataset
df = pd.read_csv('MachineLearning/NVDA1Y.csv')

# Separate dates for plotting
train_dates = pd.to_datetime(df['Date'])

# Training variables - using columns 1:7 (excluding the date as a data point)
cols = list(df)[1:7]
df_for_training = df[cols].astype(float)

# Scaling the data
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Preparing the training data
trainX = []
trainY = []

n_future = 7  # Number of days we want to predict into the future
n_past = 28  # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Defining the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
# model.add(Dropout(0.2))  # Optional dropout layer to prevent overfitting
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Fitting the model
history = model.fit(trainX, trainY, epochs=1000, batch_size=16, validation_split=0.1, verbose=1)

# Forecasting
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()
forecast = model.predict(trainX[-n_future:])

# Performing inverse transformation to rescale back to original range
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

# Preparing forecast dates
forecast_dates = [time_i.date() for time_i in forecast_period_dates]

# Creating a DataFrame for the forecast
df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

# Preparing the original data for plotting
original = df[['Date', 'Open']]
original['Date'] = pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2023-08-23']

# Plotting the results
sns.lineplot(data=original, x='Date', y='Open', label='Original')
sns.lineplot(data=df_forecast, x='Date', y='Open', label='Forecast')

# Display the plot
plt.show()