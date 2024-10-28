import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

# Disable unnecessary TensorFlow logs for cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Load dataset from a specified CSV file.
def load_dataset(filepath):
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

#Preprocess data by scaling and returning the prepared dataset for training.
def preprocess_data(df, columns)
    df_for_training = df[columns].astype(float)
    scaler = StandardScaler().fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    return df_for_training_scaled, scaler

#Prepare the training data for the LSTM model.
def prepare_training_data(df_scaled, n_past, n_future):
   
    trainX, trainY = [], []
    for i in range(n_past, len(df_scaled) - n_future + 1):
        trainX.append(df_scaled[i - n_past:i, 0:df_scaled.shape[1]])
        trainY.append(df_scaled[i + n_future - 1:i + n_future, 0])
    return np.array(trainX), np.array(trainY)

#Define and compile the LSTM model.
def build_model(input_shape):
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dense(1))  # Adjust the output layer to match the dimension of trainY
    model.compile(optimizer='adam', loss='mse')
    return model


#Train the model with the prepared training data.
def train_model(model, trainX, trainY, epochs=100, batch_size=16, validation_split=0.1):
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    return history

#Generate future predictions and perform inverse transformation.
def make_forecast(model, trainX, scaler, n_future, forecast_dates, df_for_training_shape):
    
    forecast = model.predict(trainX[-n_future:])
    forecast_copies = np.repeat(forecast, df_for_training_shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]
    return pd.DataFrame({'Date': forecast_dates, 'Open': y_pred_future})

def plot_forecast(original, forecast_df):
    """Plot the original and forecasted data."""
    sns.lineplot(data=original, x='Date', y='Open', label='Original')
    sns.lineplot(data=forecast_df, x='Date', y='Open', label='Forecast')
    plt.show()

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    df = load_dataset('MachineLearning/NVDA1Y.csv')
    df_for_training_scaled, scaler = preprocess_data(df, columns=list(df)[1:7])

    # Prepare the training data
    n_past, n_future = 28, 7
    trainX, trainY = prepare_training_data(df_for_training_scaled, n_past, n_future)

    # Build and train the model
    model = build_model((trainX.shape[1], trainX.shape[2]))
    history = train_model(model, trainX, trainY)

    # Forecast future values
    forecast_period_dates = pd.date_range(df['Date'].iloc[-1], periods=n_future, freq='1d').tolist()
    forecast_df = make_forecast(model, trainX, scaler, n_future, forecast_period_dates, df_for_training_scaled.shape)

    # Plot the results
    df['Date'] = pd.to_datetime(df['Date'])
    original = df[['Date', 'Open']].loc[df['Date'] >= '2023-08-23']
    plot_forecast(original, forecast_df)