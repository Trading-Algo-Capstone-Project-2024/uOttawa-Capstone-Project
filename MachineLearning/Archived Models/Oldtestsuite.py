# Oldtestsuite.py

import pytest
import numpy as np
import pandas as pd
import math
from unittest.mock import MagicMock
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Assuming the support functions from the main code are available
def plot_predictions(test, predicted):
    plt.plot(test, color='red', label='Real AMD Stock Price')
    plt.plot(predicted, color='blue', label='Predicted AMD Stock Price')
    plt.title('AMD Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('AMD Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    return rmse


# Test 1: Data Loading and Preprocessing
def test_data_loading():
    dataset = pd.read_csv("../AMD5Y.csv", index_col='Date', parse_dates=['Date'])
    assert not dataset.empty, "Dataset failed to load"
    assert 'Close' in dataset.columns, "'Close' column missing from dataset"

def test_data_splitting():
    dataset = pd.read_csv("../AMD5Y.csv", index_col='Date', parse_dates=['Date'])
    training_set = dataset[:'2023'].iloc[:, 3:4].values
    test_set = dataset['2024':].iloc[:, 3:4].values
    assert len(training_set) > 0, "Training set is empty"
    assert len(test_set) > 0, "Test set is empty"

def test_scaler():
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set = np.array([[100], [200], [300]])
    scaled_data = scaler.fit_transform(training_set)
    assert scaled_data.min() >= 0 and scaled_data.max() <= 1, "Scaler not within specified range"


# Test 2: Data Preparation for Training (Sequence Generation)
def test_sequence_generation():
    data = np.random.rand(100, 1)  # Dummy data
    x_train, y_train = [], []
    for i in range(60, 100):
        x_train.append(data[i - 60:i, 0])
        y_train.append(data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    assert x_train.shape == (40, 60), "x_train shape mismatch"
    assert y_train.shape == (40,), "y_train shape mismatch"


# Test 3: LSTM Model Structure
def test_model_architecture():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    assert len(model.layers) == 9, "Model layer count mismatch"
    assert isinstance(model.layers[0], LSTM), "First layer is not LSTM"
    assert isinstance(model.layers[-1], Dense), "Last layer is not Dense"


# Test 4: Model Training
def test_model_compilation():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(60, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    assert isinstance(model.optimizer, RMSprop), "Optimizer mismatch"
    assert model.loss == 'mean_squared_error', "Loss function mismatch"
def test_model_training():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(60, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # Generate dummy data for testing
    x_train = np.random.rand(10, 60, 1)
    y_train = np.random.rand(10, 1)

    history = model.fit(x_train, y_train, epochs=1, batch_size=2, verbose=0)
    assert 'loss' in history.history, "Loss not recorded in training history"


# Test 5: Prediction and Post-Processing
def test_prediction_shape():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(60, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # Generate dummy data for testing
    x_test = np.random.rand(5, 60, 1)
    predictions = model.predict(x_test)
    assert predictions.shape == (5, 1), "Prediction shape mismatch"

def test_return_rmse():
    test_data = np.array([3, -0.5, 2, 7])
    predicted_data = np.array([2.5, 0.0, 2, 8])
    rmse = return_rmse(test_data, predicted_data)
    expected_rmse = math.sqrt(mean_squared_error(test_data, predicted_data))
    assert abs(rmse - expected_rmse) < 1e-6, "RMSE calculation mismatch"


# Test 6: Visualization
def test_plot_predictions():
    test_data = np.random.rand(100)
    predicted_data = np.random.rand(100)
    try:
        plot_predictions(test_data, predicted_data)
    except Exception as e:
        pytest.fail(f"Plotting function failed with exception: {e}")


import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.autograph.set_verbosity(0)

# Test 1: Data Loading and Preprocessing
def test_data_loading():
    df = pd.read_csv('../NVDA1Y.csv')
    assert not df.empty, "Dataset failed to load"
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in required_columns:
        assert col in df.columns, f"'{col}' column missing from dataset"

def test_date_conversion():
    df = pd.read_csv('../NVDA1Y.csv')
    train_dates = pd.to_datetime(df['Date'])
    assert train_dates.dtype == 'datetime64[ns]', "Date conversion to datetime failed"

def test_data_scaling():
    df = pd.read_csv('../NVDA1Y.csv')
    cols = list(df)[1:7]
    df_for_training = df[cols].astype(float)
    scaler = StandardScaler()
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    assert abs(df_for_training_scaled.mean()) < 1e-5, "Scaled data mean is not close to zero"
    assert np.allclose(df_for_training_scaled.std(), 1, atol=1e-5), "Scaled data std deviation is not close to one"

# Test 2: Data Preparation for Training
def test_sequence_generation():
    df = pd.read_csv('../NVDA1Y.csv')
    cols = list(df)[1:7]
    df_for_training = df[cols].astype(float)
    scaler = StandardScaler()
    df_for_training_scaled = scaler.fit_transform(df_for_training)

    n_future = 7
    n_past = 28
    trainX, trainY = [], []
    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    assert trainX.shape[1] == n_past, "Incorrect sequence length for trainX"
    assert trainY.shape[1] == 1, "trainY should only contain one element per sequence"

# Test 3: LSTM Model Structure and Training
def test_model_architecture():
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(28, 6), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dense(1))

    assert len(model.layers) == 3, "Model should have three layers"
    assert isinstance(model.layers[0], LSTM) and model.layers[0].units == 64, "First LSTM layer configuration mismatch"
    assert isinstance(model.layers[-1], Dense), "Last layer is not Dense"

def test_model_compilation():
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(28, 6), return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    loss_name = model.loss if isinstance(model.loss, str) else model.loss.get_config()['name']
    assert loss_name == 'mean_squared_error', "Loss function mismatch"

def test_model_training():
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(28, 6), return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    x_train = np.random.rand(10, 28, 6)
    y_train = np.random.rand(10, 1)

    history = model.fit(x_train, y_train, epochs=1, batch_size=2, validation_split=0.1, verbose=0)
    assert 'loss' in history.history, "Loss not recorded in training history"

# Test 4: Forecasting
def test_forecast_shape():
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(28, 6), return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    x_test = np.random.rand(7, 28, 6)
    forecast = model.predict(x_test)
    assert forecast.shape == (7, 1), "Forecast shape mismatch"

def test_inverse_scaling():
    scaler = StandardScaler()
    dummy_data = np.random.rand(100, 6)
    scaler.fit(dummy_data)
    scaled_data = scaler.transform(dummy_data[:7])

    forecast_copies = np.repeat(scaled_data[:, 0].reshape(-1, 1), 6, axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]
    assert len(y_pred_future) == 7, "Inverse scaling output length mismatch"

# Test 5: Visualization
def test_plot_forecast():
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-08-23', periods=50),
        'Open': np.random.rand(50) * 100
    })

    df_forecast = pd.DataFrame({
        'Date': pd.date_range(start='2023-09-10', periods=7),
        'Open': np.random.rand(7) * 100
    })

    try:
        sns.lineplot(data=df, x='Date', y='Open', label='Original')
        sns.lineplot(data=df_forecast, x='Date', y='Open', label='Forecast')
        plt.show()
    except Exception as e:
        pytest.fail(f"Plotting function failed with exception: {e}")