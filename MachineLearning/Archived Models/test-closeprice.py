# test-closeprice.py

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