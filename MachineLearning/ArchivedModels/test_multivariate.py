# test_multivariate.py

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