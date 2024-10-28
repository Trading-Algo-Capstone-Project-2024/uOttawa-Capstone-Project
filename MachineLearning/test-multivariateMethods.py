#test-multivariateMethods.py

import pytest
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from multivariateMethods import *


def test_load_dataset():
    # Test valid file path
    df = load_dataset('NVDA1Y.csv')
    assert 'Date' in df.columns
    
    # Test invalid file path
    with pytest.raises(FileNotFoundError):
        load_dataset('invalid_path.csv')

def test_preprocess_data():
    df = pd.DataFrame({'Open': [1, 2], 'Close': [2, 3]})
    df_scaled, scaler = preprocess_data(df, columns=['Open', 'Close'])
    assert isinstance(df_scaled, np.ndarray)

def test_prepare_training_data():
    df_scaled = np.random.rand(100, 6)  # Mock data
    trainX, trainY = prepare_training_data(df_scaled, n_past=28, n_future=7)
    assert trainX.shape[1] == 28 and trainX.shape[2] == 6
    
    
def test_make_forecast_valid_input():
    # Setup for a basic test with valid data
    model = Sequential()
    model.add(Dense(1, input_shape=(28, 6)))  # Minimal model for testing
    model.compile(optimizer='adam', loss='mse')
    trainX = np.random.rand(30, 28, 6)  # Mock data, 30 samples
    scaler = StandardScaler().fit(trainX.reshape(-1, 6))
    n_future = 7
    forecast_dates = pd.date_range('2023-08-23', periods=n_future).tolist()
    df_for_training_shape = (30, 6)

    # Run test
    forecast_df = make_forecast(model, trainX, scaler, n_future, forecast_dates, df_for_training_shape)
    assert isinstance(forecast_df, pd.DataFrame)
    assert forecast_df.shape[0] == n_future
    assert 'Date' in forecast_df.columns and 'Open' in forecast_df.columns

def test_make_forecast_empty_trainX():
    model = Sequential()
    model.add(Dense(1, input_shape=(28, 6)))
    model.compile(optimizer='adam', loss='mse')
    trainX = np.array([]).reshape(0, 28, 6)  # Empty array
    scaler = StandardScaler().fit(np.random.rand(30, 6))
    n_future = 7
    forecast_dates = pd.date_range('2023-08-23', periods=n_future).tolist()
    df_for_training_shape = (30, 6)

    with pytest.raises(ValueError):
        make_forecast(model, trainX, scaler, n_future, forecast_dates, df_for_training_shape)

def test_make_forecast_mismatched_df_shape():
    model = Sequential()
    model.add(Dense(1, input_shape=(28, 6)))
    model.compile(optimizer='adam', loss='mse')
    trainX = np.random.rand(30, 28, 6)
    scaler = StandardScaler().fit(trainX.reshape(-1, 6))
    n_future = 7
    forecast_dates = pd.date_range('2023-08-23', periods=n_future).tolist()
    df_for_training_shape = (30, 7)  # Incorrect shape

    with pytest.raises(ValueError):
        make_forecast(model, trainX, scaler, n_future, forecast_dates, df_for_training_shape)
        
        
def test_plot_forecast_valid_data():
    # Setup for valid plot test
    original = pd.DataFrame({
        'Date': pd.date_range('2023-08-23', periods=30),
        'Open': np.random.rand(30)
    })
    forecast_df = pd.DataFrame({
        'Date': pd.date_range('2023-09-22', periods=7),
        'Open': np.random.rand(7)
    })

    # Run plot function and ensure no errors
    try:
        plot_forecast(original, forecast_df)
        plt.close()  # Close plot after test
    except Exception as e:
        pytest.fail(f"plot_forecast raised an exception: {e}")

def test_plot_forecast_empty_forecast():
    original = pd.DataFrame({
        'Date': pd.date_range('2023-08-23', periods=30),
        'Open': np.random.rand(30)
    })
    forecast_df = pd.DataFrame(columns=['Date', 'Open'])  # Empty DataFrame

    try:
        plot_forecast(original, forecast_df)
        plt.close()
    except Exception as e:
        pytest.fail(f"plot_forecast raised an exception with empty forecast data: {e}")

def test_plot_forecast_invalid_date_column():
    original = pd.DataFrame({
        'Date': ['not a date'] * 30,
        'Open': np.random.rand(30)
    })
    forecast_df = pd.DataFrame({
        'Date': pd.date_range('2023-09-22', periods=7),
        'Open': np.random.rand(7)
    })

    with pytest.raises(TypeError):
        plot_forecast(original, forecast_df)
        plt.close()