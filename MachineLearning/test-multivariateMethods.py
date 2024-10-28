#test-multivariateMethods.py

import pytest
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from multivariateMethods import *
import matplotlib
matplotlib.use('Agg')

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
        
        
def test_build_and_train_model():
    # Step 1: Define input shape and use build_model
    input_shape = (28, 6)  # (time_steps, features)
    model = build_model(input_shape)

    # Step 2: Verify model structure
    assert isinstance(model, Sequential), "Model should be an instance of Sequential"
    assert len(model.layers) == 3, f"Expected 3 layers, got {len(model.layers)}"

    # Step 3: Check layer configurations
    assert isinstance(model.layers[0], LSTM), "First layer should be an LSTM layer"
    assert model.layers[0].output_shape == (None, 28, 64)
    assert isinstance(model.layers[1], LSTM), "Second layer should be an LSTM layer"
    assert model.layers[1].output_shape == (None, 32)
    assert isinstance(model.layers[2], Dense), "Third layer should be a Dense layer"
    assert model.layers[2].output_shape == (None, 1)
    assert model.optimizer._name == "adam", "Expected optimizer 'adam'"
    assert model.loss == "mse", "Expected loss 'mse'"

    # Step 4: Create dummy training data
    # Here we use batch size of 16, 28 time steps, 6 features
    dummy_trainX = np.random.rand(16, 28, 6).astype(np.float32)  # Shape: (batch_size, time_steps, features)
    dummy_trainY = np.random.rand(16, 1).astype(np.float32)      # Shape: (batch_size, output_dim)

    # Step 5: Train model on dummy data (one epoch, no validation)
    history = model.fit(dummy_trainX, dummy_trainY, epochs=1, batch_size=4, verbose=0)

    # Verify training results
    assert history.history['loss'] is not None, "Model did not train as expected."

    # Step 6: Test model prediction on new dummy data
    dummy_testX = np.random.rand(1, 28, 6).astype(np.float32)  # Shape: (1, time_steps, features)
    prediction = model.predict(dummy_testX)

    # Verify output shape
    assert prediction.shape == (1, 1), f"Expected prediction shape (1, 1), got {prediction.shape}"

    print("All assertions passed for test_build_and_train_model.")