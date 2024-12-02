# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from itertools import product
import warnings

'''

This document is for testing multiple hyperparameter combinations and finding the best results.

We intend this file to return a .csv/.xlsx with all combinations for review - this will be accomplished at a later date

'''

warnings.filterwarnings('ignore')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seed for reproducibility
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)
np.random.seed(42)

# Function to run the model with given hyperparameters
def run_model(sequence_length, forecast_horizon, hidden_size, num_layers, num_epochs, batch_size, learning_rate, training_size, skiprows, ticker):

    # Load the dataset
    df = pd.read_csv(f'MachineLearning/Financial Data/{ticker}/{ticker}Full.csv', skiprows=skiprows, parse_dates=['Date'], index_col='Date')

    # Handle missing values
    df = df.dropna()

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    # Create sequences for LSTM
    def create_sequences(data, target_column, sequence_length, forecast_horizon, df):
        xs = []
        ys = []
        target_idx = df.columns.get_loc(target_column)
        num_samples = len(data) - sequence_length - forecast_horizon + 1
        for i in range(num_samples):
            x = data[i:i + sequence_length]
            y = data[i + sequence_length:i + sequence_length + forecast_horizon, target_idx]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_data, 'Close', sequence_length, forecast_horizon, df)

    # Split data into training and testing sets
    train_size = int(training_size * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert data to PyTorch tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the LSTM model
    class StockPriceLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(StockPriceLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # Define the LSTM layer
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

            # Define the output layer
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # Initialize hidden and cell states with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

            # Forward propagate the LSTM
            out, _ = self.lstm(x, (h0, c0))

            # Get the output of the last time step
            out = self.fc(out[:, -1, :])
            return out

    # Instantiate the model
    input_size = X_train.shape[2]     # Number of features
    output_size = forecast_horizon    # Predicting multiple days ahead
    model = StockPriceLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for sequences, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sequences)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Optional: Print training progress
        # avg_loss = epoch_loss / len(train_loader)
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

    # Evaluating the model
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for sequences, targets in test_loader:
            outputs = model(sequences)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    # Convert lists to arrays
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Inverse transform predictions and actuals
    def inverse_transform(scaled_data, index, scaler, num_features):
        temp = np.zeros((scaled_data.shape[0], num_features))
        temp[:, index] = scaled_data
        inv_transformed = scaler.inverse_transform(temp)
        return inv_transformed[:, index]

    num_features = df.shape[1]
    close_index = df.columns.get_loc('Close')

    # Inverse transform each prediction and actual set
    predictions_inv = []
    actuals_inv = []

    for i in range(predictions.shape[0]):
        pred_inv = inverse_transform(predictions[i], close_index, scaler, num_features)
        act_inv = inverse_transform(actuals[i], close_index, scaler, num_features)
        predictions_inv.append(pred_inv)
        actuals_inv.append(act_inv)

    predictions_inv = np.array(predictions_inv)
    actuals_inv = np.array(actuals_inv)

    # Flatten the arrays for calculating overall RMSE
    predictions_flat = predictions_inv.flatten()
    actuals_flat = actuals_inv.flatten()

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(actuals_flat, predictions_flat))
    print(f'RMSE: {rmse:.2f}')

    return rmse

# Hyperparameter grids
ticker = "AAPL"
sequence_length_list = [30, 60]#[30, 60]          # Number of past days to consider
forecast_horizon_list = [1, 2, 3, 4, 5]           # Number of days ahead to predict
hidden_size_list = [10, 15, 20]              # Number of features in hidden state
num_layers_list = [2, 3, 4, 5]#[2, 3, 4, 5]                 # Number of stacked LSTM layers
num_epochs_list = [50, 100]              # Number of epochs for training
batch_size_list = [8, 16, 24]#[8, 16, 32]               # Batch size for training
learning_rate_list = [0.001]     # Learning rate for the optimizer
training_size_list = [0.9]#[0.7, 0.8, 0.9]               # Ratio of dataset used for training
skiprows_list = [57, 253, 505]#[57, 253, 505]           #  Skip 57 first rows(buffer for rsi&hma), first year, second year

# Generate all combinations of hyperparameters
from itertools import product

hyperparameter_combinations = list(product(sequence_length_list,
                                           forecast_horizon_list,
                                           hidden_size_list,
                                           num_layers_list,
                                           num_epochs_list,
                                           batch_size_list,
                                           learning_rate_list,
                                           training_size_list,
                                           skiprows_list,
                                           ))

# Store results
results = []

# Loop over all combinations
for idx, (sequence_length, forecast_horizon, hidden_size, num_layers, num_epochs, batch_size, learning_rate, training_size, skiprows) in enumerate(hyperparameter_combinations):
    print(f"\nTesting combination {idx+1}/{len(hyperparameter_combinations)}:")
    print(f"sequence_length={sequence_length}, forecast_horizon={forecast_horizon}, hidden_size={hidden_size}, num_layers={num_layers}, num_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, training_size={training_size}, skiprows={skiprows}")


    #Using a try-catch here to be able to continue even if presented with an error
    try:
        rmse = run_model(sequence_length, forecast_horizon, hidden_size, num_layers, num_epochs, batch_size, learning_rate, training_size, range(1, skiprows), ticker)
        results.append({
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'training_size': training_size,
            'skiprows': skiprows,
            'RMSE': rmse
        })
        
    #
    except Exception as e:
        print(f"Error with combination {idx+1}: {e}")
        continue

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nAll Results:")
print(results_df)
results_df.to_csv(f"{ticker}ParametersTesting.csv")

# Find the best hyperparameter combination
best_result = results_df.loc[results_df['RMSE'].idxmin()]
print("\nBest Hyperparameter Combination:")
print(best_result)
