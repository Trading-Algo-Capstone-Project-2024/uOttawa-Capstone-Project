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

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seed for reproducibility
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)
np.random.seed(42)

ticker = "AAPL"
# 1. Load the dataset, skipping rows with missing data after the header
df = pd.read_csv(f'MachineLearning/Financial Data/{ticker}/{ticker}Full.csv', skiprows=range(1,57), parse_dates=['Date'], index_col='Date')
# df = df.drop(columns=['avg_sentiment_score'])



# Display the first few rows
print(df.head())

# 2. Handle missing values (if any)
df = df.dropna()  # Drop rows with missing values

# Verify that there are no missing values
print(df.isnull().sum())

# 3. Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

# 4. Create sequences for LSTM
def create_sequences(data, target_column, sequence_length, forecast_horizon):
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

# Set hyperparameters
sequence_length = 30      # Number of past days to consider
forecast_horizon = 1       # Number of days ahead to predict
hidden_size = 20            # Number of features in hidden state
num_layers = 3        # Number of stacked LSTM layers
num_epochs = 100            # Number of epochs for training
batch_size = 16             # Batch size for training
learning_rate = 0.001       # Learning rate for the optimizer
training_size = 0.9         # Ratio of dataset being used for training 


#row skip =  ; sequence_length =  ; forecast_horizon =  ; hidden_size =  ; num_layers =  ; num_epochs =  ; batch_size =  ; learning_rate =  ; training_size =  =====  RMSE


#row skip = 1,57 ; sequence_length = 60 ; forecast_horizon = 2 ; hidden_size = 20 ; num_layers = 5 ; num_epochs = 50 ; batch_size = 16 ; learning_rate = 0.001 ; training_size = 0.8  ===== 9.55 RMSE
#row skip = 1,57 ; sequence_length = 60 ; forecast_horizon = 2 ; hidden_size = 20 ; num_layers = 5 ; num_epochs = 100 ; batch_size = 16 ; learning_rate = 0.001 ; training_size = 0.8  ===== 9.88 RMSE -> better at being in the ball park for testing 

#row skip = 57 ; sequence_length = 60 ; forecast_horizon = 1 ; hidden_size = 20 ; num_layers = 2 ; num_epochs = 100 ; batch_size = 16 ; learning_rate = 0.001 ; training_size = 0.8 ===== 6.08 RMSE
#row skip = 253; sequence_length = 30; forecast_horizon =  2 ; hidden_size = 20; num_layers = 3 ; num_epochs = 100 ; batch_size = 16 ; learning_rate = 0.001 ; training_size = 0.8 ===== 8.47  RMSE


# Create sequences
X, y = create_sequences(scaled_data, 'Close', sequence_length, forecast_horizon)

# Display shapes
print("Input shape:", X.shape)    # (num_samples, sequence_length, num_features)
print("Output shape:", y.shape)   # (num_samples, forecast_horizon)

# 5. Split data into training and testing sets
train_size = int(training_size * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. Convert data to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 7. Define the LSTM model
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

# 8. Training the model
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

    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

# 9. Evaluating the model
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for sequences, targets in test_loader:
        outputs = model(sequences)
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

# Convert lists to arrays
predictions = np.concatenate(predictions, axis=0)  # Shape: (num_samples, forecast_horizon)
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

predictions_inv = np.array(predictions_inv)  # Shape: (num_samples, forecast_horizon)
actuals_inv = np.array(actuals_inv)

# Flatten the arrays for calculating overall RMSE
predictions_flat = predictions_inv.flatten()
actuals_flat = actuals_inv.flatten()

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(actuals_flat, predictions_flat))
print(f'Root Mean Squared Error: {rmse:.2f}')

# 10. Plotting the results for test data
# Plotting the first 5 predictions vs actuals
import matplotlib.dates as mdates

# num_plots = 0  # Number of samples to plot
# plt.figure(figsize=(12, 8))
# for i in range(num_plots):
#     plt.subplot(num_plots, 1, i + 1)
#     dates = df.index[-len(actuals_inv) + i] + pd.to_timedelta(np.arange(forecast_horizon), unit='D')
#     plt.plot(dates, actuals_inv[i], label='Actual Close Price')
#     plt.plot(dates, predictions_inv[i], label='Predicted Close Price')
#     plt.title(f'Sample {i + 1}')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.tight_layout()

# plt.show()

# 11. Generating Predictions Across the Entire Dataset

# Prepare the data for inference
full_sequences = []
num_samples = len(scaled_data) - sequence_length - forecast_horizon + 1

for i in range(num_samples):
    x = scaled_data[i:i + sequence_length]
    full_sequences.append(x)

full_sequences = np.array(full_sequences)

# Convert to tensor and move to device
full_sequences_tensor = torch.tensor(full_sequences, dtype=torch.float32).to(device)

# Generate predictions over the entire dataset
model.eval()
with torch.no_grad():
    full_predictions = []
    for i in range(0, len(full_sequences_tensor), batch_size):
        sequences = full_sequences_tensor[i:i + batch_size]
        outputs = model(sequences)
        full_predictions.append(outputs.cpu().numpy())

full_predictions = np.concatenate(full_predictions, axis=0)  # Shape: (num_samples, forecast_horizon)

# Inverse transform the predictions
full_predictions_inv = []

for i in range(full_predictions.shape[0]):
    pred_inv = inverse_transform(full_predictions[i], close_index, scaler, num_features)
    full_predictions_inv.append(pred_inv)

full_predictions_inv = np.array(full_predictions_inv)  # Shape: (num_samples, forecast_horizon)

# For plotting, we'll take the first prediction from each forecast
full_predictions_single = full_predictions_inv[:, 0]  # First day ahead prediction

# 12. Predicting the Next 5 Days Beyond the Dataset

# Get the last sequence from the dataset
last_sequence = scaled_data[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, num_features)

# Convert to tensor
last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)

# Generate future predictions
model.eval()
with torch.no_grad():
    future_predictions = model(last_sequence_tensor)
    future_predictions = future_predictions.cpu().numpy()

# Inverse transform future predictions
future_predictions_inv = inverse_transform(future_predictions[0], close_index, scaler, num_features)

# 13. Preparing Dates and Combining Data

# Get dates for the predictions
prediction_dates = df.index[sequence_length + forecast_horizon - 1:]

# Extend the dates to include future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

# Combine the dates
all_dates = prediction_dates.append(future_dates)

# Get the actual 'Close' prices starting from sequence_length + forecast_horizon - 1
actual_prices = df['Close'].values[sequence_length + forecast_horizon - 1:]

# Combine actual prices with NaNs for future dates
all_actual_prices = np.concatenate([actual_prices, [np.nan]*forecast_horizon])

# Combine predicted prices
all_predicted_prices = np.concatenate([full_predictions_single, future_predictions_inv])

# 14. Plotting the Results Across the Entire Dataset

plt.figure(figsize=(14, 7))
plt.plot(all_dates, all_actual_prices, label='Actual Close Price')
plt.plot(all_dates, all_predicted_prices, label='Predicted Close Price')
plt.title('Stock Close Price Prediction Across Entire Dataset')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()



