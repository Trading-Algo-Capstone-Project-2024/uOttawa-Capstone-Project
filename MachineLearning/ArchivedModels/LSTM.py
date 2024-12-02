import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler

#going to want to use the MachineLearning/Financial Data/AMD/AMDFull.csv path
#or MachineLearning/Financial Data/{ticker}/{ticker}Full.csv path
#currently there are 12 columns in AMDFull, including the Date.
#12 features? to be seen

ticker = "AMD"

# Load the dataset
#we are skipping rows 1 to 50 since they do not have RSI/HMA values setup yet
#row 0 is the header
df = pd.read_csv(f'MachineLearning/Financial Data/{ticker}/{ticker}Full.csv', skiprows=range(1,56), parse_dates=['Date'], index_col='Date')
# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the data
scaled_data = scaler.fit_transform(df)

# Convert back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

# Display the first few rows of the scaled data


def create_sequences(data, target_column, sequence_length):
    xs = []
    ys = []
    target_idx = df.columns.get_loc(target_column)
    for i in range(len(data) - sequence_length):
        x = data[i:i + sequence_length]
        y = data[i + sequence_length][target_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Set the sequence length (e.g., 60 days)
sequence_length = 60

# Create sequences
X, y = create_sequences(scaled_data, 'Close', sequence_length)

# Display the shapes
print("Input shape:", X.shape)
print("Output shape:", y.shape)


# Define the training data length
train_size = int(0.8 * len(X))

# Split into training and testing sets
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Display the shapes
print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)


# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model
input_size = X_train_tensor.shape[2]  # Number of features
hidden_size = 50
num_layers = 2

model = StockPriceLSTM(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Number of epochs
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    for sequences, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    # Print training loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for sequences, targets in test_loader:
        outputs = model(sequences)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(targets.tolist())

# Convert predictions and actuals back to original scale
predictions = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predictions), df.shape[1]-1)), np.array(predictions).reshape(-1,1)), axis=1)
)[:, -1]
actuals = scaler.inverse_transform(
    np.concatenate((np.zeros((len(actuals), df.shape[1]-1)), np.array(actuals).reshape(-1,1)), axis=1)
)[:, -1]

# Calculate RMSE
from sklearn.metrics import mean_squared_error
import math

rmse = math.sqrt(mean_squared_error(actuals, predictions))
print(f'Root Mean Squared Error: {rmse:.2f}')



import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(actuals, label='Actual Close Price')
plt.plot(predictions, label='Predicted Close Price')
plt.title('Stock Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
