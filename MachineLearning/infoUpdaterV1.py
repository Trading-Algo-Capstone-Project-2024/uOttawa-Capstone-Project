import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def calculate_rsi(data, window=14, ticker="AMD"):
    # Calculate the difference in prices
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    
    # Calculate the average gain and loss
    avg_gain = pd.Series(gain, index=data.index).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=window, min_periods=1).mean()
    
    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)
    
    # Align RSI index with data index
    rsi = rsi.reindex(data.index)
    
    # Add the RSI as a new column in the DataFrame using .loc to avoid SettingWithCopyWarning
    data.loc[:, f'RSI_{window}_custom'] = rsi

    return data



    
def weighted_moving_average(series, length):
    """Calculate a Weighted Moving Average (WMA)"""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hull_moving_average(series, period):
    """Calculate Hull Moving Average (HMA)"""
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    # Step 1: Calculate WMA of half length
    wma_half = weighted_moving_average(series, half_length)

    # Step 2: Calculate WMA of full length
    wma_full = weighted_moving_average(series, period)

    # Step 3: Double the half-length WMA and subtract the full-length WMA
    hma_intermediate = 2 * wma_half - wma_full

    # Step 4: Calculate the WMA of the intermediate result with sqrt length
    hma = weighted_moving_average(hma_intermediate, sqrt_length)
    
    return hma

def updateStock(ticker="AMD", start="2023-01-01", end="2024-11-01"):
    '''
    This function updates a given Ticker's .csv file. 
    Default values are set for AMD from 2023-01-01 to 2024-11-01 in case of input errors or missing parameters.
    '''
    data = yf.download(ticker, start, end, interval="1d")
    
    # Calculate RSI for 14-day and 50-day windows and assign to DataFrame columns
    data = calculate_rsi(data, 14, ticker)
    data = calculate_rsi(data, 50, ticker)
    
    # Calculate HMA for 10-day and 50-day periods
    data['HMA_10_custom'] = hull_moving_average(data['Close'], 10)
    data['HMA_50_custom'] = hull_moving_average(data['Close'], 50)
    
    # Print data to verify column addition before saving
    print("Data before saving:\n", data[['RSI_14_custom', 'RSI_50_custom', 'HMA_10_custom', 'HMA_50_custom']].tail())
    
    # Save to CSV file
    data.to_csv(f"MachineLearning\\Financial Data\\{ticker}Test.csv")
    
    # Plotting for visualization and testing
    # plt.figure(figsize=(14, 7))
    # plt.plot(data['Close'], label='Close Price', linewidth=1.5)
    # plt.plot(data['HMA_10_custom'], label='HMA 10 (Short-Term)', linestyle='--')
    # plt.plot(data['HMA_50_custom'], label='HMA 50 (Long-Term)', linestyle='-.')
    # plt.title(f"{ticker} Close Price with HMA 10 and HMA 50")
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.legend()
    # plt.show()

# # Run the update function for AAPL data as an example
updateStock("AAPL", "2022-01-01", "2024-10-01")
