from turtle import update
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime, timedelta

def calculate_rsi(data, window=14):
    '''
    Calculating the RSI manually
    '''
    #Calculate the difference in prices
    delta = data['Close'].diff()
    
    #Separate gains and losses
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    
    #Calculate the average gain and loss
    avg_gain = pd.Series(gain, index=data.index).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=window, min_periods=1).mean()
    
    #Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    #Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    #filling NaN rows with 0
    rsi = rsi.fillna(0)
    
    #Align RSI index with data index
    rsi = rsi.reindex(data.index)
    
    #Add the RSI as a new column in the DataFrame using .loc to avoid SettingWithCopyWarning
    data.loc[:, f'RSI_{window}_custom'] = rsi

    return data



    
def weighted_moving_average(series, length):
    """
    Calculate a Weighted Moving Average (WMA)
    """
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hull_moving_average(series, period):
    """
    Calculate Hull Moving Average (HMA)
    """
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    #Calculate WMA of half length
    wma_half = weighted_moving_average(series, half_length)

    #Calculate WMA of full length
    wma_full = weighted_moving_average(series, period)

    #Double the half-length WMA and subtract the full-length WMA
    hma_intermediate = 2 * wma_half - wma_full

    #Calculate the WMA of the intermediate result with sqrt length
    hma = weighted_moving_average(hma_intermediate, sqrt_length)
    
    return hma


def NYSE_latestDay():
    '''
    Finds last day of open market on the NYSE
    '''
    #Get the NYSE market calendar
    nyse = mcal.get_calendar('NYSE')
    today = datetime.today()
    
    #Get the last 10 valid market dates (enables holidays/weekends to not break the code)
    schedule = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
    
    #Get the last open market day
    last_open_day = schedule.index[-1]  # The most recent open market day
    return last_open_day.date()

def updateStock(ticker="AMD", start="2021-01-01", end=NYSE_latestDay()):
    '''
    This function updates a given Ticker's .csv file. 
    Default values are set for AMD from 2021-01-01 to last operating day of NYSE in case of input errors or missing parameters.
    '''
    data = yf.download(ticker, start, end, interval="1d")
    
    #Calculate RSI for 14-day and 50-day periods 
    #This one needs to replace the data directly since I couldn't get the function to return exactly what I needed 
    #* point of improvment in the future possibly
    data = calculate_rsi(data, 14)
    data = calculate_rsi(data, 50)
    
    #Calculate HMA for 10-day and 50-day periods
    data['HMA_10_custom'] = hull_moving_average(data['Close'], 10)
    data['HMA_50_custom'] = hull_moving_average(data['Close'], 50)
    
    #Save to CSV file
    data.to_csv(f"MachineLearning\\Financial Data\\{ticker}.csv")
    


#4 stock tickers that I have interest in
updateStock("AAPL")
updateStock("AMD")
updateStock("NVDA")
updateStock("VOO")