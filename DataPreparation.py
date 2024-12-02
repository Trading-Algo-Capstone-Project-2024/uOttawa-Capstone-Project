from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from turtle import update
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime, timedelta



# Function to perform sentiment analysis using the FinBERT model
def pipelineMethod(payload):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    res = classifier(payload)
    return res[0]



def scrape(ticker, numPages):
    # Define the columns for the DataFrame
    columns = ['datetime', 'date', 'title', 'source', 'link', 'top_sentiment', 'sentiment_score']
    data = []

    counter = 0
    
    
    # Scrape the data from the website
    for page in range(numPages):
        
        print(f"Scanning Page {page}")
        print()
        
        url = f'https://markets.businessinsider.com/news/{ticker}-stock?p={page}'
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'lxml')

        articles = soup.find_all('div', class_='latest-news__story')
        for article in articles:
            datetime_str = article.find('time', class_='latest-news__date').get('datetime')
            datetime_obj = pd.to_datetime(datetime_str)  # Convert to datetime object
            date = datetime_obj.date()  # Extract just the date
            
            title = article.find('a', class_='news-link').text.strip()
            source = article.find('span', class_='latest-news__source').text.strip()
            link = article.find('a', class_='news-link').get('href')

            # Perform sentiment analysis on the title
            output = pipelineMethod(title)
            top_sentiment = output['label']
            sentiment_score = output['score']
            
            # Collect the data in a list
            data.append([datetime_str, date, title, source, link, top_sentiment, sentiment_score])
            
            counter += 1

    print(f'{counter} headlines scraped from {numPages} pages')

    # Create a DataFrame and save it to CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'Web Scrapper/{ticker}sentiment.csv', index=False, encoding='utf-8')

# scrape('AAPL',168)



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
# updateStock("AAPL")
# updateStock("AMD")
# updateStock("NVDA")
# updateStock("VOO")
# I would probably want to do a VGT ETF for tech sector


def addSentiment(ticker):
    '''
    This function merges the stock data with the averaged sentiment score per day.
    '''
    # Step 1: Read the .csv file
    amd_df = pd.read_csv(f'MachineLearning/Financial Data/{ticker}/{ticker}.csv', skiprows=2)
    column_names = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'RSI_14_custom', 'RSI_50_custom', 'HMA_10_custom', 'HMA_50_custom']
    amd_df.columns = column_names
    amd_df['Date'] = pd.to_datetime(amd_df['Date']).dt.tz_localize(None)

    # Step 2: Read the sentiment data
    sentiment_df = pd.read_csv(f'MachineLearning/Financial Data/{ticker}/{ticker}SentData.csv')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    # Step 3: Calculate average sentiment score per day
    daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'sentiment_score': 'avg_sentiment_score'}, inplace=True)

    # Step 4: Merge the DataFrames on the date
    merged_df = pd.merge(amd_df, daily_sentiment, left_on='Date', right_on='date', how='left')
    merged_df.drop('date', axis=1, inplace=True)

    # Ensure DataFrame is sorted by date
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    # Step 5: Replace zeros in 'avg_sentiment_score' with NaN
    merged_df.loc[merged_df['avg_sentiment_score'] == 0, 'avg_sentiment_score'] = np.nan

    # Step 6: Forward-fill NaN values using ffill()
    merged_df['avg_sentiment_score'] = merged_df['avg_sentiment_score'].ffill()

    # Step 7: Fill any remaining NaN values with 0.5
    merged_df['avg_sentiment_score'] = merged_df['avg_sentiment_score'].fillna(0.5)

    # Step 8: Save to a new CSV file
    merged_df.to_csv(f'MachineLearning/Financial Data/{ticker}/{ticker}Full.csv', index=False)


# addSentiment('AAPL')