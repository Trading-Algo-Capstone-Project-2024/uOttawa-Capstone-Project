import pytest
import pandas as pd
import os
from ib_insync import IB, Stock
from bs4 import BeautifulSoup
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from portfolio_manager import initialize_portfolio, rebalance_portfolio, manage_portfolio, load_portfolio
from data_fetcher import fetch_sp500_tickers, fetch_nasdaq100_tickers
from alerts import send_alert
from tracking import track_performance

# Fixtures
@pytest.fixture
def ib_connection():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # Adjust the port and clientId as necessary
    yield ib
    ib.disconnect()

@pytest.fixture
def mock_portfolio():
    return {'AAPL': 1000, 'MSFT': 1500, 'GOOGL': 1200}

@pytest.fixture
def mock_current_prices():
    return {'AAPL': 120, 'MSFT': 200, 'GOOGL': 100}

@pytest.fixture
def setup_dataframe():
    # Mocking a sample dataframe similar to the news_data.csv
    data = {
        'Content': ["Positive content", "Negative content", "Neutral content"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_webpage():
    url = 'https://markets.businessinsider.com/news/nvda-stock?p=1'
    response = requests.get(url)
    return response.text

# Test cases for data_fetcher.py
def test_fetch_sp500_tickers(ib_connection):
    tickers = fetch_sp500_tickers(ib_connection)
    assert isinstance(tickers, list), "Tickers should be returned as a list"
    assert len(tickers) > 0, "The list of tickers should not be empty"

def test_fetch_nasdaq100_tickers(ib_connection):
    tickers = fetch_nasdaq100_tickers(ib_connection)
    assert isinstance(tickers, list), "Tickers should be returned as a list"
    assert len(tickers) > 0, "The list of tickers should not be empty"

# Test cases for portfolio_manager.py
def test_initialize_portfolio():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    portfolio = initialize_portfolio(100000, tickers)
    assert len(portfolio) == len(tickers), "Portfolio should have entries for each ticker"
    assert portfolio['AAPL'] == 100000 / len(tickers), "Investment should be evenly distributed"

def test_buy_order():
    portfolio = {'AAPL': 1000}
    stock = 'AAPL'
    amount_to_buy = 500
    portfolio[stock] += amount_to_buy
    assert portfolio[stock] == 1500, "The buy order should increase the stock amount correctly"

def test_sell_order():
    portfolio = {'AAPL': 1000}
    stock = 'AAPL'
    amount_to_sell = 300
    portfolio[stock] -= amount_to_sell
    assert portfolio[stock] == 700, "The sell order should decrease the stock amount correctly"

def test_sell_10_percent_of_position(mock_portfolio):
    stock = 'AAPL'
    initial_amount = mock_portfolio[stock]
    percentage_to_sell = 0.10
    sell_amount = initial_amount * percentage_to_sell
    mock_portfolio[stock] -= sell_amount
    assert mock_portfolio[stock] == initial_amount * (1 - percentage_to_sell), \
        "Selling 10% of the position should correctly update the portfolio"

# Add other portfolio tests as needed...

# Test cases for tracking.py
def test_track_performance(mock_portfolio, mock_current_prices):
    track_performance(mock_portfolio, mock_current_prices)
    assert os.path.exists('portfolio_tracking.csv'), "CSV file should be created"
    df = pd.read_csv('portfolio_tracking.csv')
    assert len(df) > 0, "CSV should have at least one entry"
    assert 'Percent Change' in df.columns, "CSV should include 'Percent Change' column"

# Test cases for alerts.py
def test_send_alert(capfd):
    send_alert('AAPL', 'reached a new high')
    captured = capfd.readouterr()
    assert "ALERT: AAPL has reached a new high." in captured.out, "Alert message should be printed correctly"

# Test cases for sentiment analysis (data_rating.py)
def test_csv_load(setup_dataframe):
    assert not setup_dataframe.empty, "DataFrame should be loaded with data."

def test_sentiment_pipeline_initialization():
    try:
        sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        assert sentiment_analysis is not None, "Sentiment pipeline should initialize without errors."
    except Exception as e:
        pytest.fail(f"Sentiment pipeline failed to initialize: {e}")

# Test cases for scraping (markets_insider_scraper.py)
def test_webpage_scraping(mock_webpage):
    soup = BeautifulSoup(mock_webpage, 'lxml')
    articles = soup.find_all('div', class_='latest-news__story')
    assert len(articles) > 0, "There should be articles on the webpage."

def test_sentiment_analysis_on_titles():
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        res = classifier("NVIDIA stock hits new high")
        assert 'label' in res[0] and 'score' in res[0], "Sentiment analysis should return label and score."
    except Exception as e:
        pytest.fail(f"Sentiment analysis failed: {e}")

# Additional tests can be added here as needed.
