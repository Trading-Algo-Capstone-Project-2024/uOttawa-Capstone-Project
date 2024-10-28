import pytest
import os
import pandas as pd
from ib_insync import IB
from data_fetcher import fetch_sp500_tickers, fetch_nasdaq100_tickers
from portfolio_manager import initialize_portfolio, rebalance_portfolio, manage_portfolio
from tracking import track_performance
from alerts import send_alert

# Set up fixtures
@pytest.fixture
def ib_connection():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # Adjust the port and clientId as necessary
    yield ib
    ib.disconnect()

@pytest.fixture
def mock_portfolio():
    return {'AAPL': 1000, 'MSFT': 1500}

@pytest.fixture
def mock_current_prices():
    return {'AAPL': 120, 'MSFT': 200}

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

def test_rebalance_portfolio(monkeypatch):
    portfolio = {'AAPL': 1000, 'MSFT': 1000}
    current_prices = {'AAPL': 1100, 'MSFT': 950}

    def mock_send_alert(ticker, condition):
        print(f"Mock Alert: {ticker} - {condition}")

    monkeypatch.setattr('alerts.send_alert', mock_send_alert)
    rebalance_portfolio(portfolio, current_prices)

def test_manage_portfolio(ib_connection):
    portfolio = manage_portfolio(ib_connection)
    assert isinstance(portfolio, dict), "Final portfolio should be a dictionary"
    assert len(portfolio) > 0, "Portfolio should contain tickers"

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

# Test cases for main.py
def test_ib_connection(ib_connection):
    assert ib_connection.isConnected(), "IBKR should be connected successfully"
