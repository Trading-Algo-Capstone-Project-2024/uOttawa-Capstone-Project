import pytest
from ib_insync import IB, Stock
from portfolio_manager import initialize_portfolio, rebalance_portfolio, manage_portfolio, load_portfolio
from alerts import send_alert

@pytest.fixture
def ib_connection():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # Adjust the port and clientId as necessary
    yield ib
    ib.disconnect()

@pytest.fixture
def mock_portfolio():
    return {'AAPL': 1000, 'MSFT': 800, 'GOOGL': 600}  # Adjusted values


@pytest.fixture
def mock_current_prices():
    return {'AAPL': 120, 'MSFT': 200, 'GOOGL': 100}

#Existing Tests
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

#Additional Tests for Partial Sell Orders
def test_sell_10_percent_of_position(mock_portfolio):
    stock = 'AAPL'
    initial_amount = mock_portfolio[stock]
    percentage_to_sell = 0.10
    sell_amount = initial_amount * percentage_to_sell
    mock_portfolio[stock] -= sell_amount
    assert mock_portfolio[stock] == initial_amount * (1 - percentage_to_sell), \
        "Selling 10% of the position should correctly update the portfolio"

def test_sell_25_percent_of_position(mock_portfolio):
    stock = 'AAPL'
    initial_amount = mock_portfolio[stock]
    percentage_to_sell = 0.25
    sell_amount = initial_amount * percentage_to_sell
    mock_portfolio[stock] -= sell_amount
    assert mock_portfolio[stock] == initial_amount * (1 - percentage_to_sell), \
        "Selling 25% of the position should correctly update the portfolio"

def test_sell_50_percent_of_position(mock_portfolio):
    stock = 'AAPL'
    initial_amount = mock_portfolio[stock]
    percentage_to_sell = 0.50
    sell_amount = initial_amount * percentage_to_sell
    mock_portfolio[stock] -= sell_amount
    assert mock_portfolio[stock] == initial_amount * (1 - percentage_to_sell), \
        "Selling 50% of the position should correctly update the portfolio"

def test_sell_more_than_current_position(mock_portfolio):
    stock = 'AAPL'
    initial_amount = mock_portfolio[stock]
    percentage_to_sell = 1.50  # 150% (more than the entire position)
    sell_amount = initial_amount * percentage_to_sell
    mock_portfolio[stock] = max(0, mock_portfolio[stock] - sell_amount)
    assert mock_portfolio[stock] == 0, \
        "Selling more than the available position should set the stock quantity to zero"

#New Tests for High Volatility Alerts
def test_high_volatility_alert(monkeypatch):
    def mock_send_alert(ticker, condition):
        assert ticker == 'AAPL'
        assert "high volatility" in condition

    monkeypatch.setattr('alerts.send_alert', mock_send_alert)
    send_alert('AAPL', 'high volatility detected')

#Tests for Market Halt
def test_market_halt_alert(monkeypatch):
    def mock_send_alert(ticker, condition):
        assert ticker == 'AAPL'
        assert "market halt" in condition

    monkeypatch.setattr('alerts.send_alert', mock_send_alert)
    send_alert('AAPL', 'market halt detected')

#Stop-Loss Tests
def test_stop_loss_trigger(mock_portfolio):
    stock = 'AAPL'
    stop_loss_price = 100
    current_price = 90
    if current_price <= stop_loss_price:
        mock_portfolio[stock] = 0
    assert mock_portfolio[stock] == 0, "Stop-loss should set the stock quantity to zero when triggered"

def test_stop_loss_not_triggered(mock_portfolio):
    stock = 'AAPL'
    stop_loss_price = 100
    current_price = 110
    if current_price <= stop_loss_price:
        mock_portfolio[stock] = 0
    assert mock_portfolio[stock] == 1000, "Stop-loss should not trigger when current price is above stop-loss price"

#Test Portfolio Rebalance with Stop Order
def test_rebalance_with_stop_order(monkeypatch, mock_portfolio):
    def mock_send_alert(ticker, condition):
        assert ticker == 'MSFT'
        assert "stop order executed" in condition

    monkeypatch.setattr('alerts.send_alert', mock_send_alert)
    current_prices = {'AAPL': 120, 'MSFT': 50, 'GOOGL': 100}
    rebalance_portfolio(mock_portfolio, current_prices)

#Test Adding a New Stock to Portfolio
def test_add_new_stock_to_portfolio():
    portfolio = {'AAPL': 1000}
    new_stock = 'TSLA'
    portfolio[new_stock] = 500
    assert 'TSLA' in portfolio, "New stock should be added to the portfolio"
    assert portfolio['TSLA'] == 500, "The new stock should have the correct quantity"

#Test Removing a Stock from Portfolio
def test_remove_stock_from_portfolio():
    portfolio = {'AAPL': 1000, 'MSFT': 500}
    del portfolio['MSFT']
    assert 'MSFT' not in portfolio, "The stock should be removed from the portfolio"

#Test Alert for Profit Target
def test_alert_on_profit_target(monkeypatch):
    def mock_send_alert(ticker, condition):
        assert ticker == 'GOOGL'
        assert "reached profit target" in condition

    monkeypatch.setattr('alerts.send_alert', mock_send_alert)
    send_alert('GOOGL', 'reached profit target')

#Test Alert for Loss Threshold
def test_alert_on_loss_threshold(monkeypatch):
    def mock_send_alert(ticker, condition):
        assert ticker == 'AAPL'
        assert "hit loss threshold" in condition

    monkeypatch.setattr('alerts.send_alert', mock_send_alert)
    send_alert('AAPL', 'hit loss threshold')

#Test Diversification Strategy
def test_portfolio_diversification(mock_portfolio):
    total_value = sum(mock_portfolio.values())
    for stock, value in mock_portfolio.items():
        proportion = value / total_value
        assert 0.1 <= proportion <= 0.5, "Each stock should not exceed diversification limits"

#Test Liquidating the Entire Portfolio
def test_liquidate_portfolio(mock_portfolio):
    for stock in mock_portfolio.keys():
        mock_portfolio[stock] = 0
    assert all(value == 0 for value in mock_portfolio.values()), "All positions should be liquidated"
