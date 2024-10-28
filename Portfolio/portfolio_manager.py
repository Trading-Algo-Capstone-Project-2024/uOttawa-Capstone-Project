import os
import pandas as pd
from ib_insync import IB, Stock  # Make sure to import Stock and any other necessary classes
from data_fetcher import fetch_sp500_tickers, fetch_nasdaq100_tickers
from alerts import send_alert

# Constants
STARTING_CAPITAL = 100000  # Starting capital
CSV_FILE_NAME = 'portfolio_tracking.csv'
TAKE_PROFIT_THRESHOLDS = [0.05, 0.1, 0.15]  # 5%, 10%, 15% gains
CUT_LOSS_THRESHOLDS = [-0.02, -0.05, -0.1]  # 2%, 5%, 10% losses

# Initialize Portfolio
def initialize_portfolio(capital, tickers):
    investment_per_stock = capital / len(tickers)
    portfolio = {ticker: investment_per_stock for ticker in tickers}
    return portfolio

# Load existing portfolio
def load_portfolio():
    if os.path.exists(CSV_FILE_NAME):
        df = pd.read_csv(CSV_FILE_NAME)
        portfolio = {row['Ticker']: row['Original Investment'] for _, row in df.iterrows()}
        return portfolio
    else:
        return {}

# Update portfolio based on performance
def rebalance_portfolio(portfolio, current_prices):
    for ticker, initial_amount in portfolio.items():
        current_value = initial_amount * current_prices[ticker]
        original_investment = initial_amount
        
        # Check profit thresholds
        for threshold in TAKE_PROFIT_THRESHOLDS:
            if current_value / original_investment >= (1 + threshold):
                sell_amount = 0.25 * original_investment
                portfolio[ticker] -= sell_amount
                send_alert(ticker, f"risen by {threshold * 100:.0f}% - Sold {sell_amount:.2f}.")
                break  # Exit loop after first threshold is triggered

        # Check loss thresholds
        for threshold in CUT_LOSS_THRESHOLDS:
            if current_value / original_investment < (1 + threshold):
                send_alert(ticker, f"dropped by {-threshold * 100:.0f}%. Consider cutting losses.")

        # Automatic rebalancing for movements <= 5%
        if abs(current_value / original_investment - 1) <= 0.05:
            # Automatically sell 10% of the position
            sell_amount = 0.10 * original_investment
            portfolio[ticker] -= sell_amount
            print(f'Automatically sold {sell_amount:.2f} of {ticker} due to <= 5% change.')

# Function to track performance and save to CSV
def track_performance(portfolio, current_prices):
    data = []
    
    for ticker, initial_amount in portfolio.items():
        current_value = initial_amount * current_prices[ticker]
        original_investment = initial_amount
        percent_change = (current_value - original_investment) / original_investment * 100
        
        # Store the data
        data.append({
            'Ticker': ticker,
            'Original Investment': original_investment,
            'Current Value': current_value,
            'Percent Change': percent_change,
            'Position Size': initial_amount
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Append data to CSV file
    if os.path.exists(CSV_FILE_NAME):
        df.to_csv(CSV_FILE_NAME, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE_NAME, index=False)

# Main function to manage portfolio
def manage_portfolio(ib):
    # Load existing portfolio or initialize a new one
    portfolio = load_portfolio()
    
    if not portfolio:  # If portfolio is empty, initialize
        sp500_tickers = fetch_sp500_tickers(ib)
        nasdaq_tickers = fetch_nasdaq100_tickers(ib)
        all_tickers = list(set(sp500_tickers) | set(nasdaq_tickers))
        portfolio = initialize_portfolio(STARTING_CAPITAL, all_tickers)

    # Simulate quarterly rebalancing
    for quarter in range(4):  # 4 quarters in a year
        current_prices = {}

        # Fetch current prices for each stock in the portfolio
        for ticker in portfolio.keys():
            contract = Stock(ticker, 'SMART', 'USD')
            historical_data = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='MIDPOINT',
                useRTH=True
            )
            # Take the most recent close price
            if historical_data:
                current_prices[ticker] = historical_data[-1].close

        # Update portfolio based on current prices
        rebalance_portfolio(portfolio, current_prices)
        
        # Track performance and save to CSV
        track_performance(portfolio, current_prices)
        
        # Reinvest capital in the top 10 performers
        performance = sorted(portfolio.items(), key=lambda x: x[1], reverse=True)[:10]
        reinvest_amount = sum([amount for ticker, amount in performance])
        
        # Reallocate to top performers
        if reinvest_amount > 0:
            reinvest_per_stock = reinvest_amount / 10  # Reinvest equally in top 10 performers
            for ticker, amount in performance:
                portfolio[ticker] += reinvest_per_stock
                print(f'Reinvested {reinvest_per_stock:.2f} into {ticker}.')

    return portfolio
