import yfinance as yf

# Define the ticker symbol for Nvidia
ticker_symbol = "NVDA"

# Fetch historical stock price data for Nvidia from Yahoo Finance
nvidia_data = yf.download(ticker_symbol, start="2021-01-01", end="2022-01-01")

# Calculate daily returns of Nvidia stock
nvidia_data['Daily Return'] = nvidia_data['Adj Close'].pct_change()

# Calculate average daily return
avg_daily_return = nvidia_data['Daily Return'].mean()

# Fetch historical data for S&P 500 index (or any other market index)
market_data = yf.download("^GSPC", start="2021-01-01", end="2022-01-01")

# Calculate daily returns of the market index
market_data['Daily Return'] = market_data['Adj Close'].pct_change()

# Calculate average daily return of the market index
avg_market_return = market_data['Daily Return'].mean()

# Assume a risk-free rate (e.g., 10-year Treasury yield)
risk_free_rate = 0.02

# Calculate beta (covariance of stock returns with market returns divided by variance of market returns)
covariance = nvidia_data['Daily Return'].cov(market_data['Daily Return'])
variance = market_data['Daily Return'].var()
beta = covariance / variance

# Calculate expected return using CAPM
expected_return = risk_free_rate + beta * (avg_market_return - risk_free_rate)

print(f"Expected return of Nvidia (CAPM): {expected_return:.2%}")
