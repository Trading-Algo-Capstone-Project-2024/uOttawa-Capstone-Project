from ib_insync import *
from portfolio_manager import manage_portfolio

# Connect to IBKR TWS
ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=1)  # Make sure the port matches your working script
    print("Connection successful!")
except Exception as e:
    print("Connection failed:", e)
    exit(1)  # Exit if connection fails

# Run the portfolio management
final_portfolio = manage_portfolio(ib)
print("Final Portfolio:")
print(final_portfolio)

# Disconnect from IBKR
ib.disconnect()
