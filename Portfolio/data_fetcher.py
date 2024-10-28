from ib_insync import *

# Fetch S&P 500 tickers
def fetch_sp500_tickers(ib):
    sp500_etf = Stock('SPY', 'SMART', 'USD')
    contract_details = ib.reqContractDetails(sp500_etf)
    tickers = [detail.contract.symbol for detail in contract_details]
    return tickers

# Fetch NASDAQ-100 tickers
def fetch_nasdaq100_tickers(ib):
    nasdaq100_etf = Stock('QQQ', 'SMART', 'USD')
    contract_details = ib.reqContractDetails(nasdaq100_etf)
    tickers = [detail.contract.symbol for detail in contract_details]
    return tickers
