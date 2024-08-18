from ib_insync import *
#we are using these API commands https://ib-insync.readthedocs.io/api.html


ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

stock = Stock('AMD', 'SMART', 'USD')


bars = ib.reqHistoricalData(
    stock, endDateTime='20240404 16:00:00 US/Eastern', durationStr='3 D',
    barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)
#the end time speicifies that we want data from 30 trading days ago, from date set, default date is today.
#can ctrl click to see what other options are shown for diff options (opens an ib.py file)


# convert to pandas dataframe (pandas needs to be installed):
df = util.df(bars)
print(df)


#If we end up expanding into it, we could get tick data 
# market_data = ib.reqMktData(stock, '', False, False)
# ib.sleep(2)
#this task is asynchronous, we can set this up to work with semaphores or after other functions

# def onPendingTicker(tickers):
#     print("pending ticker event received")
#     print(tickers)

# ib.pendingTickersEvent += onPendingTicker

# print(market_data)

#using ib.run() we can automatically set this file to loop itself forever.

#I need to make this be more digestible, probably in form of a XLSX or CSV file to display it in the console
# ib.run()
