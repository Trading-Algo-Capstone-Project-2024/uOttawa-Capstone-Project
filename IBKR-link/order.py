from ib_insync import *
#we are using these API commands https://ib-insync.readthedocs.io/api.html


ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)



stock = Stock('AMD', 'SMART', 'USD')

order = LimitOrder('BUY', 5, 200.00)

trade = ib.placeOrder(stock, order)

print(trade)

def orderFilled(trade, fill):
    print("order has been filled")
    print(order)
    print(fill)
    
trade.fillEvent += orderFilled

ib.sleep(3)

for order in ib.orders():
    print("== this is one of my orders ==")
    print(order)
    
for order in ib.trades():
    print("== this is one of my trades ==")
    print(trade)
    

ib.run()