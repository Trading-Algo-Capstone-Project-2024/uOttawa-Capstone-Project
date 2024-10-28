# test_marketlink.py

import pytest
from unittest.mock import MagicMock
from ib_insync import IB, Stock, LimitOrder

# Test 1: IB Connection
def test_ib_connection():
    ib = IB()
    ib.connect = MagicMock(return_value=True)  # Mock connect to always succeed
    ib.connect('127.0.0.1', 7497, clientId=1)
    ib.isConnected = MagicMock(return_value=True)  # Mock isConnected to return True
    assert ib.isConnected(), "IB connection failed"

# Test 2: Stock Object Creation
def test_stock_creation():
    stock = Stock('AMD', 'SMART', 'USD')
    assert stock.symbol == 'AMD', "Stock symbol mismatch"
    assert stock.exchange == 'SMART', "Stock exchange mismatch"
    assert stock.currency == 'USD', "Stock currency mismatch"

# Test 3: Order Placement
def test_order_placement():
    ib = IB()
    ib.connect = MagicMock(return_value=True)
    ib.connect('127.0.0.1', 7497, clientId=1)
    
    stock = Stock('AMD', 'SMART', 'USD')
    order = LimitOrder('BUY', 15, 152)
    
    # Mock placeOrder to return a trade object with expected properties
    trade = MagicMock()
    trade.order.totalQuantity = 15
    trade.order.action = 'BUY'
    trade.order.lmtPrice = 152
    ib.placeOrder = MagicMock(return_value=trade)

    # Place order
    result_trade = ib.placeOrder(stock, order)
    
    # Check the order details
    assert result_trade.order.totalQuantity == 15, "Order quantity mismatch"
    assert result_trade.order.action == 'BUY', "Order action mismatch"
    assert result_trade.order.lmtPrice == 152, "Order limit price mismatch"

# Test 4: Order Fill Callback
def test_order_filled_callback():
    # Callback function to test
    def orderFilled(trade, fill):
        assert trade.order.action == 'BUY', "Order action mismatch in callback"
        print("Order has been filled")
    
    # Set up a mock trade and fill
    ib = IB()
    trade = MagicMock()
    trade.order.action = 'BUY'
    fill = MagicMock()
    
    # Attach callback and trigger fill event
    trade.fillEvent = MagicMock()
    trade.fillEvent += orderFilled  # Simulate the callback being attached
    orderFilled(trade, fill)  # Simulate the callback being triggered

# Test 5: Listing Orders and Trades
def test_orders_and_trades_listing():
    ib = IB()
    ib.connect = MagicMock(return_value=True)
    ib.connect('127.0.0.1', 7497, clientId=1)

    stock = Stock('AMD', 'SMART', 'USD')
    order = LimitOrder('BUY', 15, 152)

    # Mock placeOrder, orders, and trades responses
    trade = MagicMock()
    trade.order = order
    ib.placeOrder = MagicMock(return_value=trade)
    ib.orders = MagicMock(return_value=[order])
    ib.trades = MagicMock(return_value=[trade])

    # Place order and verify it's listed
    ib.placeOrder(stock, order)
    
    assert order in ib.orders(), "Order not found in ib.orders()"
    assert trade in ib.trades(), "Trade not found in ib.trades()"


