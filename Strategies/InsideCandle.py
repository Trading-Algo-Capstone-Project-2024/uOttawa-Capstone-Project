#We can use a library called Backtrader to get information on past market conditions
#We could graph this in other libraries as well

#This is a rough translation of the pinescript file of the same name, WIP.


import backtrader as bt

class InsideCandleStrategy(bt.Strategy):
    params = (
        ('timeframe', '1D'),
        ('rr_ratio', 2.00),
        ('show_moving_average', True),
        ('average_length', 100),
        ('upcolor', '#2962ff'),
        ('dncolor', '#e91e63'),
        ('line_width', 2),
    )

    def __init__(self):
        self.data_high1 = self.data.high(-1)
        self.data_low1 = self.data.low(-1)
        self.inside = bt.indicators.CrossOver(self.data.high, self.data_high1) + bt.indicators.CrossOver(self.data_low1, self.data.low)
        self.ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.average_length)
        self.bbw = bt.indicators.StdDev(self.data.close, period=self.params.average_length)
        self.ma2 = self.ma + self.bbw if self.ma > self.ma(-1) else self.ma - self.bbw

    def next(self):
        if self.inside and self.data.close > self.data_high1:
            self.buy()
            self.sell(exectype=bt.Order.Stop, price=self.data_low1)
            self.sell(exectype=bt.Order.Limit, price=self.data.close+(self.data.close-self.data_low1)*self.params.rr_ratio)

        if self.inside and self.data.close < self.data_low1:
            self.sell()
            self.buy(exectype=bt.Order.Stop, price=self.data_high1)
            self.buy(exectype=bt.Order.Limit, price=self.data.close-(self.data_high1-self.data.close)*self.params.rr_ratio)