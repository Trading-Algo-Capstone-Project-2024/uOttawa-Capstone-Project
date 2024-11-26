# uOttawa-Trading-Capstone-2024


Project description and goals


Core Stages

  Market scanner
  
    REQ: Article sentiment scanner
      - Use a LLM to analyze articles from breaking news around the world to predict if market sentiment is positive or negative before the markets open
      - Select a few stocks to keep track for the chosen timeframe using factors such as volatility, volume, price action and breaking news 
      - Apply the same LLM sentiment analysis process to the news related to the specific stock to determine market sentiment
          Side goal: make this work for any market
          
    REQ: Using Interactive Broker’s API we call to get real time market information (usually in the form of indicators and level 2 data)
      Obtain key information from the API such as :
      - Volume
      - Asks
      - Bids
      - Level 2 data
      - Specific markets indicator (RSI,EMA, etc - some of these we will need to calculate in house)
      - etc.
  Trading algorithm
  
    Using info pulled from yahoofinance and IBKR API (+ inhouse calculations)
    - Create strategies with a certain % of accuracy needed to make profit based on a return/losses ratio
    - If possible, we will create multiple strategies to adapt to market conditions. For example, we could have a strategy to go long (buy) and another one to go short (sell)
        - Side goal : show different time frames: 
            short - 1 day; medium - 1 week; long - 1 month+
    REQ: Using IBKR’s API we make calls to place orders (in paper trading)


  Machine Learning Stock Prediction

    Using data pulled from:
      - Yahoofinance's api
      - Our Market Scanner's sentiment score

    We organise all of the values into a stock specific .csv file to feed to our LSTM (Long Short Term Memory) Model.
    The model evaluates the patterns shown within the datasets and tries to predict what the stock prices will be in the near future.
    Through an iterative process of testing hyperparameters, we then optimize our algorithm to predict stock prices as close as possible to actual stock prices.
  
    We graph the best options to show off and consult on our future trades.
 





Student Names
Nicholas Turbide - 300175302 - nturb007@uottawa.ca
Jean-Gabriel de Montigny - 300164831 - jdemo037@uottawa.ca


Customer Name
- Adelphe Ekponon


Customer Affiliation
- Telfer School of Management, uOttawa


Customer email
- ekponon@telfer.uottawa.ca
