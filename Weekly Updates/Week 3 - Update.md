Team Members and their currentfuture roles
    Since we are only 2 members, the responsibilities will be shared 5050

    List of responsibilities
        - API call configuration
        - Data collection
        - Creating trading strategies
        - Market sentiment analysis using LLM
        - Backtesting strategies
        - Testing the code (QA)
        - Ensuring 100% coverage of test cases
        - Ensuring features reliability (ie. buying and selling, etc)


        Nicholas Turbide
        - English Communication

        Jean-Gabriel de Montigny
        - French Communication


Objectives
    Benefit to customer
        - Customer’s trading strategy setup
        - Market sentiment scanner
    Key objective
        - Making a profitable trading strategy
    Criteria for success
        - Having a successful trading strategy backtested over a period of one month
            - Criteria for a successful strategy
                - ((winning %  profit per win)(losing%  loss per loss)  0)

Expected architecture
    - Python logic base
    - Python API calls 
        - Using Interactive Brokers
    - Graphing TBD
        - Would probably be coded in R


Anticipated risks (engineering challenges)
    Market Movement Coverage
    - Make sure our codealgo still works in the case of trading halts
    Sentiment tracker accuracy
    - Holding sentiment over a time period
    Volatility comprehension
    - No bugs or crash during high volatility and volume
    - Getting accurate level 2 data during high volume periods
    Negative trading strategy
    - Strategy only works for a short timeframe
    - Profit ratioaccuracy rate of the strategy is too low
    Factor weight
    - Making sure each factor in the algorithm (market sentiment, indicators,market conditions, etc) is correctlyoptimally weighted


Legal and social issues
    - Tax return complexity (if real money is used)
        - Trading in other markets


Initial plans for first release
What will be delivered at the end of the first semester, what at the end of the second one


First Semester Releases
    Market Scanner
    - Market sentiment
    Data Collection
    Respective QA
    Code base for algorithm and API calls
    - Theory crafting of strategies

Second Semester Releases
    Backtested strategies
    Implemented strategies
    - Machine Learning module
    Code base and strategy optimisation
