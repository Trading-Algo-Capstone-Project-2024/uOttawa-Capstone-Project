import pandas as pd
import os

CSV_FILE_NAME = 'portfolio_tracking.csv'

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
