from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Your Alpha Vantage API key
api_key = 'your_api_key'

# Initialize TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch daily stock data
data, meta_data = ts.get_daily(symbol='NVDA', outputsize='full')

# Convert the index to datetime format for easier filtering
data.index = pd.to_datetime(data.index)

# Filter the data from July 24th, 2024, onwards
start_date = '2024-07-24'
filtered_data = data.loc[start_date:]

# Add the "movement" column based on whether the stock closed higher or lower than it opened
filtered_data['movement'] = filtered_data.apply(lambda row: 'Up' if row['4. close'] > row['1. open'] else 'Down', axis=1)

# Display the filtered rows
print(filtered_data)

# Optional: Save the filtered data to a CSV file
filtered_data.to_csv('nvda_stock_data_july24_to_now.csv')
