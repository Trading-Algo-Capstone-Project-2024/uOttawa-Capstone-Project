import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'Web Scrapper/markets_insider_data.csv' 
df = pd.read_csv(file_path)

# Ensure the 'datetime' column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime']).dt.date  # Convert to just the date, not time

# Group by the date and calculate the average sentiment score for each day
daily_sentiment = df.groupby('datetime')['sentiment_score'].mean().reset_index()

# Plot the average sentiment score for each day
plt.figure(figsize=(10, 6))
sns.lineplot(data=daily_sentiment, x='datetime', y='sentiment_score')

# Add titles and labels
plt.title('Average Sentiment Score by Date')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()
