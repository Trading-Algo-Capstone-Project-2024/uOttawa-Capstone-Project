import pandas as pd
import csv

import pandas as pd
from transformers import pipeline

# Read the CSV file
df = pd.read_csv('../uOttawa-Capstone-Project/Web Scrapper/news_data.csv')

# Initialize sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", revision="main")

# Function to calculate sentiment and confidence and add them to DataFrame
def add_sentiment_confidence(row):
    content = row['Content']
    
    # Truncate content if it exceeds the maximum sequence length
    max_sequence_length = 512
    if len(content) > max_sequence_length:
        content = content[:max_sequence_length]
    
    result = sentiment_analysis(content)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    row['Sentiment'] = sentiment
    row['Confidence'] = confidence
    return row

# Apply the function to each row in the DataFrame
df = df.apply(add_sentiment_confidence, axis=1)

# Save the modified DataFrame back to the CSV file
df.to_csv('../uOttawa-Capstone-Project/Web Scrapper/news_data_with_sentiment.csv', index=False)
