from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from lightweight_charts import Chart
import csv


app = Flask(__name__)

@app.route('/')
def index():
    # Read the CSV file containing sentiment data
    df_sentiment = pd.read_csv('../uOttawa-Capstone-Project/Web Scrapper/news_data_with_sentiment.csv')

    # Filter the data for the specific stock ticker (e.g., NVDA)
    nvda_rows = df_sentiment[df_sentiment['Stock'] == 'NVDA']

    # Create a list of tuples containing headline, content, and sentiment
    printed_data = [(headline, content, sentiment) for headline, content, sentiment in zip(nvda_rows['Headline'], nvda_rows['Content'], nvda_rows['Sentiment'])]

    # Pass the data to the HTML template
    return render_template('index.html', printed_data=printed_data)

if __name__ == '__main__':
    app.run(debug=True)
