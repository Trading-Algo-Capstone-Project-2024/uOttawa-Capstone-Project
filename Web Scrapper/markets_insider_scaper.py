from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Function to perform sentiment analysis using the FinBERT model
def pipelineMethod(payload):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    res = classifier(payload)
    return res[0]



def scrape(ticker, numPages):
    # Define the columns for the DataFrame
    columns = ['datetime', 'date', 'title', 'source', 'link', 'top_sentiment', 'sentiment_score']
    data = []

    counter = 0
    
    
    # Scrape the data from the website
    for page in range(numPages):
        
        print(f"Scanning Page {page}")
        print()
        
        url = f'https://markets.businessinsider.com/news/{ticker}-stock?p={page}'
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'lxml')

        articles = soup.find_all('div', class_='latest-news__story')
        for article in articles:
            datetime_str = article.find('time', class_='latest-news__date').get('datetime')
            datetime_obj = pd.to_datetime(datetime_str)  # Convert to datetime object
            date = datetime_obj.date()  # Extract just the date
            
            title = article.find('a', class_='news-link').text.strip()
            source = article.find('span', class_='latest-news__source').text.strip()
            link = article.find('a', class_='news-link').get('href')

            # Perform sentiment analysis on the title
            output = pipelineMethod(title)
            top_sentiment = output['label']
            sentiment_score = output['score']
            
            # Collect the data in a list
            data.append([datetime_str, date, title, source, link, top_sentiment, sentiment_score])
            
            counter += 1

    print(f'{counter} headlines scraped from {numPages} pages')

    # Create a DataFrame and save it to CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'Web Scrapper/{ticker}sentiment.csv', index=False, encoding='utf-8')

scrape('AMD',5)
