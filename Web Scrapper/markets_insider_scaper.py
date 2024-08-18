from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def pipelineMethod(payload):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    res = classifier(payload)
    return res[0]

columns = ['datetime', 'title', 'source', 'link', 'top_sentiment', 'sentiment_score']
data = []

counter = 0

for page in range(1, 5):
    url = f'https://markets.businessinsider.com/news/nvda-stock?p={page}'
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'lxml')

    articles = soup.find_all('div', class_='latest-news__story')
    for article in articles:
        datetime = article.find('time', class_='latest-news__date').get('datetime')
        title = article.find('a', class_='news-link').text.strip()
        source = article.find('span', class_='latest-news__source').text.strip()
        link = article.find('a', class_='news-link').get('href')

        output = pipelineMethod(title)
        top_sentiment = output['label']
        sentiment_score = output['score']
        
        # Collect the data in a list
        data.append([datetime, title, source, link, top_sentiment, sentiment_score])
        
        counter += 1

        # Print the title safely to avoid encoding errors
        print(title.encode('utf-8', 'replace').decode('utf-8'))

print(f'{counter} headlines scraped from 4 pages')

# Create a DataFrame and save it to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('Web Scrapper/markets_insider_data.csv', index=False, encoding='utf-8')
