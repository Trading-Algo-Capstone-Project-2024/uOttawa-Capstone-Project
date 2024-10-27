import pytest
import pandas as pd
from bs4 import BeautifulSoup
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@pytest.fixture
def mock_webpage():
    url = 'https://markets.businessinsider.com/news/nvda-stock?p=1'
    response = requests.get(url)
    return response.text

def test_webpage_scraping(mock_webpage):
    soup = BeautifulSoup(mock_webpage, 'lxml')
    articles = soup.find_all('div', class_='latest-news__story')
    assert len(articles) > 0, "There should be articles on the webpage."

def test_sentiment_analysis_on_titles():
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        res = classifier("NVIDIA stock hits new high")
        assert 'label' in res[0] and 'score' in res[0], "Sentiment analysis should return label and score."
    except Exception as e:
        pytest.fail(f"Sentiment analysis failed: {e}")

def test_dataframe_creation():
    data = {
        'datetime': ['2023-10-25T10:00:00Z'],
        'date': ['2023-10-25'],
        'title': ['NVIDIA hits new record'],
        'source': ['Market Insider'],
        'link': ['https://markets.businessinsider.com/news/nvda-stock'],
        'top_sentiment': ['positive'],
        'sentiment_score': [0.85]
    }
    df = pd.DataFrame(data)
    assert not df.empty, "DataFrame should be created with data."
    assert 'top_sentiment' in df.columns, "DataFrame should contain the sentiment column."

def test_csv_output():
    try:
        data = {
            'datetime': ['2023-10-25T10:00:00Z'],
            'date': ['2023-10-25'],
            'title': ['NVIDIA hits new record'],
            'source': ['Market Insider'],
            'link': ['https://markets.businessinsider.com/news/nvda-stock'],
            'top_sentiment': ['positive'],
            'sentiment_score': [0.85]
        }
        df = pd.DataFrame(data)
        df.to_csv('markets_insider_data.csv', index=False)
        loaded_df = pd.read_csv('markets_insider_data.csv')
        assert not loaded_df.empty, "CSV should be saved and loaded correctly."
    except Exception as e:
        pytest.fail(f"Failed to save/load CSV: {e}")
