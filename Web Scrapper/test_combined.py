import pytest
import pandas as pd
from transformers import pipeline

@pytest.fixture
def setup_dataframe():
    # Mocking a sample dataframe similar to the news_data.csv
    data = {
        'Content': ["Positive content", "Negative content", "Neutral content"]
    }
    return pd.DataFrame(data)

def test_csv_load(setup_dataframe):
    assert not setup_dataframe.empty, "DataFrame should be loaded with data."

def test_sentiment_pipeline_initialization():
    try:
        sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        assert sentiment_analysis is not None, "Sentiment pipeline should initialize without errors."
    except Exception as e:
        pytest.fail(f"Sentiment pipeline failed to initialize: {e}")

def test_sentiment_and_confidence_calculation(setup_dataframe):
    sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    def add_sentiment_confidence(row):
        content = row['Content']
        max_sequence_length = 512
        if len(content) > max_sequence_length:
            content = content[:max_sequence_length]
        result = sentiment_analysis(content)
        sentiment = result[0]['label']
        confidence = result[0]['score']
        row['Sentiment'] = sentiment
        row['Confidence'] = confidence
        return row
    
    modified_df = setup_dataframe.apply(add_sentiment_confidence, axis=1)
    assert 'Sentiment' in modified_df.columns and 'Confidence' in modified_df.columns, "Sentiment and Confidence columns should be added."

def test_csv_save():
    # This test checks if the DataFrame can be saved correctly
    try:
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_csv('test.csv', index=False)
        loaded_df = pd.read_csv('test.csv')
        assert not loaded_df.empty, "CSV should be saved and loaded correctly."
    except Exception as e:
        pytest.fail(f"Failed to save/load CSV: {e}")

import pytest
import pandas as pd
from transformers import pipeline

@pytest.fixture
def setup_dataframe():
    # Mocking a sample dataframe similar to the news_data.csv
    data = {
        'Content': ["Positive content", "Negative content", "Neutral content"]
    }
    return pd.DataFrame(data)

def test_csv_load(setup_dataframe):
    assert not setup_dataframe.empty, "DataFrame should be loaded with data."

def test_sentiment_pipeline_initialization():
    try:
        sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        assert sentiment_analysis is not None, "Sentiment pipeline should initialize without errors."
    except Exception as e:
        pytest.fail(f"Sentiment pipeline failed to initialize: {e}")

def test_sentiment_and_confidence_calculation(setup_dataframe):
    sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    def add_sentiment_confidence(row):
        content = row['Content']
        max_sequence_length = 512
        if len(content) > max_sequence_length:
            content = content[:max_sequence_length]
        result = sentiment_analysis(content)
        sentiment = result[0]['label']
        confidence = result[0]['score']
        row['Sentiment'] = sentiment
        row['Confidence'] = confidence
        return row
    
    modified_df = setup_dataframe.apply(add_sentiment_confidence, axis=1)
    assert 'Sentiment' in modified_df.columns and 'Confidence' in modified_df.columns, "Sentiment and Confidence columns should be added."

def test_csv_save():
    # This test checks if the DataFrame can be saved correctly
    try:
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_csv('test.csv', index=False)
        loaded_df = pd.read_csv('test.csv')
        assert not loaded_df.empty, "CSV should be saved and loaded correctly."
    except Exception as e:
        pytest.fail(f"Failed to save/load CSV: {e}")


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

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

@pytest.fixture
def setup_driver():
    driver = webdriver.Chrome()  # Ensure the correct driver path is set
    yield driver
    driver.quit()

def test_driver_initialization(setup_driver):
    assert setup_driver is not None, "WebDriver should be initialized successfully."

def test_url_loading(setup_driver):
    url = 'https://finviz.com/quote.ashx?t=NVDA&p=d'
    setup_driver.get(url)
    assert "NVDA" in setup_driver.title and "NVIDIA" in setup_driver.title, "Page should load correctly with NVDA and NVIDIA in the title."


def test_news_table_load(setup_driver):
    url = 'https://finviz.com/quote.ashx?t=NVDA&p=d'
    setup_driver.get(url)
    news_table = WebDriverWait(setup_driver, 10).until(
        EC.presence_of_element_located((By.ID, "news-table"))
    )
    assert news_table is not None, "News table should be located."

def test_headline_extraction(setup_driver):
    url = 'https://finviz.com/quote.ashx?t=NVDA&p=d'
    setup_driver.get(url)
    html = BeautifulSoup(setup_driver.page_source, 'html.parser')
    finviz_news_table = html.find(id='news-table')
    headline = finviz_news_table.find('tr').a.getText()
    assert len(headline) > 0, "Headline should not be empty."

# def test_article_content_load(setup_driver):
#     url = 'https://finviz.com/quote.ashx?t=NVDA&p=d'
#     setup_driver.get(url)
#     html = BeautifulSoup(setup_driver.page_source, 'html.parser')
#     finviz_news_table = html.find(id='news-table')
#     article_url = finviz_news_table.find('tr').a['href']
#     setup_driver.get(article_url)
#     try:
#         # Try a more generic approach using a common tag (like div or p)
#         article_content = WebDriverWait(setup_driver, 30).until(
#             EC.presence_of_element_located((By.TAG_NAME, "p"))
#         ).text
#         assert len(article_content) > 0, "Article content should not be empty."
#     except:
#         # Fall back to the original or alternative class approach if needed
#         try:
#             article_content = WebDriverWait(setup_driver, 30).until(
#                 EC.visibility_of_element_located((By.CLASS_NAME, "alternative-class"))
#             ).text
#             assert len(article_content) > 0, "Article content should not be empty."
#         except Exception as e:
#             pytest.fail(f"Article content could not be loaded: {e}")

