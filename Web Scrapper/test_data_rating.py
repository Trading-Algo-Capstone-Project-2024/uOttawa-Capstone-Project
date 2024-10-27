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
