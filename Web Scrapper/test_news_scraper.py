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


