import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd


# Define the stock ticker
stock = 'NVDA'
news = {}

# Define the URL for scraping news from Finviz
url = f'https://finviz.com/quote.ashx?t={stock}&p=d'

# Initialize the Selenium WebDriver
driver = webdriver.Chrome()  # Change this to the path of your web driver executable

# Send a request to the URL
driver.get(url)

# Wait for the news table to load
try:
    news_table = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "news-table"))
    )
except:
    print("Timeout waiting for news table to load")
    driver.quit()

# Parse the HTML content
html = BeautifulSoup(driver.page_source, 'html.parser')
finviz_news_table = html.find(id='news-table')
news[stock] = finviz_news_table

# Open the CSV file in append mode
with open('news_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Counter to keep track of the number of articles scraped
    count = 0
    
    # Filter and store necessary information in news_parsed
    for stock, news_item in news.items():
        for row in news_item.findAll('tr'):
            try:
                # Get the headline and URL of the news article
                headline = row.a.getText()
                article_url = row.a['href']
                
                # Send a request to the article URL
                driver.get(article_url)
                
                # Wait for the article content to load
                try:
                    article_content = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "caas-body"))
                    ).text
                except:
                    article_content = "Content not available"
                
                # Write the headline, URL, and content to the CSV file
                csv_writer.writerow([stock, headline,article_content, article_url ])
                
                # Increment the counter
                count += 1
                
                # Check if reached the limit of 10 articles
                if count == 10:
                    break
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Check if reached the limit of 10 articles
        if count == 10:
            break

# Quit the WebDriver
driver.quit()
