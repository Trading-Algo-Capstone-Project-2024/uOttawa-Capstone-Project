import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# Define the stock ticker
stock = 'NVDA'
news = {}

# Define the base URL for TradingView
base_url = 'https://www.tradingview.com'

# Define the URL for scraping news from TradingView
url = f'{base_url}/symbols/NASDAQ-{stock}/news/'

# Initialize the Selenium WebDriver
driver = webdriver.Chrome()  # Ensure your ChromeDriver path is set if necessary

# Send a request to the URL
driver.get(url)

# Simulate scrolling to load dynamic content
scroll_pause_time = 2
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to the bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Wait to load the page
    time.sleep(scroll_pause_time)
    
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Wait for the news articles to load
try:
    articles_div = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "list-iTt_Zp4a"))
    )
    print("Articles div found!")
except:
    print("Timeout waiting for news articles to load")
    driver.quit()

# Parse the HTML content
html = BeautifulSoup(driver.page_source, 'html.parser')
news[stock] = html.find('div', class_='list-iTt_Zp4a')

# Open the CSV file in append mode
with open('news_data_tradingview.csv', 'a', newline='', encoding='utf-8') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the header if it's a new file
    csv_writer.writerow(['Stock', 'Title', 'Content', 'URL'])
    
    # Counter to keep track of the number of articles scraped
    count = 0
    
    # Filter and store necessary information in news_parsed
    for row in news[stock].findAll('a', href=True):
        try:
            # Get the headline and URL of the news article
            title_div = row.find('div', class_='title-HY0D0owe title-DmjQR0Aa')
            headline = title_div.getText() if title_div else "No title found"
            article_url = base_url + row['href']
            
            # Print the headline to verify it's working
            print(f"Processing Article: {headline}")
            
            # Send a request to the article URL
            driver.get(article_url)
            
            # Wait for the article content to load
            try:
                article_content = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*=content], div[class*=body]"))
                ).text
            except:
                article_content = "Content not available"
            
            # Write the stock, headline, content, and URL to the CSV file
            csv_writer.writerow([stock, headline, article_content, article_url])
            
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
