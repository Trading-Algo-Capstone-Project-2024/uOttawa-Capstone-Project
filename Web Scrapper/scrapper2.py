import requests
from bs4 import BeautifulSoup

def scrape_yahoo_finance(stock_ticker):
    # Base URL of Yahoo Finance
    base_url = 'https://finance.yahoo.com/quote/'

    # Construct the URL using the stock ticker
    url = f'{base_url}{stock_ticker}'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the current price of the stock
        price_element = soup.find("fin-streamer", attrs={"data-test": "qsp-price"})
        if price_element:
            price = price_element.text.strip()
            print(f"Current price of {stock_ticker}: {price}")
        else:
            print("Failed to fetch current price.")

        # Extract URLs of articles
        article_links = []
        headlines = soup.find_all("h3", attrs={"class":"Mb(5px)"})
        for headline in headlines:
            link = headline.find('a')
            if link:
                article_links.append(link['href'])

        # Scrape content of each article
        for article_link in article_links:
            article_url = f'https://finance.yahoo.com{article_link}'
            article_response = requests.get(article_url)
            if article_response.status_code == 200:
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                article_content = article_soup.find('div', class_='caas-body').get_text()
                print(article_content)
            else:
                print(f"Failed to fetch article content. Status code: {article_response.status_code}")
    else:
        print("Failed to fetch data. Status code:", response.status_code)

# Example usage:
stock_ticker = 'NVDA'
scrape_yahoo_finance(stock_ticker)
