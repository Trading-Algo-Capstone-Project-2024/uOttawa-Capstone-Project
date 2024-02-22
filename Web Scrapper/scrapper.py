import requests
from bs4 import BeautifulSoup

# URL of the website you want to scrape
url = 'https://finance.yahoo.com/quote/NVDA'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Extract data by finding HTML elements
headlines = soup.find_all("h3", attrs={"class":"Mb(5px)"})

# Print the extracted data
for headline in headlines:
    print(headline.text)

