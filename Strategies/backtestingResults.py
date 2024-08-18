import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# Initialize the Selenium WebDriver
driver = webdriver.Chrome()  # Change this to the path of your web driver executable if necessary

# Define the URL for scraping data from TradingView
url = 'https://www.tradingview.com/chart/gIOCjNJ7/'

# Send a request to the URL
driver.get(url)

# Wait for the specific element to load
try:
    data_div = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '#bottom-area > div.bottom-widgetbar-content.backtesting > div > div.widgetContainer-_qqQIz_4'))
    )
except Exception as e:
    print(f"Timeout waiting for data div to load: {e}")
    driver.quit()

# Extract the data from the element
rows = data_div.find_elements(By.TAG_NAME, 'tr')
data = []
for row in rows:
    cols = row.find_elements(By.TAG_NAME, 'td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

# Quit the WebDriver
driver.quit()

# Create a DataFrame
df = pd.DataFrame(data)

# Check if data was successfully scraped
if not df.empty:
    # Export the data to an Excel spreadsheet
    df.to_excel('tradingview_data.xlsx', index=False)
    print("Data successfully exported to tradingview_data.xlsx")
else:
    print("No data to export.")
