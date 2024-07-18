from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

# URL of the website
url = "https://www.the-numbers.com/movie/budgets/all"

# Set up the Selenium WebDriver
driver = webdriver.Chrome()  # Make sure the chromedriver is in your PATH
driver.get(url)

# Wait for the page to fully load
driver.implicitly_wait(10)  # Waits for up to 10 seconds

# Get the page source
html_content = driver.page_source

# Close the WebDriver
driver.quit()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Find the table containing the movie budgets
table = soup.find("table")

# Check if the table exists
if table:
    # Extract table headers
    headers = []
    for th in table.find_all("th"):
        headers.append(th.text.strip())

    # Extract table rows
    rows = []
    for tr in table.find_all("tr")[1:]:  # Skip the header row
        cells = tr.find_all("td")
        row = [cell.text.strip() for cell in cells]
        rows.append(row)

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Display the DataFrame
    print(df.head())

    # Save the DataFrame to a CSV file
    df.to_csv("movie_budgets.csv", index=False)
else:
    print("Table not found on the webpage.")
