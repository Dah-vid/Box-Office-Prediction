import os
import requests
from bs4 import BeautifulSoup
import pandas as pd


# Function to scrape a single page of data
def scrape_page(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check that the request was successful

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')

    # Initialize lists to store the data
    numbers = []
    release_dates = []
    movie_names = []
    budgets = []
    domestic_grosses = []
    worldwide_grosses = []

    # Iterate through the table rows
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) >= 5:
            numbers.append(cols[0].text.strip())  # Number
            release_dates.append(cols[1].text.strip())  # Release Date
            movie_names.append(cols[2].text.strip())  # Movie Name
            budgets.append(cols[3].text.strip())  # Budget
            domestic_grosses.append(cols[4].text.strip())  # Domestic Gross
            worldwide_grosses.append(cols[5].text.strip() if len(cols) > 5 else "")

    return numbers, release_dates, movie_names, budgets, domestic_grosses, worldwide_grosses


# Base URL of the website
base_url = "https://www.the-numbers.com/movie/budgets/all"

# Headers to mimic a request from a web browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Initialize lists to store the combined data
all_numbers = []
all_release_dates = []
all_movie_names = []
all_budgets = []
all_domestic_grosses = []
all_worldwide_grosses = []

# Iterate over the pages
for start in range(0, 7100, 100):  # Adjust the range as needed for 2100 entries
    url = base_url if start == 0 else f"{base_url}/{start}"
    print(f"Scraping URL: {url}")  # Print the URL being scraped for debugging

    numbers, release_dates, movie_names, budgets, domestic_grosses, worldwide_grosses = scrape_page(url, headers)

    print(f"Scraped {len(numbers)} records from {url}")  # Debugging output to track progress

    all_numbers.extend(numbers)
    all_release_dates.extend(release_dates)
    all_movie_names.extend(movie_names)
    all_budgets.extend(budgets)
    all_domestic_grosses.extend(domestic_grosses)
    all_worldwide_grosses.extend(worldwide_grosses)

# Create a DataFrame with the combined data and remove duplicates
data = {
    'Number': all_numbers,
    'Release Date': all_release_dates,
    'Movie Name': all_movie_names,
    'Budget': all_budgets,
    'Domestic Gross': all_domestic_grosses,
    'Worldwide Gross': all_worldwide_grosses
}
unique_data = pd.DataFrame(data).drop_duplicates()

# Print the total number of records and the DataFrame
print(f"Total records scraped: {len(unique_data)}")
print(unique_data)

# Get the path to the user's Documents folder
documents_path = os.path.expanduser("~/Documents")

# Save the DataFrame to a CSV file in the Documents folder
unique_data.to_csv(os.path.join(documents_path, 'movie_budgets_and_revenues.csv'), index=False)
