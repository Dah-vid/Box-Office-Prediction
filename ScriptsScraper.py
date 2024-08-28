import requests
from bs4 import BeautifulSoup
import os

# Fetch and parse the HTML from the all genre page
genre_url = "https://imsdb.com/all-scripts.html"
response = requests.get(genre_url)
response.raise_for_status()  # Check if the request was successful
soup = BeautifulSoup(response.content, "html.parser")

# Extract intermediate URLs
# Assuming script links are within <p> tags with <a> tags inside them
script_links = soup.select("p a[href^='/Movie Scripts/']")
intermediate_urls = ["https://imsdb.com" + link['href'] for link in script_links]

# Base output directory
output_dir = "./scripts"
os.makedirs(output_dir, exist_ok=True)

for intermediate_url in intermediate_urls:
    try:
        # Step 3: Navigate to intermediate page to get final script URL
        response = requests.get(intermediate_url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the final script URL
        final_link = soup.select_one("a[href^='/scripts/']")
        if final_link is None:
            print(f"No final script link found on {intermediate_url}")
            continue
        final_url = "https://imsdb.com" + final_link['href']

        # Step 4: Fetch the final script page
        response = requests.get(final_url)
        response.raise_for_status()  # Check if the request was successful
        html_content = response.content

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the screenplay content within <pre> tags (or other relevant tags if different)
        screenplay = soup.find_all('pre')

        # Extract the screenplay text
        screenplay_text = "\n".join([block.get_text(separator="\n") for block in screenplay])

        # Create a filename based on the script title or URL
        script_title = final_url.split('/')[-1].replace(".html", "").replace("%20", "_")
        output_path = os.path.join(output_dir, f"{script_title}_Screenplay.txt")

        # Save the screenplay to a text file
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(screenplay_text)

        print(f"Screenplay for {script_title} extracted and saved to {output_path}")

    except Exception as e:
        print(f"Failed to scrape {intermediate_url}: {e}")
