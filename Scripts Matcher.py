import pandas as pd
import os
import re
##Code to match movie titles in box office csv that are in scripts folder
# Define the paths
csv_file_path = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\movie_budgets_and_revenues.csv"
scripts_folder_path = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\all scripts"
output_csv_file_path = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\matched_scripts.csv"
repeated_titles_file_path = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\repeated_movies.txt"

# Function to standardize the titles by removing special characters and converting to lower case
def standardize_title(title):
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)  # Remove special characters
    title = re.sub(r'\s+', ' ', title).strip()  # Replace multiple spaces with a single space
    return title.lower()

# Load the CSV file
movie_data = pd.read_csv(csv_file_path)

# Standardize the titles in the movie data CSV
movie_data['standardized_title'] = movie_data['Movie Name'].apply(standardize_title)

# Identify duplicated titles (those that appear more than once)
duplicated_titles = movie_data[movie_data.duplicated('standardized_title', keep=False)]['standardized_title'].unique()

# Filter out the duplicated titles from the movie data
unique_movie_data = movie_data[~movie_data['standardized_title'].isin(duplicated_titles)]

# List all the script files in the scripts folder
script_files = [f for f in os.listdir(scripts_folder_path) if f.endswith('.txt')]

# Standardize the titles in the script file names
script_file_titles = {standardize_title(f.replace('_Screenplay.txt', '').replace('-', ' ')): f for f in script_files}

# Create a DataFrame to map scripts to movie data
matched_scripts = []
for index, row in unique_movie_data.iterrows():
    standardized_title = row['standardized_title']
    if standardized_title in script_file_titles:
        matched_scripts.append({
            'Movie Name': row['Movie Name'],
            'Script File': script_file_titles[standardized_title],
            'Budget': row['Budget'],
            'Domestic Gross': row['Domestic Gross'],
            'Worldwide Gross': row['Worldwide Gross']
        })

# Convert to DataFrame
matched_scripts_df = pd.DataFrame(matched_scripts)

# Save the matched results to a CSV file
matched_scripts_df.to_csv(output_csv_file_path, index=False)

# Save the repeated titles to a .txt file
with open(repeated_titles_file_path, 'w') as file:
    for title in duplicated_titles:
        file.write(title + '\n')

print(f"Matched results saved to {output_csv_file_path}")
print(f"Repeated movie titles saved to {repeated_titles_file_path}")
