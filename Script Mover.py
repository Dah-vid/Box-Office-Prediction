import os
import shutil
import pandas as pd
#Code to move scripts that are in the matched csv a seperate folder
# Define the path to the matched scripts data CSV
matched_scripts_csv_path = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\matched_scripts.csv"

# Load the matched scripts data
matched_scripts_df = pd.read_csv(matched_scripts_csv_path)

# Define the source and destination directories
source_dir = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\all scripts"
destination_dir = r'C:\Users\david\Documents\MSC Coursework\Dissertation\Code\matched_scripts'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Initialize a list to store the names of files not found
not_found_files = []

# Loop through the matched scripts and move each file
for script_file in matched_scripts_df['Script File']:
    source_path = os.path.join(source_dir, script_file)
    destination_path = os.path.join(destination_dir, script_file)

    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f'Moved: {script_file}')
    else:
        print(f'File not found: {script_file}')
        not_found_files.append(script_file)

# Check how many files were not found
print(f'\nTotal files listed in CSV: {len(matched_scripts_df)}')
print(f'Total files moved: {len(matched_scripts_df) - len(not_found_files)}')
print(f'Total files not found: {len(not_found_files)}')

# If any files were not found, save their names to a text file
if not_found_files:
    not_found_file_path = os.path.join(destination_dir, 'not_found_files.txt')
    with open(not_found_file_path, 'w') as f:
        for file_name in not_found_files:
            f.write(f"{file_name}\n")
    print(f"List of files not found saved to: {not_found_file_path}")
