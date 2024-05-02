#%%
from tqdm import tqdm
import pandas as pd

file = "Magazine_Subscriptions.jsonl"
output_file = "filtered_data.csv"

# Open the CSV file for writing
with open(output_file, 'w') as csv_file:
    # Write header to CSV file
    csv_file.write("user_id,asin,rating,timestamp\n")

    # Open the JSON lines file for reading
    with open(file, 'r') as json_file:
        # Get total number of lines in the file
        total_lines = sum(1 for _ in json_file)
        
        # Return to the start of the file
        json_file.seek(0)
        
        # Create a tqdm progress bar
        for line in tqdm(json_file, total=total_lines, desc="Processing"):
            # Read the JSON object from the line
            data = pd.read_json(line.strip(), typ='series')

            # Check if rating is >= 4
            if data['rating'] >= 4:
                # Write selected columns to CSV file
                csv_file.write(f"{data['user_id']},{data['asin']},{data['rating']},{data['timestamp']}\n")
# %%
