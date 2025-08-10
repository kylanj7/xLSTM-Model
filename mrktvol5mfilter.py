import pandas as pd
import glob
import os
import shutil

# Define source and destination directories
source_dir = r"G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\Sorted_data\Cleaned_Data\Tech_Clean"
destination_dir = r"G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\Sorted_data\Cleaned_Data\Over_5M_Volume"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Get all CSV files in the source directory
csv_files = glob.glob(os.path.join(source_dir, "*.csv"))

for file_path in csv_files:
    # Read CSV file (pandas automatically skips header)
    df = pd.read_csv(file_path)
    
    # Calculate average of 3rd column (index 2)
    # Make sure there are at least 3 columns
    if df.shape[1] < 3:
        print(f"File {file_path} skipped: less than 3 columns")
        continue

    avg_val = df.iloc[:, 2].mean()
    
    # Check if average is over 5,000,000
    if avg_val > 5_000_000:
        # Move the file to the destination directory
        filename = os.path.basename(file_path)
        dest_path = os.path.join(destination_dir, filename)
        shutil.move(file_path, dest_path)
        print(f"Moved {filename} to {destination_dir} (average = {avg_val})")
    else:
        print(f"File {file_path} not moved (average = {avg_val})")
