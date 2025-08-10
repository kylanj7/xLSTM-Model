import os
import pandas as pd

# 📁 Paths
raw_data_dir = r"G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\Sorted_data\Energy"
clean_data_dir = r"G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\Sorted_data\Cleaned_Data\Energy_Clean"
os.makedirs(clean_data_dir, exist_ok=True)

# 🔢 Thresholds
MIN_FILE_SIZE_KB = 75
MIN_ROWS = 100

# 📄 Optional: Log skipped files
skipped_log = os.path.join(clean_data_dir, "skipped_files.txt")
log_entries = []

# 🚀 Clean each CSV
for filename in os.listdir(raw_data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(raw_data_dir, filename)

        # Skip files smaller than 75 KB
        if os.path.getsize(filepath) < MIN_FILE_SIZE_KB * 1024:
            print(f"⚠️ Skipped (too small): {filename}")
            log_entries.append(f"{filename} - too small")
            continue

        try:
            # Read CSV, skipping first 2 rows
            df = pd.read_csv(filepath, skiprows=2)

            # Skip if not enough data
            if df.shape[0] < MIN_ROWS:
                print(f"⚠️ Skipped (only {df.shape[0]} rows): {filename}")
                log_entries.append(f"{filename} - only {df.shape[0]} rows")
                continue

            # Rename columns
            df.columns = ['Date', 'Close/Last', 'High', 'Low', 'Open', 'Volume']

            # Convert Date column and sort
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')

            # Reorder to match model expectations
            df = df[['Date', 'Close/Last', 'Volume', 'Open', 'High', 'Low']]

            # Save cleaned file
            output_path = os.path.join(clean_data_dir, filename)
            df.to_csv(output_path, index=False)
            print(f"✅ Cleaned: {filename}")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            log_entries.append(f"{filename} - error: {str(e)}")

# 📝 Write skipped log
if log_entries:
    with open(skipped_log, "w") as f:
        for entry in log_entries:
            f.write(entry + "\n")
    print(f"\n🗒️ Skipped file log saved to: {skipped_log}")
