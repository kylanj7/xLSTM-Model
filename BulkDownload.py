import pandas as pd
import yfinance as yf
import os
import time

# Paths
METADATA_PATH = r'G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\nasdaq_meta_data.csv'
OUTPUT_DIR = r'G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\Sorted_data'

# Load metadata
df = pd.read_csv(METADATA_PATH)

# Clean tickers
df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
df = df.dropna(subset=['Symbol'])

# Ensure no duplicate tickers
tickers_info = df.drop_duplicates(subset=['Symbol'])

print(f"Total tickers to download: {len(tickers_info)}")

failed_tickers = []

for _, row in tickers_info.iterrows():
    ticker = row['Symbol']
    sector = row['Sector'] if pd.notna(row['Sector']) and row['Sector'] != '' else 'Unknown'

    try:
        print(f"Downloading {ticker} ({sector})...")
        data = yf.download(ticker, start='2018-01-01', end='2023-12-31', progress=False)

        if data.empty:
            print(f"⚠️ No data found for {ticker}")
            failed_tickers.append(ticker)
        else:
            # Create sector-specific subdirectory
            sector_dir = os.path.join(OUTPUT_DIR, sector.replace("/", "-"))  # Replace slashes to avoid path issues
            os.makedirs(sector_dir, exist_ok=True)

            # Save CSV
            file_path = os.path.join(sector_dir, f"{ticker}.csv")
            data.to_csv(file_path)

        time.sleep(1)  # Be polite with API
    except Exception as e:
        print(f"❌ Failed to download {ticker}: {e}")
        failed_tickers.append(ticker)

# Save failed tickers
with open(os.path.join(OUTPUT_DIR, "failed_tickers.txt"), 'w') as f:
    for t in failed_tickers:
        f.write(t + '\n')

print(f"\n✅ Download complete. Success: {len(tickers_info) - len(failed_tickers)}, Failed: {len(failed_tickers)}")
