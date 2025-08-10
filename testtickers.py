
import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOGL', 'A', 'AA']  # Mix of working and failing tickers
for t in tickers:
    try:
        data = yf.download(t, start='2018-01-01', end='2023-12-31')
        if data.empty:
            print(f"No data for {t}")
        else:
            print(f"Success: {t} data downloaded")
    except Exception as e:
        print(f"Error downloading {t}: {e}")
