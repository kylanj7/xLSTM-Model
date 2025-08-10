import os
from datetime import datetime

# Base project folder (adjust if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder paths
DATA_DIR = r"G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\Sorted_data\Cleaned_Data\Tech_Clean"
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
PLOTS_DIR = os.path.join(BASE_DIR, 'Plots')

# Ensure folders exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')
