import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import glob
import os
from data_prep import DataPreparer
from model import xLSTM
from train import train_model
from predict import predict
from plot_utils import plot_training_history, plot_predictions_vs_actual
from backtest import simulate_trading, trading_metrics
import numpy as np

# Import from config if it exists, otherwise use default paths
try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = r'G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\HistoricalData\Sorted_data\Cleaned_Data\Tech_Clean'

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

def main():
    # Print some diagnostic information
    print(f"Using PyTorch version: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load all CSVs
    print(f"Loading data from: {DATA_DIR}")
    all_files = glob.glob(f"{DATA_DIR}/*.csv")
    print(f"Found {len(all_files)} CSV files")
    
    if len(all_files) == 0:
        print(f"Error: No CSV files found in {DATA_DIR}")
        return
        
    df_list = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataframe shape: {full_df.shape}")

    # Check if data is empty
    if full_df.empty:
        print("Error: Combined dataframe is empty")
        return
        
    # Print column names for debugging
    print(f"Columns in dataframe: {full_df.columns.tolist()}")

    # Prepare data
    print("Preparing data...")
    preparer = DataPreparer()
    scaled_data, df_processed = preparer.prepare_data(full_df)
    print(f"Scaled data shape: {scaled_data.shape}")

    # Find index of 'Close/Last' column
    try:
        close_idx = df_processed.columns.get_loc('Close/Last')
        print(f"Close price column index: {close_idx}")
    except KeyError:
        print("Error: 'Close/Last' column not found in DataFrame")
        print(f"Available columns: {df_processed.columns.tolist()}")
        return

    # Set up dataset and dataloaders
    seq_length = 60
    print(f"Using sequence length: {seq_length}")
    train_size = int(0.8 * len(scaled_data))
    print(f"Train/test split at index: {train_size}")
    
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - seq_length:]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Create full training dataset
    full_train_dataset = StockDataset(train_data, seq_length)
    
    # Manually split into train and validation sets
    train_dataset_size = int(0.9 * len(full_train_dataset))
    val_dataset_size = len(full_train_dataset) - train_dataset_size
    print(f"Training dataset size: {train_dataset_size}")
    print(f"Validation dataset size: {val_dataset_size}")
    
    # Generate indices and shuffle
    indices = np.arange(len(full_train_dataset))
    np.random.shuffle(indices)
    train_indices = indices[:train_dataset_size]
    val_indices = indices[train_dataset_size:]
    
    # Create subsets using Subset
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Shuffle done above
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(StockDataset(test_data, seq_length), batch_size=32)

    # Create model
    print("Creating model...")
    input_size = scaled_data.shape[1]
    print(f"Input size: {input_size}")
    model = xLSTM(input_size=input_size, hidden_size=128, num_layers=3, output_size=input_size)
    model.to(device)

    # Train
    print("Starting training...")
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    plot_training_history(train_losses, val_losses)

    # Predict
    print("Making predictions...")
    predictions, actuals = predict(model, test_loader, device)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Actuals shape: {actuals.shape}")
    
    # Inverse transform to get actual prices
    pred_prices = preparer.scaler.inverse_transform(predictions)[:, close_idx]
    actual_prices = preparer.scaler.inverse_transform(actuals)[:, close_idx]
    
    print(f"Predicted prices shape: {pred_prices.shape}")
    print(f"Actual prices shape: {actual_prices.shape}")
    
    # Plot results
    plot_predictions_vs_actual(pred_prices, actual_prices)

    # Backtest
    print("Running backtest...")
    portfolio_value, _ = simulate_trading(pred_prices, actual_prices)
    metrics = trading_metrics(portfolio_value)
    print("Trading performance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
