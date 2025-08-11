import os
import matplotlib.pyplot as plt
from config import PLOTS_DIR, get_timestamp

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), 'training_history')
    plt.close()

def plot_predictions_vs_actual(predicted, actual, last_n=100):
    plt.figure(figsize=(10, 6))
    plt.plot(actual[-last_n:], label='Actual', color='steelblue', linewidth=2)
    plt.plot(predicted[-last_n:], label='Predicted', color='orange', linewidth=2)
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(f'Predictions vs Actual (Last {last_n} Days)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), 'predictions_vs_actual')
    plt.close()

def plot_forecast(historical_prices, forecast_prices, historical_days=50, forecast_days=30):
    import numpy as np
    plt.figure(figsize=(10, 6))
    historical_x = np.arange(0, historical_days)
    forecast_x = np.arange(historical_days - 1, historical_days + forecast_days)

    plt.plot(historical_x, historical_prices, label='Historical', color='blue', linewidth=2)
    plt.plot(forecast_x, np.concatenate([[historical_prices[-1]], forecast_prices]),
             label='Forecast', color='red', linestyle='--', linewidth=2)
    plt.axvline(x=historical_days - 1, color='black', linestyle=':', alpha=0.7)
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(f'{forecast_days}-Day Price Forecast')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), 'price_forecast')
    plt.close()

def plot_error_distribution(errors):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Error ($)')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), 'prediction_error_distribution')
    plt.close()

def save_plot(fig, prefix='plot'):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    timestamp = get_timestamp()
    save_path = os.path.join(PLOTS_DIR, f'{prefix}_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
