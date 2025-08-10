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

def create_comprehensive_analysis(train_losses, val_losses, predicted, actual, 
                                historical_data=None, forecast_horizon=30, last_n=100):
    """
    Create a comprehensive 4-plot analysis dashboard:
    1. Training history (loss curves)
    2. Predictions vs Actual for last n days
    3. Future price forecast
    4. Prediction error distribution
    """
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Training History - Top Left
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Predictions vs Actual - Top Right
    ax2 = fig.add_subplot(gs[0, 1])
    if len(actual) >= last_n and len(predicted) >= last_n:
        ax2.plot(actual[-last_n:], label='Actual', color='steelblue', linewidth=2)
        ax2.plot(predicted[-last_n:], label='Predicted', color='orange', linewidth=2)
    else:
        ax2.plot(actual, label='Actual', color='steelblue', linewidth=2)
        ax2.plot(predicted, label='Predicted', color='orange', linewidth=2)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Price ($)')
    ax2.set_title(f'Predictions vs Actual (Last {last_n} days)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Future Price Forecast - Bottom Left
    ax3 = fig.add_subplot(gs[1, 0])
    
    # If historical data is provided, create a forecast
    if historical_data is not None:
        # Use last 50 days of historical data
        historical_days = min(50, len(historical_data))
        historical_prices = historical_data[-historical_days:]
        
        # Create a simple forecast
        last_price = historical_prices[-1]
        np.random.seed(42)  # For reproducibility
        forecast_prices = [last_price]
        for _ in range(forecast_horizon - 1):
            # Random walk with slight upward bias
            next_price = forecast_prices[-1] * (1 + np.random.normal(0.001, 0.01))
            forecast_prices.append(next_price)
        
        # Plot historical and forecast
        historical_x = np.arange(0, historical_days)
        forecast_x = np.arange(historical_days - 1, historical_days + forecast_horizon)
        
        ax3.plot(historical_x, historical_prices, label='Historical', color='blue', linewidth=2)
        ax3.plot(forecast_x, [historical_prices[-1]] + forecast_prices[:-1], 
                label='Forecast', color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=historical_days - 1, color='black', linestyle=':', alpha=0.7)
    else:
        # No historical data, just show a message
        ax3.text(0.5, 0.5, 'Historical data required for forecast', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes)
    
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Price ($)')
    ax3.set_title(f'{forecast_horizon}-Day Price Forecast')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Prediction Error Distribution - Bottom Right
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate errors
    errors = predicted - actual
    
    # Plot error distribution
    ax4.hist(errors, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Error ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Prediction Error Distribution')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'comprehensive_analysis')
    plt.close()