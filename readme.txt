# Stock Price Prediction Trading Bot

A comprehensive machine learning trading system that uses an extended LSTM (xLSTM) neural network to predict stock prices and execute automated trading strategies.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Backtesting](#backtesting)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Advanced Neural Network**: Custom xLSTM model with attention mechanisms and multi-scale convolutions
- **Automated Data Pipeline**: Bulk download and cleaning of stock data from Yahoo Finance
- **Comprehensive Backtesting**: Trading simulation with performance metrics
- **Real-time Predictions**: Future price forecasting capabilities
- **Visualization Suite**: Training metrics, predictions, and comprehensive analysis plots
- **Modular Design**: Clean separation of concerns across multiple modules
- **Early Stopping**: Prevents overfitting during model training
- **GPU Support**: CUDA acceleration for faster training

## 🏗️ Architecture

The system follows a modular architecture with clear separation of responsibilities:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   Model Layer   │    │ Analysis Layer  │
│                 │    │                 │    │                 │
│ • Data Download │    │ • xLSTM Model   │    │ • Backtesting   │
│ • Data Cleaning │───▶│ • Training      │───▶│ • Visualization │
│ • Preprocessing │    │ • Prediction    │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt shows TensorFlow, but the project actually uses PyTorch. You'll need:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib scipy yfinance tqdm
```

## ⚡ Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd trading-bot
   ```

2. **Configure data paths** in `config.py`:
   ```python
   DATA_DIR = "path/to/your/data"
   ```

3. **Download stock data**:
   ```bash
   python BulkDownload.py
   ```

4. **Clean the data**:
   ```bash
   python cleandata.py
   ```

5. **Run the main training and prediction pipeline**:
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
trading-bot/
├── core/
│   ├── config.py              # Configuration and paths
│   ├── data_prep.py           # Data preprocessing utilities
│   ├── model.py               # xLSTM neural network architecture
│   ├── train.py               # Model training logic
│   └── predict.py             # Prediction functionality
├── analysis/
│   ├── backtest.py            # Trading simulation and metrics
│   └── plot_utils.py          # Visualization utilities
├── data/
│   ├── BulkDownload.py        # Bulk stock data downloader
│   ├── cleandata.py           # Data cleaning pipeline
│   └── removesmalldata.py     # Data filtering utilities
├── main.py                    # Main execution script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## ⚙️ Configuration

### Data Paths

Update `config.py` with your data directories:

```python
DATA_DIR = "path/to/your/cleaned/data"
MODELS_DIR = "path/to/save/models"
PLOTS_DIR = "path/to/save/plots"
```

### Model Parameters

Key hyperparameters in `main.py`:

- `seq_length`: Number of time steps for sequence learning (default: 60)
- `hidden_size`: LSTM hidden dimension (default: 128)
- `num_layers`: Number of LSTM layers (default: 3)
- `batch_size`: Training batch size (default: 32)

## 📊 Data Pipeline

### 1. Data Download

```bash
python BulkDownload.py
```

Downloads historical stock data from Yahoo Finance for NASDAQ companies, organized by sector.

### 2. Data Cleaning

```bash
python cleandata.py
```

- Removes files smaller than 75KB
- Filters out stocks with less than 100 data points
- Standardizes column names and formats
- Sorts data chronologically

### 3. Volume Filtering

```bash
python removesmalldata.py
```

Removes stocks with average volume below 5M to focus on liquid securities.

## 🧠 Model Architecture

### xLSTM Features

- **Multi-layer LSTM**: Deep recurrent layers for temporal pattern learning
- **Attention Mechanism**: Focuses on relevant time steps
- **Multi-scale Convolutions**: Captures patterns at different time scales
- **Skip Connections**: Prevents vanishing gradients
- **Batch Normalization**: Stabilizes training
- **Dropout Regularization**: Prevents overfitting

### Architecture Diagram

```
Input Sequence (60 timesteps)
         ↓
    LSTM Layers (3x128 units)
         ↓
    Attention Mechanism
         ↓
    Multi-scale Conv1D (3,5,7 kernels)
         ↓
    Skip Connections + Gating
         ↓
    Dense Layers + Dropout
         ↓
    Output (Price Prediction)
```

## 📈 Backtesting

The backtesting module simulates trading performance:

### Trading Strategy

- **Buy Signal**: Predicted price increase > threshold (default: 1%)
- **Sell Signal**: Predicted price decrease > threshold (default: 1%)
- **Position Sizing**: Available cash / current price

### Performance Metrics

- **Total Return %**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

### Example Usage

```python
from backtest import simulate_trading, trading_metrics

portfolio_value, signals = simulate_trading(
    pred_prices, actual_prices, 
    initial_cash=10000, 
    threshold=0.01
)

metrics = trading_metrics(portfolio_value)
print(f"Total Return: {metrics['Total Return %']:.2f}%")
```

## 📊 Visualization

### Available Plots

1. **Training History**: Loss curves during training
2. **Predictions vs Actual**: Model accuracy visualization
3. **Price Forecasts**: Future price predictions
4. **Error Distribution**: Prediction error analysis
5. **Comprehensive Dashboard**: 4-panel analysis view

### Example

```python
from plot_utils import create_comprehensive_analysis

create_comprehensive_analysis(
    train_losses, val_losses, 
    predicted_prices, actual_prices,
    historical_data, forecast_horizon=30
)
```

## 💡 Usage Examples

### Training a New Model

```python
from model import xLSTM
from train import train_model

model = xLSTM(input_size=6, hidden_size=128, num_layers=3, output_size=6)
trained_model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, device, epochs=50
)
```

### Making Predictions

```python
from predict import predict

predictions, actuals = predict(model, test_loader, device)
```

### Running Backtest

```python
from backtest import simulate_trading, trading_metrics

portfolio_value, signals = simulate_trading(pred_prices, actual_prices)
metrics = trading_metrics(portfolio_value)
```

## 🔧 Advanced Configuration

### Custom Data Sources

Modify `BulkDownload.py` to use different:
- Time ranges
- Stock exchanges
- Data providers

### Model Customization

Adjust the xLSTM architecture in `model.py`:
- Add more attention heads
- Modify convolution kernel sizes
- Change dropout rates

### Trading Strategy

Customize trading logic in `backtest.py`:
- Implement stop-loss orders
- Add position sizing rules
- Include transaction costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the `License.txt` file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Stock trading involves risk, and past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions.

## 🆘 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review existing code examples

---

**Happy Trading! 📈**
