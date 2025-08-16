# xLSTM Trading Bot 🚀

⚠️ **DEVELOPMENT PHASE - NOT PRODUCTION READY** ⚠️

An advanced AI-powered trading bot that combines a custom xLSTM (Extended Long Short-Term Memory) neural network with historical market data to execute automated trading strategies through the Alpaca API.

## 🎯 Features

- **Custom xLSTM Architecture**: Enhanced LSTM with attention mechanisms, multi-scale convolutions, and residual connections
- **Risk Management**: Stop-loss, take-profit, position sizing, and daily trade limits
- **Backtesting**: Historical performance analysis with detailed metrics
- **Model Performance Tracking**: Automated testing and model ranking system
- **Live Market Integration**: Real-time data fetching and market hours detection

## 🏗️ Project Structure

### Training System (`C:\users\user\Desktop\xLSTMtraining\`)
```
├── main.py                 # Main training pipeline orchestrator
├── config.py               # Training configuration and paths
├── model.py                # xLSTM neural network architecture
├── data_prep.py            # Data preprocessing and scaling
├── plot_utils.py           # Visualization and plotting tools
├── backtest.py             # Trading simulation and metrics
├── automatetesting.py      # Automated model testing framework
└── import os.py            # OS utilities and configuration
```

### Generated Assets
```
├── Models/                 # Trained model storage
│   └── best_model.pth      # Best performing model weights
├── Scalers/                # Data preprocessing artifacts
│   └── scaler.pkl          # Fitted data scalers
├── Plots/                  # Generated visualizations
│   ├── training_history_*.png
│   ├── predictions_vs_actual_*.png
│   └── combined_analysis_*.png
└── Logs/                   # Performance tracking
    ├── perf_metrics.csv    # Automated test results
    ├── perf_metrics.txt    # Detailed performance logs
    └── changelog.txt       # Version history and updates
```

### Data Pipeline
```
HistoricalData/
├── BulkDownload.py         # NASDAQ data downloader
├── debug.py                # Data debugging utilities
├── nasdaq_meta_data.csv    # Stock metadata and sector info
├── removesmalldata.py      # Data quality filters
├── testtickers.py          # Ticker validation tools
├── NASDAQ_5Y_Raw/          # Raw 5-year historical data
│   └── [ticker].csv        # Individual stock files
├── Sorted_data/            # Sector-organized data
│   ├── cleandata.py        # Data cleaning pipeline
│   ├── mrktvol5mfilter.py  # Volume-based filtering
│   ├── removesmalldata.py  # Size-based filtering
│   ├── failed_tickers.txt  # Processing error log
│   ├── Healthcare/         # Healthcare sector stocks
│   ├── Tech/               # Technology sector stocks
│   ├── Energy/             # Energy sector stocks
│   ├── {SectorN}/          # ...
│   └── Cleaned_Data/       # Post-processed datasets
│       ├── Healthcare_Clean/
│       ├── Tech_Clean/
│       ├── Energy_Clean/
│       └── {Sector}_Clean/ # Additional cleaned sector folders
│       └── Over_5M_Volume/ # High-volume filtered stocks
└── Fine_Tune_Data/        # Additional training data, user selected for fine tuning.
```

### Live Trading System (`C:\users\user\Desktop\PaperTrade\`)
```
├── alpaca_main.py          # Main trading bot orchestrator
├── alpaca_config.py        # API credentials and trading config
├── alpaca_model.py         # Model definitions and utilities
├── model_handler.py        # ML model loading and prediction
├── alpaca_client.py        # Alpaca API integration
├── trading_strategy.py     # Trading logic and risk management
├── requirements.txt        # Dependencies
├── Model/                  # Trained model storage
│   └── best_model_*.pth    # Production model weights
└── Scaler/                 # Data preprocessing artifacts
    └── scaler.pkl          # Production data scalers
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/xlstm-trading-bot.git
cd xlstm-trading-bot
pip install -r requirements.txt
```

### 2. Configuration

Update `alpaca_config.py` with your Alpaca API credentials:

```python
@dataclass
class TradingConfig:
    API_KEY: str = 'your_alpaca_api_key'
    SECRET_KEY: str = 'your_alpaca_secret_key'
    BASE_URL: str = 'https://paper-api.alpaca.markets'  # Paper trading
    
    # Model paths
    MODEL_PATH: str = r'C:\users\user\Desktop\PaperTrade\Model\best_model_*.pth'
    SCALER_PATH: str = r'C:\users\user\Desktop\PaperTrade\Scaler\scaler.pkl'
```

### 3. Prepare Training Data

```bash
# Navigate to training directory
cd C:\users\user\Desktop\xLSTMtraining\

# Download and organize NASDAQ data
python HistoricalData/BulkDownload.py

# Clean and filter data by sector
cd HistoricalData/Sorted_data
python cleandata.py
python mrktvol5mfilter.py
```

### 4. Train Models

```bash
# Train a single model
python main.py

# Automated model testing (trains multiple models and ranks them)
python automatetesting.py
```

### 5. Deploy Best Model

```bash
# Copy best model and scaler to trading bot directory
cp Models/best_model_*.pth C:\users\user\Desktop\PaperTrade\Model\
cp Scalers/scaler.pkl C:\users\user\Desktop\PaperTrade\Scaler\

# Navigate to trading directory and run
cd C:\users\user\Desktop\PaperTrade\
python alpaca_main.py
```

## 🧠 xLSTM Architecture

Our enhanced LSTM model features:

- **Layer Normalization**: Improved training stability
- **Attention Mechanism**: Focus on relevant time periods
- **Multi-scale Convolutions**: Capture patterns at different time scales
- **Residual Connections**: Better gradient flow and training
- **Feature Gating**: Intelligent feature selection

## 📊 Model Training

### Train a New Model

```bash
# Update DATA_DIR in config.py to point to your historical data
python main.py  # Training version
```

### Automated Model Testing

Run multiple training sessions and automatically rank models by performance:

```bash
python automatetesting.py
```

This will:
- Train multiple models with different random initializations
- Backtest each model on historical data
- Rank models by performance metrics
- Generate detailed performance logs

## 💹 Trading Strategy

### Risk Management Features

- **Position Sizing**: Maximum 10% of portfolio per trade
- **Stop Loss**: 5% automatic stop loss
- **Take Profit**: 10% take profit targets
- **Daily Limits**: Maximum trades per day
- **Account Protection**: Minimum balance requirements

### Signal Generation

The bot uses the xLSTM model to predict future price movements and generates trading signals based on:
- Prediction confidence thresholds
- Market conditions
- Risk parameters
- Position management rules

## 📈 Backtesting & Performance

### Key Metrics Tracked

- **Total Return %**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Holding period analysis

### Performance Analysis

```python
# Run backtest on historical data
python backtest.py

# Generate performance visualizations
python plot_utils.py
```

## 🔧 Configuration Options

### Trading Parameters

```python
MAX_POSITION_SIZE: float = 0.1      # 10% max position size
STOP_LOSS: float = 0.05             # 5% stop loss
TAKE_PROFIT: float = 0.10           # 10% take profit
MAX_DAILY_TRADES: int = 10          # Daily trade limit
CHECK_INTERVAL: int = 60            # Check every 60 seconds
```

### Model Parameters

```python
input_size = number_of_features
hidden_size = 128
num_layers = 3
dropout = 0.3
sequence_length = 60  # 60 time steps for prediction
```

## 📋 Requirements

```
alpaca-trade-api>=3.1.1
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
torch>=1.12.0
matplotlib>=3.4.0
```

## 🛠️ Data Processing Tools

### Bulk Data Download
- **BulkDownload.py**: Downloads 5-year historical data for all NASDAQ stocks
- **testtickers.py**: Validates ticker symbols and checks data availability
- **nasdaq_meta_data.csv**: Contains stock metadata and sector classifications

### Data Cleaning Pipeline
- **cleandata.py**: Standardizes CSV format, handles missing data, ensures proper column ordering
- **mrktvol5mfilter.py**: Filters stocks by average trading volume (>5M threshold)
- **removesmalldata.py**: Removes files below minimum size/quality thresholds
- **failed_tickers.txt**: Logs any tickers that failed during processing

### Quality Assurance
- Minimum file size: 75KB
- Minimum data points: 100 rows
- Volume threshold: 5 million average daily volume
- Automatic error logging and debugging utilities

- **Paper Trading Default**: Starts with paper trading to prevent real money loss
- **Market Hours Check**: Only trades during market hours
- **Connection Validation**: Verifies API connectivity before trading
- **Error Handling**: Comprehensive error logging and recovery
- **Account Monitoring**: Real-time balance and position tracking

## 📊 Data Requirements & Processing

### Raw Data Collection

The system downloads and processes NASDAQ 5-year historical data:

```bash
# Download raw NASDAQ data
python HistoricalData/BulkDownload.py

# Validate tickers and metadata
python HistoricalData/testtickers.py
```

### Data Processing Pipeline

1. **Sector Organization**: Raw data is automatically sorted by sector (Healthcare, Tech, Energy)
2. **Data Cleaning**: Standardizes column names, handles missing values, ensures date formatting
3. **Quality Filtering**: Removes files below minimum size/row thresholds
4. **Volume Filtering**: Isolates high-volume stocks (>5M average volume)

```bash
# Clean data by sector
cd HistoricalData/Sorted_data
python cleandata.py

# Filter by market volume
python mrktvol5mfilter.py

# Remove low-quality data
python removesmalldata.py
```

### Expected Data Format

The model expects CSV files with the following columns:
- `Date`: Timestamp (YYYY-MM-DD format)
- `Close/Last`: Closing price
- `Volume`: Trading volume
- `Open`: Opening price
- `High`: Highest price
- `Low`: Lowest price

### Data Quality Standards

- **Minimum File Size**: 75KB to ensure sufficient data
- **Minimum Rows**: 100+ data points for reliable training
- **Volume Threshold**: 5M+ average volume for liquidity
- **Sector Classification**: Organized by NASDAQ sector categories

## 🔄 Complete Workflow

### Training Phase
1. **Data Collection**: Download 5-year NASDAQ historical data using `BulkDownload.py`
2. **Data Organization**: Sort stocks by sector (Healthcare, Tech, Energy)
3. **Data Cleaning**: Process and standardize data format with quality filters
4. **Volume Filtering**: Select high-liquidity stocks (>5M average volume)
5. **Model Training**: Train xLSTM models on cleaned sector data
6. **Automated Testing**: Run multiple training sessions with `automatetesting.py`
7. **Model Selection**: Rank models by performance metrics and select best performer

### Deployment Phase
1. **Model Export**: Copy best model weights and scalers to `PaperTrade` directory
2. **Configuration**: Update `alpaca_config.py` with API credentials and model paths
3. **Paper Testing**: Validate strategy with paper trading using `alpaca_main.py`
4. **Live Deployment**: Execute real-time trading with risk management
5. **Performance Monitoring**: Track metrics and log performance

### Continuous Improvement
1. **Performance Analysis**: Review trading logs and metrics
2. **Model Retraining**: Periodically retrain on new market data
3. **Strategy Optimization**: Adjust risk parameters based on performance
4. **Data Updates**: Refresh historical data and sector classifications

## 🚨 Disclaimer

This trading bot is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk and consider paper trading before deploying with real money.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Alpaca Markets for their excellent trading API
- PyTorch team for the deep learning framework
- The quantitative finance community for research and insights

---

**⚠️ Remember**: Always test thoroughly with paper trading before using real money!

