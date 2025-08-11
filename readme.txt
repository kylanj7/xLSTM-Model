# 📈 xLSTM Stock Price Forecasting & Backtesting

⚠️ **DEVELOPMENT PHASE - NOT PRODUCTION READY** ⚠️

A comprehensive Python project for stock market forecasting using an enhanced LSTM (xLSTM) model with attention mechanisms, multi-scale convolutions, and backtesting capabilities.

> **Important Notice**: This model is currently in active development and experimental phases. It is **NOT** suitable for production deployment or real trading decisions. Use for research and educational purposes only.

## 🏗️ Project Structure

project/
├── main.py              # Main orchestration script
├── model.py             # Enhanced xLSTM architecture with attention & convolutions
├── data_prep.py         # Data preprocessing and scaling utilities
├── train.py             # Training loop with early stopping & model checkpointing
├── predict.py           # Model loading and prediction functions
├── plot_utils.py        # Visualization and charting utilities
├── backtest.py          # Trading simulation with performance metrics
├── automatetesting.py   # Automated testing framework with progress tracking
├── config.py            # Configuration and path management
├── change_log.txt       # Development history and optimizations
├── requirements.txt     # Python dependencies
├── HistoricalData/      # Data management and preprocessing pipeline
│   ├── bulkdownload.py  # Bulk stock data download script
│   ├── meta_data.csv    # Stock metadata and ticker information
│   ├── debug/           # Debug files and logs
│   ├── removesmalldata.py # Script to filter out low-volume data
│   ├── testtickers.py   # Ticker validation and testing
│   └── Sorted_data/     # Organized stock data by sector
│       ├── {Sector1}/   # Individual sector folders (e.g., Technology, Healthcare)
│       ├── {Sector2}/   # Each containing raw sorted stock data
│       ├── {SectorN}/   # ...
│       ├── cleandata.py # Data cleaning and preprocessing script
│       ├── failed_tickers.txt # Log of failed ticker downloads
│       ├── mrktvolfilter.py   # Market volume filtering utility
│       ├── removesmalldata.py # Data size filtering script
│       └── Cleaned_Data/      # Processed and cleaned datasets
│           ├── Tech_Clean/    # Cleaned technology sector data
│           ├── Healthcare_Clean/ # Cleaned healthcare sector data
│           ├── Finance_Clean/ # Cleaned finance sector data
│           └── {Sector}_Clean/ # Additional cleaned sector folders
├── Models/              # Auto-created directory for saved models
│   └── best_model_YYYYMMDD_HHMMSS.pth  # Timestamped model checkpoints
├── Plots/               # Auto-created directory for visualizations
│   ├── training_history_YYYYMMDD_HHMMSS.png
│   ├── predictions_vs_actual_YYYYMMDD_HHMMSS.png
│   ├── price_forecast_YYYYMMDD_HHMMSS.png
│   └── prediction_error_distribution_YYYYMMDD_HHMMSS.png
└── logs/                # Auto-created directory for performance tracking
    ├── perf_metrics.txt # Text-based performance logs
    └── perf_metrics.csv # CSV performance data for analysis

## 🚀 Key Features

### Model Architecture
- **Extended LSTM (xLSTM)** with attention mechanisms for temporal focus
- **Multi-scale convolutions** (3x3, 5x5, 7x7 kernels) for pattern detection
- **Skip connections** with gating mechanisms
- **Batch normalization** and dropout for regularization
- **Configurable architecture** (hidden size: 256, layers: 2, dropout: 0.3)

### Training & Optimization
- **Early stopping** with patience mechanism (15 epochs)
- **Gradient clipping** to prevent exploding gradients
- **Model checkpointing** with timestamp-based saving
- **Train/validation split** with shuffled indices
- **Optimized hyperparameters** (learning rate: 0.0001, epochs: 100)

### Data Processing
- **Robust data loading** from multiple CSV files
- **Missing value handling** with forward/backward fill
- **MinMax scaling** for normalized inputs
- **Sequence-based datasets** (60-day lookback window)

### Visualization & Analysis
- **Training history plots** with loss curves
- **Prediction vs actual** price comparisons
- **Forecast visualization** with confidence intervals
- **Error distribution analysis**
- **Timestamped plot saving** for experiment tracking

### Backtesting Engine
- **Signal generation** with configurable thresholds (1% default)
- **Portfolio simulation** with realistic trading mechanics
- **Performance metrics**: Total Return, Sharpe Ratio, Maximum Drawdown
- **Buy/sell signal tracking** for strategy analysis

### Automated Testing
- **Multi-run testing framework** with progress bars
- **Real-time epoch monitoring** during training
- **Comprehensive logging** (TXT and CSV formats)
- **Performance metrics tracking** across multiple runs
- **Timeout handling** and error recovery

## 📦 Installation

bash
# Clone the repository
git clone <your-repo-url>
cd xlstm-stock-forecast

# Install dependencies
pip install -r requirements.txt


### Dependencies
- PyTorch (for deep learning)
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (preprocessing)
- matplotlib (visualization)
- tqdm (progress bars)

## 🛠️ Usage

### Basic Training & Prediction
bash
# Run complete pipeline (train, predict, plot, backtest)
python main.py


### Automated Testing
bash
# Run multiple training sessions with performance tracking
python automatetesting.py


### Configuration
Modify `config.py` to adjust:
- Data directory paths
- Model save locations
- Plot output directories

## ⚙️ Model Configuration

Current optimized settings based on testing:

python
# Model Architecture
hidden_size = 256        # Increased from 128 for better capacity
num_layers = 2          # Reduced from 3 for better gradient flow
dropout = 0.3           # Increased from 0.2 for regularization

# Training Parameters
learning_rate = 0.0001  # Reduced from 0.001 for stability
epochs = 100            # Increased from 10 for convergence
patience = 15           # Early stopping patience
batch_size = 32         # Optimized batch size
sequence_length = 60    # Days of historical data


## 📊 Output Examples

### Training Metrics
- Real-time epoch progress with loss values
- Training/validation loss curves
- Model checkpoint saving with timestamps

### Prediction Results
- Price forecasts vs actual values
- Error distribution analysis
- Buy/sell signal visualization

### Backtesting Performance

Trading performance metrics:
  Total Return %: 15.42
  Sharpe Ratio: 1.23
  Max Drawdown %: 8.45


## 🔧 Recent Optimizations

Based on extensive testing and documented in `change_log.txt`:

1. **Learning Rate Reduction**: 0.001 → 0.0001 for financial data stability
2. **Architecture Rebalancing**: Wider (256 hidden) but shallower (2 layers) network
3. **Extended Training**: 10 → 100 epochs with early stopping
4. **Enhanced Regularization**: Increased dropout to 0.3
5. **Improved Data Pipeline**: Robust CSV loading and preprocessing

## 📁 Data Requirements

- CSV files with stock price data
- Required columns: 'Close/Last' (others auto-detected)
- Default path: Configurable in `config.py`
- Supports multiple file concatenation

## 🎯 Performance Monitoring

The automated testing framework tracks:
- **Training convergence** (loss curves)
- **Prediction accuracy** (MAPE, directional accuracy)
- **Trading performance** (returns, risk metrics)
- **Execution time** and resource usage

## 📜 License

MIT License - Feel free to use, modify, and distribute.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

*This project represents an advanced implementation of LSTM-based financial forecasting with production-ready features for research and trading strategy development.*

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** 

- The model is in **active development** and should not be used for actual trading decisions
- Past performance does not guarantee future results
- Stock market predictions are inherently uncertain and risky
- Always consult with financial professionals before making investment decisions
- The authors assume no responsibility for any financial losses incurred through use of this software


