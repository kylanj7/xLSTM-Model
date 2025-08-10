📈 xLSTM Stock Price Forecasting & Backtesting

A modular Python project for stock market forecasting using xLSTM models, complete with data preparation, training, prediction, visualization, and trading backtests.
📂 Project Structure

project/
│
├── data_prep.py        # Data loading, cleaning, and feature engineering
├── model.py            # xLSTM model definition and architecture
├── train.py            # Training loop, loss calculation, model saving
├── predict.py          # Load trained model, make forecasts
├── plot_utils.py       # Visualization and charting functions
├── backtest.py         # Trading simulation & performance metrics
└── main.py             # Orchestrates training, prediction, plotting, backtests

🚀 Features

    xLSTM architecture for time-series forecasting

    Data scaling and feature preparation

    Configurable training and validation split

    Prediction & multi-step forecasting

    Visualization of actual vs. predicted prices

    Backtesting with simulated trades and metrics (Sharpe, drawdown, etc.)

    Modular design for easy customization

📦 Installation

# Clone the repository
git clone https://github.com/yourusername/xlstm-stock-forecast.git
cd xlstm-stock-forecast

# Install dependencies
pip install -r requirements.txt

🛠 Usage
1️⃣ Train a model

python main.py --mode train --data_path data/stock_data.csv

2️⃣ Make predictions

python main.py --mode predict --data_path data/stock_data.csv --model_path models/xlstm.pth

3️⃣ Plot results

python main.py --mode plot --data_path data/stock_data.csv --model_path models/xlstm.pth

4️⃣ Backtest strategy

python main.py --mode backtest --data_path data/stock_data.csv --model_path models/xlstm.pth

⚙ Configuration

You can adjust:

    Model hyperparameters (hidden layers, dropout, etc.) in model.py

    Training settings (epochs, batch size, learning rate) in train.py

    Forecasting window size in predict.py

    Plot styling in plot_utils.py

    Backtesting parameters in backtest.py

📊 Example Output

    Price chart: Actual vs predicted stock prices

    Trading signals: Buy/Sell markers

    Performance metrics: Accuracy, MAPE, Sharpe ratio, drawdown

📜 License

MIT License — feel free to use, modify, and share.