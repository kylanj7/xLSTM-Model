import numpy as np

def generate_signals(pred_prices, actual_prices, threshold=0.01):
    signals = []
    for i in range(1, len(pred_prices)):
        change = (pred_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
        if change > threshold:
            signals.append(1)
        elif change < -threshold:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)

def simulate_trading(pred_prices, actual_prices, initial_cash=10000, threshold=0.01):
    cash = initial_cash
    position = 0
    portfolio_value = []
    signals = generate_signals(pred_prices, actual_prices, threshold)

    for i in range(1, len(pred_prices)):
        signal = signals[i - 1]
        price = actual_prices[i]
        if signal == 1 and cash >= price:
            shares = int(cash // price)
            cash -= shares * price
            position += shares
        elif signal == -1 and position > 0:
            cash += position * price
            position = 0
        portfolio_value.append(cash + position * price)
    return np.array(portfolio_value), signals

def trading_metrics(portfolio_value):
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) else 0
    total = (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] * 100
    max_dd = np.max((np.maximum.accumulate(portfolio_value) - portfolio_value) / np.maximum.accumulate(portfolio_value)) * 100
    return {"Total Return %": total, "Sharpe Ratio": sharpe, "Max Drawdown %": max_dd}

