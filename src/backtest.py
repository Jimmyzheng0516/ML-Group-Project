import numpy as np
import pandas as pd


def binary_probabilities_to_signal(prob_pos, threshold: float = 0.50):
    prob_pos = np.asarray(prob_pos)
    signal = np.where(prob_pos > threshold, 1, -1)
    signal = np.where(prob_pos == threshold, 0, signal)
    return signal.astype(int)


def multiclass_predictions_to_signal(pred):
    pred = np.asarray(pred).astype(int)
    out = np.zeros(len(pred), dtype=int)
    out[pred > 0] = 1
    out[pred < 0] = -1
    return out


def compute_turnover(signal) -> np.ndarray:
    signal = np.asarray(signal)
    turnover = np.zeros(len(signal))
    turnover[1:] = np.abs(signal[1:] - signal[:-1]) / 2.0
    return turnover


def compute_strategy_returns(signal, ret_a_next, ret_b_next) -> np.ndarray:
    signal = np.asarray(signal)
    ret_a_next = np.asarray(ret_a_next)
    ret_b_next = np.asarray(ret_b_next)
    return 0.5 * signal * (ret_a_next - ret_b_next)


def apply_transaction_costs(strategy_returns, turnover, cost_bps: float = 5.0) -> np.ndarray:
    cost = cost_bps / 10000.0
    return np.asarray(strategy_returns) - cost * np.asarray(turnover)


def wealth_index(returns) -> pd.Series:
    returns = pd.Series(returns).fillna(0.0)
    return (1.0 + returns).cumprod()


def portfolio_metrics(returns, signal=None) -> pd.Series:
    returns = pd.Series(returns).astype(float).dropna()

    mean_return = returns.mean()
    volatility = returns.std(ddof=1)
    sharpe = mean_return / volatility if volatility > 0 else np.nan
    cumulative_return = (1.0 + returns).prod() - 1.0
    hit_rate = (returns > 0).mean()

    out = {
        "cumulative_return": cumulative_return,
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "n_obs": len(returns),
    }

    if signal is not None:
        signal = pd.Series(signal)
        out["active_rate"] = (signal != 0).mean()

    return pd.Series(out)