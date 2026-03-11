import numpy as np
import pandas as pd


def probabilities_to_signal(prob, threshold: float = 0.5):
    prob = np.asarray(prob)
    return np.where(prob >= threshold, 1, -1)


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
    returns = pd.Series(returns)
    return (1.0 + returns).cumprod()


def max_drawdown(returns) -> float:
    wealth = wealth_index(returns)
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd.min()


def portfolio_metrics(returns) -> pd.Series:
    returns = pd.Series(returns).dropna()

    mean_daily = returns.mean()
    vol_daily = returns.std()
    ann_return = mean_daily * 252
    ann_vol = vol_daily * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    mdd = max_drawdown(returns)
    calmar = ann_return / abs(mdd) if mdd < 0 else np.nan
    hit_rate = (returns > 0).mean()
    cum_return = (1.0 + returns).prod() - 1.0

    out = {
        "cumulative_return": cum_return,
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "hit_rate": hit_rate,
    }
    return pd.Series(out)