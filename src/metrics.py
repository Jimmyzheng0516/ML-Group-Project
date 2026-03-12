import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def classification_metrics(y_true, y_pred, labels) -> pd.Series:
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "n_obs": len(y_true),
    }

    for label in labels:
        label_name = str(label).replace("-", "minus_")
        out[f"n_pred_{label_name}"] = int((y_pred == label).sum())
        out[f"n_true_{label_name}"] = int((y_true == label).sum())

    return pd.Series(out)


def confusion_table(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    index = [f"actual_{label}" for label in labels]
    columns = [f"pred_{label}" for label in labels]
    return pd.DataFrame(cm, index=index, columns=columns)


def strategy_relative_metrics(strategy_returns, benchmark_returns) -> pd.Series:
    strategy_returns = pd.Series(strategy_returns).astype(float).dropna()
    benchmark_returns = pd.Series(benchmark_returns).astype(float).dropna()

    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()

    if aligned.empty:
        return pd.Series(
            {
                "beta": np.nan,
                "alpha": np.nan,
                "information_ratio": np.nan,
            }
        )

    strat = aligned["strategy"]
    bench = aligned["benchmark"]

    bench_var = np.var(bench, ddof=1)
    if np.isclose(bench_var, 0.0):
        return pd.Series(
            {
                "beta": np.nan,
                "alpha": np.nan,
                "information_ratio": np.nan,
            }
        )

    beta = np.cov(bench, strat, ddof=1)[0, 1] / bench_var
    residual = strat - beta * bench
    alpha = residual.mean()
    residual_std = residual.std(ddof=1)
    ir = alpha / residual_std if residual_std > 0 else np.nan

    return pd.Series(
        {
            "beta": beta,
            "alpha": alpha,
            "information_ratio": ir,
        }
    )