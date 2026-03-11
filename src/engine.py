from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config
from .data import load_raw_data, filter_pair, add_daily_returns
from .features import add_accounting_features, add_technical_features, build_pair_dataset
from .models import fit_logistic_regression
from .metrics import classification_metrics, confusion_table
from .backtest import (
    probabilities_to_signal,
    compute_turnover,
    compute_strategy_returns,
    apply_transaction_costs,
    portfolio_metrics,
    wealth_index,
)


def make_dataset():
    df = load_raw_data(config.RAW_DATA_PATH)
    df = filter_pair(df, config.SYMBOL_A, config.SYMBOL_B)
    df = add_daily_returns(df)
    df = add_accounting_features(df)
    df = add_technical_features(df)
    pair = build_pair_dataset(df, config.SYMBOL_A, config.SYMBOL_B)
    return pair


def time_split(df: pd.DataFrame):
    train = df[(df["date"] >= config.TRAIN_START) & (df["date"] <= config.TRAIN_END)].copy()
    test = df[(df["date"] >= config.TEST_START) & (df["date"] <= config.TEST_END)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def clip_and_scale(train: pd.DataFrame, test: pd.DataFrame, feature_cols):
    lower = train[feature_cols].quantile(config.CLIP_LOWER)
    upper = train[feature_cols].quantile(config.CLIP_UPPER)

    X_train = train[feature_cols].clip(lower=lower, upper=upper, axis=1)
    X_test = test[feature_cols].clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=train.index,
        columns=feature_cols,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=test.index,
        columns=feature_cols,
    )
    return X_train_scaled, X_test_scaled, scaler, lower, upper


def make_prediction_frame(frame: pd.DataFrame, prob, signal, strategy_gross, strategy_net, turnover):
    out = frame[["date", "y", "ret_a_next", "ret_b_next", "spread_next", "spread_t"]].copy()
    out["prob_nvda_beats_tsla"] = prob
    out["signal"] = signal
    out["turnover"] = turnover
    out["strategy_gross"] = strategy_gross
    out["strategy_net"] = strategy_net

    out["benchmark_nvda_minus_tsla"] = 0.5 * (frame["ret_a_next"] - frame["ret_b_next"])
    out["benchmark_tsla_minus_nvda"] = -out["benchmark_nvda_minus_tsla"]

    mom21_signal = (frame["mom21_diff"] >= 0).astype(int).replace({0: -1, 1: 1})
    out["benchmark_mom21_signal"] = mom21_signal
    out["benchmark_mom21"] = 0.5 * mom21_signal * (frame["ret_a_next"] - frame["ret_b_next"])

    rw_signal = (frame["spread_t"] >= 0).astype(int).replace({0: -1, 1: 1})
    out["benchmark_random_walk_signal"] = rw_signal
    out["benchmark_random_walk"] = 0.5 * rw_signal * (frame["ret_a_next"] - frame["ret_b_next"])

    return out


def save_plot(pred_train: pd.DataFrame, pred_test: pd.DataFrame, output_path: Path):
    train_wealth = wealth_index(pred_train["strategy_net"])
    test_wealth = wealth_index(pred_test["strategy_net"])

    plt.figure(figsize=(10, 5))
    plt.plot(pred_train["date"], train_wealth, label="Train net strategy")
    plt.plot(pred_test["date"], test_wealth, label="Test net strategy")
    plt.axvline(pd.to_datetime(config.TEST_START), color="black", linestyle="--", linewidth=1)
    plt.title("NVDA-TSLA long-short wealth index")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_experiment():
    pair = make_dataset()
    train, test = time_split(pair)

    feature_cols = config.FEATURES
    X_train, X_test, scaler, clip_lower, clip_upper = clip_and_scale(train, test, feature_cols)
    y_train = train["y"]
    y_test = test["y"]

    model = fit_logistic_regression(X_train, y_train, random_state=config.RANDOM_STATE)

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    pred_train = (prob_train >= config.THRESHOLD).astype(int)
    pred_test = (prob_test >= config.THRESHOLD).astype(int)

    class_train = classification_metrics(y_train, pred_train, prob_train)
    class_test = classification_metrics(y_test, pred_test, prob_test)

    signal_train = probabilities_to_signal(prob_train, threshold=config.THRESHOLD)
    signal_test = probabilities_to_signal(prob_test, threshold=config.THRESHOLD)

    turnover_train = compute_turnover(signal_train)
    turnover_test = compute_turnover(signal_test)

    strat_train_gross = compute_strategy_returns(signal_train, train["ret_a_next"], train["ret_b_next"])
    strat_test_gross = compute_strategy_returns(signal_test, test["ret_a_next"], test["ret_b_next"])

    strat_train_net = apply_transaction_costs(strat_train_gross, turnover_train, cost_bps=config.TRANSACTION_COST_BPS)
    strat_test_net = apply_transaction_costs(strat_test_gross, turnover_test, cost_bps=config.TRANSACTION_COST_BPS)

    pred_train_df = make_prediction_frame(train, prob_train, signal_train, strat_train_gross, strat_train_net, turnover_train)
    pred_test_df = make_prediction_frame(test, prob_test, signal_test, strat_test_gross, strat_test_net, turnover_test)

    port_rows = {
    "train_strategy_gross": portfolio_metrics(pred_train_df["strategy_gross"]),
    "train_strategy_net": portfolio_metrics(pred_train_df["strategy_net"]),
    "train_benchmark_nvda_minus_tsla": portfolio_metrics(pred_train_df["benchmark_nvda_minus_tsla"]),
    "train_benchmark_tsla_minus_nvda": portfolio_metrics(pred_train_df["benchmark_tsla_minus_nvda"]),
    "train_benchmark_mom21": portfolio_metrics(pred_train_df["benchmark_mom21"]),
    "train_benchmark_random_walk": portfolio_metrics(pred_train_df["benchmark_random_walk"]),

    "test_strategy_gross": portfolio_metrics(pred_test_df["strategy_gross"]),
    "test_strategy_net": portfolio_metrics(pred_test_df["strategy_net"]),
    "test_benchmark_nvda_minus_tsla": portfolio_metrics(pred_test_df["benchmark_nvda_minus_tsla"]),
    "test_benchmark_tsla_minus_nvda": portfolio_metrics(pred_test_df["benchmark_tsla_minus_nvda"]),
    "test_benchmark_mom21": portfolio_metrics(pred_test_df["benchmark_mom21"]),
    "test_benchmark_random_walk": portfolio_metrics(pred_test_df["benchmark_random_walk"]),
    }
    portfolio_table = pd.DataFrame(port_rows).T

    coef_table = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": model.coef_[0],
        }
    ).sort_values("coef", ascending=False)

    output_tables = config.ROOT / "outputs" / "tables"
    output_figures = config.ROOT / "outputs" / "figures"
    output_models = config.ROOT / "outputs" / "models"

    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    output_models.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "metric": class_train.index,
            "train": class_train.values,
            "test": class_test.reindex(class_train.index).values,
        }
    ).to_csv(output_tables / "classification_metrics.csv", index=False)

    confusion_table(y_train, pred_train).to_csv(output_tables / "confusion_train.csv")
    confusion_table(y_test, pred_test).to_csv(output_tables / "confusion_test.csv")
    portfolio_table.to_csv(output_tables / "portfolio_metrics.csv")
    coef_table.to_csv(output_tables / "logit_coefficients.csv", index=False)
    pred_train_df.to_csv(output_tables / "predictions_train.csv", index=False)
    pred_test_df.to_csv(output_tables / "predictions_test.csv", index=False)

    save_plot(pred_train_df, pred_test_df, output_figures / "wealth_curve.png")

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "clip_lower": clip_lower,
            "clip_upper": clip_upper,
            "feature_cols": feature_cols,
        },
        output_models / "logit_model.joblib",
    )

    results = {
        "dataset": pair,
        "train": train,
        "test": test,
        "X_train": X_train,
        "X_test": X_test,
        "class_train": class_train,
        "class_test": class_test,
        "confusion_train": confusion_table(y_train, pred_train),
        "confusion_test": confusion_table(y_test, pred_test),
        "portfolio_table": portfolio_table,
        "coef_table": coef_table,
        "pred_train": pred_train_df,
        "pred_test": pred_test_df,
    }
    return results