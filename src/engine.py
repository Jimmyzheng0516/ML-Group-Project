from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from . import config
from .backtest import (
    apply_transaction_costs,
    binary_probabilities_to_signal,
    compute_strategy_returns,
    compute_turnover,
    multiclass_predictions_to_signal,
    portfolio_metrics,
    wealth_index,
)
from .data import add_daily_returns, filter_pair, load_raw_data
from .features import add_accounting_features, add_technical_features, build_pair_dataset
from .metrics import classification_metrics, confusion_table, strategy_relative_metrics
from .models import fit_model, get_param_grid


def make_dataset():
    df = load_raw_data(config.RAW_DATA_PATH)
    df = filter_pair(df, config.SYMBOL_A, config.SYMBOL_B)
    df = add_daily_returns(df)
    df = add_accounting_features(df)
    df = add_technical_features(df)
    return build_pair_dataset(df, config.SYMBOL_A, config.SYMBOL_B)


def build_rolling_windows(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    min_date = df["date"].min()
    max_date = df["date"].max()

    windows = []
    anchor = min_date
    window_num = 1

    while True:
        train_start = anchor
        train_end = train_start + pd.DateOffset(years=config.TRAIN_YEARS) - pd.Timedelta(days=1)

        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=config.VAL_MONTHS) - pd.Timedelta(days=1)

        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=config.TEST_MONTHS) - pd.Timedelta(days=1)

        if test_end > max_date:
            break

        train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
        val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        if len(train) > 0 and len(val) > 0 and len(test) > 0:
            windows.append(
                {
                    "window_id": f"win_{window_num:02d}",
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train": train.reset_index(drop=True),
                    "val": val.reset_index(drop=True),
                    "test": test.reset_index(drop=True),
                }
            )
            window_num += 1

        anchor = anchor + pd.DateOffset(months=config.STEP_MONTHS)

    return windows


def windows_to_table(windows):
    rows = []
    for w in windows:
        rows.append(
            {
                "window_id": w["window_id"],
                "train_start": w["train_start"],
                "train_end": w["train_end"],
                "val_start": w["val_start"],
                "val_end": w["val_end"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "n_train": len(w["train"]),
                "n_val": len(w["val"]),
                "n_test": len(w["test"]),
            }
        )
    return pd.DataFrame(rows)


def fit_preprocessor(reference_frame: pd.DataFrame, feature_cols):
    lower = reference_frame[feature_cols].quantile(config.CLIP_LOWER)
    upper = reference_frame[feature_cols].quantile(config.CLIP_UPPER)

    X_ref = reference_frame[feature_cols].clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    scaler.fit(X_ref)

    return scaler, lower, upper


def transform_features(frame: pd.DataFrame, feature_cols, scaler, lower, upper):
    X = frame[feature_cols].clip(lower=lower, upper=upper, axis=1)
    X = pd.DataFrame(
        scaler.transform(X),
        index=frame.index,
        columns=feature_cols,
    )
    return X


def make_multiclass_target(spread_next: pd.Series, abs_threshold: float) -> pd.Series:
    y = pd.Series(0, index=spread_next.index, dtype=int)
    y[spread_next >= abs_threshold] = 1
    y[spread_next <= -abs_threshold] = -1
    return y


def tuning_targets(train: pd.DataFrame, val: pd.DataFrame, class_type: str, params):
    if class_type == "binary":
        return (
            train["y_binary"].astype(int),
            val["y_binary"].astype(int),
            {
                "labels": [0, 1],
                "abs_threshold": np.nan,
                "multiclass_quantile": np.nan,
            },
        )

    q = params["multiclass_quantile"]
    abs_threshold = float(train["spread_next"].abs().quantile(q))

    y_train = make_multiclass_target(train["spread_next"], abs_threshold)
    y_val = make_multiclass_target(val["spread_next"], abs_threshold)

    return (
        y_train,
        y_val,
        {
            "labels": [-1, 0, 1],
            "abs_threshold": abs_threshold,
            "multiclass_quantile": q,
        },
    )


def final_targets(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, class_type: str, best_params):
    if class_type == "binary":
        y_train = train["y_binary"].astype(int)
        y_val = val["y_binary"].astype(int)
        y_test = test["y_binary"].astype(int)

        if config.REFIT_ON_TRAIN_PLUS_VAL:
            refit = pd.concat([train, val], ignore_index=True)
            y_refit = refit["y_binary"].astype(int)
        else:
            y_refit = y_train.copy()

        return (
            y_train,
            y_val,
            y_test,
            y_refit,
            {
                "labels": [0, 1],
                "abs_threshold": np.nan,
                "multiclass_quantile": np.nan,
            },
        )

    q = best_params["multiclass_quantile"]

    if config.REFIT_ON_TRAIN_PLUS_VAL:
        reference = pd.concat([train, val], ignore_index=True)
    else:
        reference = train.copy()

    abs_threshold = float(reference["spread_next"].abs().quantile(q))

    y_train = make_multiclass_target(train["spread_next"], abs_threshold)
    y_val = make_multiclass_target(val["spread_next"], abs_threshold)
    y_test = make_multiclass_target(test["spread_next"], abs_threshold)
    y_refit = make_multiclass_target(reference["spread_next"], abs_threshold)

    return (
        y_train,
        y_val,
        y_test,
        y_refit,
        {
            "labels": [-1, 0, 1],
            "abs_threshold": abs_threshold,
            "multiclass_quantile": q,
        },
    )


def extract_probability_frame(model, X: pd.DataFrame, labels):
    prob = model.predict_proba(X)
    out = pd.DataFrame(index=X.index)

    for label in labels:
        col = f"prob_{label}"
        if label in model.classes_:
            idx = list(model.classes_).index(label)
            out[col] = prob[:, idx]
        else:
            out[col] = 0.0

    return out


def probabilities_or_predictions_to_signal(pred_label, prob_frame: pd.DataFrame, class_type: str):
    if class_type == "binary":
        return binary_probabilities_to_signal(
            prob_frame["prob_1"].to_numpy(),
            threshold=config.BINARY_THRESHOLD,
        )

    return multiclass_predictions_to_signal(pred_label)


def summarize_classification(pred_df: pd.DataFrame, labels, model_name: str, class_type: str):
    class_row = classification_metrics(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=labels,
    ).to_frame().T

    class_row.insert(0, "split", pred_df["split"].iloc[0])
    class_row.insert(0, "window_id", pred_df["window_id"].iloc[0])
    class_row.insert(0, "model_id", pred_df["model_id"].iloc[0])
    class_row.insert(0, "class_type", class_type)
    class_row.insert(0, "model_name", model_name)

    conf_df = confusion_table(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=labels,
    ).reset_index(names="actual_label")

    conf_df.insert(0, "split", pred_df["split"].iloc[0])
    conf_df.insert(0, "window_id", pred_df["window_id"].iloc[0])
    conf_df.insert(0, "model_id", pred_df["model_id"].iloc[0])
    conf_df.insert(0, "class_type", class_type)
    conf_df.insert(0, "model_name", model_name)

    return class_row, conf_df


def summarize_portfolios(pred_df: pd.DataFrame, model_name: str, class_type: str):
    rows = []

    portfolios = {
        "strategy_gross": ("strategy_gross", "signal"),
        "strategy_net": ("strategy_net", "signal"),
        "benchmark_nvda_minus_tsla": ("benchmark_nvda_minus_tsla", None),
        "benchmark_tsla_minus_nvda": ("benchmark_tsla_minus_nvda", None),
        "benchmark_mom21": ("benchmark_mom21", "benchmark_mom21_signal"),
        "benchmark_random_walk": ("benchmark_random_walk", "benchmark_random_walk_signal"),
    }

    for portfolio_name, (ret_col, signal_col) in portfolios.items():
        signal = pred_df[signal_col] if signal_col is not None else None
        row = portfolio_metrics(pred_df[ret_col], signal=signal).to_dict()

        row.update(
            {
                "model_name": model_name,
                "class_type": class_type,
                "model_id": pred_df["model_id"].iloc[0],
                "window_id": pred_df["window_id"].iloc[0],
                "split": pred_df["split"].iloc[0],
                "portfolio_name": portfolio_name,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_relative_performance(pred_df: pd.DataFrame, model_name: str, class_type: str):
    rows = []

    for strategy_name in ["strategy_gross", "strategy_net"]:
        for benchmark_name in config.PERFORMANCE_BENCHMARKS:
            rel = strategy_relative_metrics(
                pred_df[strategy_name],
                pred_df[benchmark_name],
            ).to_dict()

            rel.update(
                {
                    "model_name": model_name,
                    "class_type": class_type,
                    "model_id": pred_df["model_id"].iloc[0],
                    "window_id": pred_df["window_id"].iloc[0],
                    "split": pred_df["split"].iloc[0],
                    "strategy_name": strategy_name,
                    "benchmark_name": benchmark_name,
                }
            )
            rows.append(rel)

    return pd.DataFrame(rows)


def coefficient_table(model, feature_cols, model_name: str, class_type: str, window_id: str):
    if model_name != "logit":
        return None

    coef = np.atleast_2d(model.coef_)
    rows = []

    if class_type == "binary":
        for feature, value in zip(feature_cols, coef[0]):
            rows.append(
                {
                    "model_name": model_name,
                    "class_type": class_type,
                    "model_id": f"{model_name}_{class_type}",
                    "window_id": window_id,
                    "class_label": 1,
                    "feature": feature,
                    "coef": value,
                }
            )
    else:
        for class_label, coef_row in zip(model.classes_, coef):
            for feature, value in zip(feature_cols, coef_row):
                rows.append(
                    {
                        "model_name": model_name,
                        "class_type": class_type,
                        "model_id": f"{model_name}_{class_type}",
                        "window_id": window_id,
                        "class_label": class_label,
                        "feature": feature,
                        "coef": value,
                    }
                )

    return pd.DataFrame(rows)


def make_prediction_frame(
    frame: pd.DataFrame,
    y_true,
    y_pred,
    prob_frame: pd.DataFrame,
    signal,
    strategy_gross,
    strategy_net,
    turnover,
    model_name: str,
    class_type: str,
    window_id: str,
    split: str,
    target_info: dict,
):
    out = frame[["date", "ret_a_next", "ret_b_next", "spread_t", "spread_next", "mom21_diff"]].copy()

    out["model_name"] = model_name
    out["class_type"] = class_type
    out["model_id"] = f"{model_name}_{class_type}"
    out["window_id"] = window_id
    out["split"] = split
    out["abs_threshold"] = target_info["abs_threshold"]
    out["multiclass_quantile"] = target_info["multiclass_quantile"]

    out["y_true"] = np.asarray(y_true)
    out["y_pred"] = np.asarray(y_pred)
    out["signal"] = np.asarray(signal)
    out["is_active"] = (out["signal"] != 0).astype(int)

    out["turnover"] = np.asarray(turnover)
    out["strategy_gross"] = np.asarray(strategy_gross)
    out["strategy_net"] = np.asarray(strategy_net)

    for col in prob_frame.columns:
        out[col] = prob_frame[col].to_numpy()

    spread_leg = 0.5 * (frame["ret_a_next"] - frame["ret_b_next"])
    out["benchmark_nvda_minus_tsla"] = spread_leg
    out["benchmark_tsla_minus_nvda"] = -spread_leg

    mom21_signal = np.sign(frame["mom21_diff"]).astype(int)
    out["benchmark_mom21_signal"] = mom21_signal
    out["benchmark_mom21"] = 0.5 * mom21_signal * (frame["ret_a_next"] - frame["ret_b_next"])

    rw_signal = np.sign(frame["spread_t"]).astype(int)
    out["benchmark_random_walk_signal"] = rw_signal
    out["benchmark_random_walk"] = 0.5 * rw_signal * (frame["ret_a_next"] - frame["ret_b_next"])

    return out


def evaluate_candidate(train, val, feature_cols, model_name, class_type, params):
    scaler, lower, upper = fit_preprocessor(train, feature_cols)
    X_train = transform_features(train, feature_cols, scaler, lower, upper)
    X_val = transform_features(val, feature_cols, scaler, lower, upper)

    y_train, y_val, target_info = tuning_targets(train, val, class_type, params)

    model = fit_model(model_name, X_train, y_train, params, config)

    y_val_pred = model.predict(X_val)
    prob_val = extract_probability_frame(model, X_val, target_info["labels"])
    signal_val = probabilities_or_predictions_to_signal(y_val_pred, prob_val, class_type)

    turnover_val = compute_turnover(signal_val)
    strat_gross_val = compute_strategy_returns(signal_val, val["ret_a_next"], val["ret_b_next"])
    strat_net_val = apply_transaction_costs(
        strat_gross_val,
        turnover_val,
        cost_bps=config.TRANSACTION_COST_BPS,
    )

    class_stats = classification_metrics(y_val, y_val_pred, target_info["labels"])
    port_stats = portfolio_metrics(strat_net_val, signal=signal_val)

    row = {
        "model_name": model_name,
        "class_type": class_type,
        "candidate_id": None,
        "window_id": None,
        "val_accuracy": class_stats["accuracy"],
        "val_n_obs": class_stats["n_obs"],
        "val_cumulative_return": port_stats["cumulative_return"],
        "val_mean_return": port_stats["mean_return"],
        "val_volatility": port_stats["volatility"],
        "val_sharpe": port_stats["sharpe"],
        "val_hit_rate": port_stats["hit_rate"],
        "val_active_rate": port_stats.get("active_rate", np.nan),
        "selected_abs_threshold_train_only": target_info["abs_threshold"],
        "multiclass_quantile": target_info["multiclass_quantile"],
        "C": params.get("C", np.nan),
        "hidden_layer_sizes": params.get("hidden_layer_sizes", np.nan),
        "alpha": params.get("alpha", np.nan),
        "learning_rate_init": params.get("learning_rate_init", np.nan),
    }

    return row


def add_selection_columns(tuning_df: pd.DataFrame):
    tuning_df = tuning_df.copy()

    tuning_df["selection_metric"] = np.nan

    if config.MODEL_SELECTION_OBJECTIVE == "sharpe":
        tuning_df["selection_metric"] = tuning_df["val_sharpe"]
    elif config.MODEL_SELECTION_OBJECTIVE == "accuracy":
        tuning_df["selection_metric"] = tuning_df["val_accuracy"]
    elif config.MODEL_SELECTION_OBJECTIVE == "cumulative_return":
        tuning_df["selection_metric"] = tuning_df["val_cumulative_return"]
    else:
        raise ValueError(f"Unknown MODEL_SELECTION_OBJECTIVE: {config.MODEL_SELECTION_OBJECTIVE}")

    tuning_df["selection_metric"] = tuning_df["selection_metric"].replace([np.inf, -np.inf], np.nan)
    tuning_df["selection_metric_filled"] = tuning_df["selection_metric"].fillna(-1e12)

    tuning_df = tuning_df.sort_values(
        by=[
            "selection_metric_filled",
            "val_accuracy",
            "val_cumulative_return",
            "val_active_rate",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    tuning_df["rank_within_window"] = np.arange(1, len(tuning_df) + 1)
    tuning_df["selected"] = 0
    tuning_df.loc[0, "selected"] = 1

    return tuning_df


def tune_hyperparameters(
    train,
    val,
    feature_cols,
    model_name,
    class_type,
    window_id,
    show_candidate_progress=False,
):
    rows = []
    param_grid = get_param_grid(model_name, class_type, config)

    iterator = param_grid
    if show_candidate_progress:
        iterator = tqdm(
            param_grid,
            desc=f"{model_name}_{class_type} | {window_id} | tuning",
            leave=False,
        )

    for i, params in enumerate(iterator, start=1):
        row = evaluate_candidate(train, val, feature_cols, model_name, class_type, params)
        row["candidate_id"] = f"cand_{i:03d}"
        row["window_id"] = window_id
        rows.append(row)

    tuning_df = pd.DataFrame(rows)
    tuning_df = add_selection_columns(tuning_df)
    best_row = tuning_df.loc[tuning_df["selected"] == 1].iloc[0].copy()

    best_params = {}
    if model_name == "logit":
        best_params["C"] = float(best_row["C"])
    else:
        best_params["hidden_layer_sizes"] = best_row["hidden_layer_sizes"]
        best_params["alpha"] = float(best_row["alpha"])
        best_params["learning_rate_init"] = float(best_row["learning_rate_init"])

    if class_type == "multiclass":
        best_params["multiclass_quantile"] = float(best_row["multiclass_quantile"])

    best_params_summary = best_row.to_frame().T
    best_params_summary.insert(0, "model_id", f"{model_name}_{class_type}")

    return tuning_df, best_params_summary, best_params


def fit_selected_model(train, val, test, feature_cols, model_name, class_type, best_params):
    if config.REFIT_ON_TRAIN_PLUS_VAL:
        reference = pd.concat([train, val], ignore_index=True)
    else:
        reference = train.copy()

    scaler, lower, upper = fit_preprocessor(reference, feature_cols)

    X_train = transform_features(train, feature_cols, scaler, lower, upper)
    X_val = transform_features(val, feature_cols, scaler, lower, upper)
    X_test = transform_features(test, feature_cols, scaler, lower, upper)
    X_refit = transform_features(reference, feature_cols, scaler, lower, upper)

    y_train, y_val, y_test, y_refit, target_info = final_targets(train, val, test, class_type, best_params)

    model = fit_model(model_name, X_refit, y_refit, best_params, config)

    fitted_object = {
        "model": model,
        "scaler": scaler,
        "clip_lower": lower,
        "clip_upper": upper,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "target_info": target_info,
    }

    return model, fitted_object, X_train, X_val, X_test, y_train, y_val, y_test, target_info


def run_single_model(
    pair: pd.DataFrame,
    windows,
    model_name: str,
    class_type: str,
    show_progress=True,
    show_candidate_progress=False,
):
    feature_cols = config.FEATURES
    model_id = f"{model_name}_{class_type}"

    tuning_rows = []
    best_param_rows = []
    prediction_frames = []
    classification_rows = []
    confusion_rows = []
    portfolio_rows = []
    relative_rows = []
    coefficient_rows = []
    fitted_objects = {}

    window_iterator = windows
    if show_progress:
        window_iterator = tqdm(
            windows,
            desc=f"{model_id} | rolling windows",
            leave=False,
        )

    for window in window_iterator:
        tuning_df, best_params_summary, best_params = tune_hyperparameters(
            train=window["train"],
            val=window["val"],
            feature_cols=feature_cols,
            model_name=model_name,
            class_type=class_type,
            window_id=window["window_id"],
            show_candidate_progress=show_candidate_progress,
        )

        tuning_df.insert(0, "model_id", model_id)
        tuning_rows.append(tuning_df)
        best_param_rows.append(best_params_summary)

        (
            model,
            fitted_object,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            target_info,
        ) = fit_selected_model(
            train=window["train"],
            val=window["val"],
            test=window["test"],
            feature_cols=feature_cols,
            model_name=model_name,
            class_type=class_type,
            best_params=best_params,
        )

        fitted_object.update(
            {
                "model_name": model_name,
                "class_type": class_type,
                "model_id": model_id,
                "window_id": window["window_id"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "val_start": window["val_start"],
                "val_end": window["val_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
            }
        )
        fitted_objects[window["window_id"]] = fitted_object

        coef_df = coefficient_table(
            model,
            feature_cols,
            model_name,
            class_type,
            window["window_id"],
        )
        if coef_df is not None:
            coefficient_rows.append(coef_df)

        for split_name, frame, X_split, y_split in [
            ("train", window["train"], X_train, y_train),
            ("val", window["val"], X_val, y_val),
            ("test", window["test"], X_test, y_test),
        ]:
            y_pred = model.predict(X_split)
            prob_frame = extract_probability_frame(model, X_split, target_info["labels"])
            signal = probabilities_or_predictions_to_signal(y_pred, prob_frame, class_type)

            turnover = compute_turnover(signal)
            strategy_gross = compute_strategy_returns(
                signal,
                frame["ret_a_next"],
                frame["ret_b_next"],
            )
            strategy_net = apply_transaction_costs(
                strategy_gross,
                turnover,
                cost_bps=config.TRANSACTION_COST_BPS,
            )

            pred_df = make_prediction_frame(
                frame=frame,
                y_true=y_split,
                y_pred=y_pred,
                prob_frame=prob_frame,
                signal=signal,
                strategy_gross=strategy_gross,
                strategy_net=strategy_net,
                turnover=turnover,
                model_name=model_name,
                class_type=class_type,
                window_id=window["window_id"],
                split=split_name,
                target_info=target_info,
            )
            prediction_frames.append(pred_df)

            class_row, conf_df = summarize_classification(
                pred_df,
                target_info["labels"],
                model_name,
                class_type,
            )
            classification_rows.append(class_row)
            confusion_rows.append(conf_df)
            portfolio_rows.append(summarize_portfolios(pred_df, model_name, class_type))
            relative_rows.append(summarize_relative_performance(pred_df, model_name, class_type))

    return {
        "model_id": model_id,
        "tuning_summary": pd.concat(tuning_rows, ignore_index=True),
        "best_params_summary": pd.concat(best_param_rows, ignore_index=True),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "classification_summary": pd.concat(classification_rows, ignore_index=True),
        "confusion_summary": pd.concat(confusion_rows, ignore_index=True),
        "portfolio_summary": pd.concat(portfolio_rows, ignore_index=True),
        "relative_summary": pd.concat(relative_rows, ignore_index=True),
        "coefficient_summary": pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else None,
        "fitted_objects": fitted_objects,
    }

def save_wealth_plot(test_predictions: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(11, 6))

    for model_id, sub in test_predictions.groupby("model_id"):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], wealth_index(sub["strategy_net"]), label=model_id)

    bench = (
        test_predictions[["date", "benchmark_nvda_minus_tsla"]]
        .drop_duplicates(subset="date")
        .sort_values("date")
    )
    plt.plot(
        bench["date"],
        wealth_index(bench["benchmark_nvda_minus_tsla"]),
        linestyle="--",
        label="benchmark_nvda_minus_tsla",
    )

    plt.title("Rolling test wealth curves")
    plt.xlabel("Date")
    plt.ylabel("Wealth index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_all_experiments(show_progress=True, show_candidate_progress=False):
    pair = make_dataset()
    windows = build_rolling_windows(pair)
    window_table = windows_to_table(windows)

    output_tables = config.ROOT / "outputs" / "tables"
    output_figures = config.ROOT / "outputs" / "figures"
    output_models = config.ROOT / "outputs" / "models"

    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    output_models.mkdir(parents=True, exist_ok=True)

    window_table.to_csv(output_tables / "rolling_windows.csv", index=False)

    all_tuning = []
    all_best = []
    all_predictions = []
    all_classification = []
    all_confusion = []
    all_portfolio = []
    all_relative = []
    all_coefficients = []

    model_iterator = config.MODEL_SPECS
    if show_progress:
        model_iterator = tqdm(
            config.MODEL_SPECS,
            desc="All models",
            leave=True,
        )

    for model_name, class_type in model_iterator:
        result = run_single_model(
            pair=pair,
            windows=windows,
            model_name=model_name,
            class_type=class_type,
            show_progress=show_progress,
            show_candidate_progress=show_candidate_progress,
        )
        model_id = result["model_id"]

        result["tuning_summary"].to_csv(output_tables / f"tuning_summary_{model_id}.csv", index=False)
        result["best_params_summary"].to_csv(output_tables / f"best_params_summary_{model_id}.csv", index=False)
        result["predictions"].to_csv(output_tables / f"predictions_{model_id}.csv", index=False)
        result["classification_summary"].to_csv(output_tables / f"classification_summary_{model_id}.csv", index=False)
        result["confusion_summary"].to_csv(output_tables / f"confusion_summary_{model_id}.csv", index=False)
        result["portfolio_summary"].to_csv(output_tables / f"portfolio_summary_{model_id}.csv", index=False)
        result["relative_summary"].to_csv(output_tables / f"relative_summary_{model_id}.csv", index=False)

        if result["coefficient_summary"] is not None:
            result["coefficient_summary"].to_csv(output_tables / f"coefficients_{model_id}.csv", index=False)

        joblib.dump(
            result["fitted_objects"],
            output_models / f"{model_id}_rolling_models.joblib",
        )

        all_tuning.append(result["tuning_summary"])
        all_best.append(result["best_params_summary"])
        all_predictions.append(result["predictions"])
        all_classification.append(result["classification_summary"])
        all_confusion.append(result["confusion_summary"])
        all_portfolio.append(result["portfolio_summary"])
        all_relative.append(result["relative_summary"])

        if result["coefficient_summary"] is not None:
            all_coefficients.append(result["coefficient_summary"])

    tuning_summary = pd.concat(all_tuning, ignore_index=True)
    best_params_summary = pd.concat(all_best, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    classification_summary = pd.concat(all_classification, ignore_index=True)
    confusion_summary = pd.concat(all_confusion, ignore_index=True)
    portfolio_summary = pd.concat(all_portfolio, ignore_index=True)
    relative_summary = pd.concat(all_relative, ignore_index=True)
    coefficient_summary = pd.concat(all_coefficients, ignore_index=True) if all_coefficients else None

    tuning_summary.to_csv(output_tables / "tuning_summary_all_models.csv", index=False)
    best_params_summary.to_csv(output_tables / "best_params_summary_all_models.csv", index=False)
    predictions.to_csv(output_tables / "predictions_all_models.csv", index=False)
    classification_summary.to_csv(output_tables / "classification_summary_all_models.csv", index=False)
    confusion_summary.to_csv(output_tables / "confusion_summary_all_models.csv", index=False)
    portfolio_summary.to_csv(output_tables / "portfolio_summary_all_models.csv", index=False)
    relative_summary.to_csv(output_tables / "relative_summary_all_models.csv", index=False)

    if coefficient_summary is not None:
        coefficient_summary.to_csv(output_tables / "coefficients_all_models.csv", index=False)

    test_predictions = predictions[predictions["split"] == "test"].copy()
    save_wealth_plot(
        test_predictions,
        output_figures / "wealth_curve_test_all_models.png",
    )

    return {
        "dataset": pair,
        "windows": window_table,
        "tuning_summary": tuning_summary,
        "best_params_summary": best_params_summary,
        "predictions": predictions,
        "classification_summary": classification_summary,
        "confusion_summary": confusion_summary,
        "portfolio_summary": portfolio_summary,
        "relative_summary": relative_summary,
        "coefficient_summary": coefficient_summary,
    }