import numpy as np
import pandas as pd


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi


def add_accounting_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["op_margin"] = safe_divide(out["oiadpq"], out["saleq"])
    out["net_margin"] = safe_divide(out["niq"], out["saleq"])
    out["leverage"] = safe_divide(out["dlttq"] + out["dlcq"], out["atq"])
    out["cash_ratio"] = safe_divide(out["cheq"], out["atq"])
    out["rnd_intensity"] = safe_divide(out["xrdq"], out["saleq"])
    out["capex_intensity"] = safe_divide(out["capxy"], out["atq"])
    out["asset_turnover"] = safe_divide(out["saleq"], out["atq"])
    out["log_mcap"] = np.log(out["close_adj"] * out["cshoq"])

    return out


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    g = out.groupby("symbol", group_keys=False)

    out["mom5"] = g["close_adj"].pct_change(5)
    out["mom21"] = g["close_adj"].pct_change(21)
    out["mom63"] = g["close_adj"].pct_change(63)

    ma20 = g["close_adj"].rolling(20).mean().reset_index(level=0, drop=True)
    out["ma_gap20"] = out["close_adj"] / ma20 - 1.0

    out["rsi14"] = g["close_adj"].apply(compute_rsi, window=14).reset_index(level=0, drop=True)

    out["rv21"] = g["ret_1d"].rolling(21).std().reset_index(level=0, drop=True) * np.sqrt(252)

    vol_ma20 = g["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    out["vol_shock20"] = np.log(out["volume"] / vol_ma20)

    return out


def pivot_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "date",
        "symbol",
        "ret_1d",
        "op_margin",
        "net_margin",
        "leverage",
        "cash_ratio",
        "rnd_intensity",
        "capex_intensity",
        "asset_turnover",
        "log_mcap",
        "mom5",
        "mom21",
        "mom63",
        "ma_gap20",
        "rsi14",
        "rv21",
        "vol_shock20",
    ]

    wide = df[keep_cols].pivot(index="date", columns="symbol")
    wide.columns = [f"{col}_{sym.lower()}" for col, sym in wide.columns]
    wide = wide.sort_index()
    return wide.reset_index()


def build_pair_dataset(df: pd.DataFrame, symbol_a: str, symbol_b: str) -> pd.DataFrame:
    wide = pivot_pair_features(df)

    a = symbol_a.lower()
    b = symbol_b.lower()

    pair = wide.copy()

    base_features = [
        "op_margin",
        "net_margin",
        "leverage",
        "cash_ratio",
        "rnd_intensity",
        "capex_intensity",
        "asset_turnover",
        "log_mcap",
        "mom5",
        "mom21",
        "mom63",
        "ma_gap20",
        "rsi14",
        "rv21",
        "vol_shock20",
    ]

    for name in base_features:
        pair[f"{name}_diff"] = pair[f"{name}_{a}"] - pair[f"{name}_{b}"]

    pair["spread_t"] = pair[f"ret_1d_{a}"] - pair[f"ret_1d_{b}"]
    pair["ret_a_next"] = pair[f"ret_1d_{a}"].shift(-1)
    pair["ret_b_next"] = pair[f"ret_1d_{b}"].shift(-1)
    pair["spread_next"] = pair["spread_t"].shift(-1)

    keep_cols = ["date", "ret_a_next", "ret_b_next", "spread_next", "spread_t"] + [f"{x}_diff" for x in base_features]
    pair = pair[keep_cols].dropna().reset_index(drop=True)
    pair["y"] = (pair["spread_next"] > 0).astype(int)

    return pair