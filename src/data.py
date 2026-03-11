from pathlib import Path
import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def filter_pair(df: pd.DataFrame, symbol_a: str, symbol_b: str) -> pd.DataFrame:
    out = df[df["symbol"].isin([symbol_a, symbol_b])].copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out.groupby("symbol")["close_adj"].pct_change()
    return out