from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "mag7.pickle"

SYMBOL_A = "NVDA"
SYMBOL_B = "TSLA"

TRAIN_START = "2011-01-03"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-03"
TEST_END = "2025-12-31"

THRESHOLD = 0.50 
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
TRANSACTION_COST_BPS = 5.0
RANDOM_STATE = 42

CLIP_LOWER = 0.01
CLIP_UPPER = 0.99

FEATURES = [
    "op_margin_diff",
    "net_margin_diff",
    "leverage_diff",
    "cash_ratio_diff",
    "rnd_intensity_diff",
    "capex_intensity_diff",
    "asset_turnover_diff",
    "log_mcap_diff",
    "mom5_diff",
    "mom21_diff",
    "mom63_diff",
    "ma_gap20_diff",
    "rsi14_diff",
    "rv21_diff",
    "vol_shock20_diff",
]