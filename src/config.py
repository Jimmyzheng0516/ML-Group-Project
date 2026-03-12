from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "mag7.pickle"

SYMBOL_A = "NVDA"
SYMBOL_B = "TSLA"

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

# rolling design
TRAIN_YEARS = 5
VAL_MONTHS = 3
TEST_MONTHS = 3
STEP_MONTHS = 3

# preprocessing
CLIP_LOWER = 0.01
CLIP_UPPER = 0.99

# trading
BINARY_THRESHOLD = 0.50
TRANSACTION_COST_BPS = 5.0

# random seed
RANDOM_STATE = 42

# model families
MODEL_SPECS = [
    ("logit", "binary"),
    ("nn", "binary"),
    ("logit", "multiclass"),
    ("nn", "multiclass"),
]

# ----- validation tuning grids -----

# multiclass threshold candidates
MULTICLASS_QUANTILE_GRID = [0.50, 0.60, 0.67, 0.75]

# logistic regression grid
LOGIT_C_GRID = [0.01, 0.10, 1.0, 10.0, 100.0]
LOGIT_MAX_ITER = 5000

# neural network grid
NN_HIDDEN_LAYER_GRID = [
    (16,),
    (32,),
    (32, 16),
    (64, 32),
]
NN_ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
NN_LEARNING_RATE_INIT_GRID = [5e-4, 1e-3]
NN_ACTIVATION = "relu"
NN_SOLVER = "adam"
NN_MAX_ITER = 1000
NN_EARLY_STOPPING = False

# model selection objective on validation set
# choices: "sharpe", "accuracy", "cumulative_return"
MODEL_SELECTION_OBJECTIVE = "sharpe"

# after validation-based selection, refit on train + validation
REFIT_ON_TRAIN_PLUS_VAL = True

PERFORMANCE_BENCHMARKS = [
    "benchmark_nvda_minus_tsla",
    "benchmark_mom21",
    "benchmark_random_walk",
]