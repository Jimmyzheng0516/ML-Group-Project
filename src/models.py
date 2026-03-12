from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_param_grid(model_name: str, class_type: str, config):
    if model_name == "logit":
        base_grid = [
            {
                "C": float(C),
            }
            for C in config.LOGIT_C_GRID
        ]

    elif model_name == "nn":
        base_grid = [
            {
                "hidden_layer_sizes": hidden_layer_sizes,
                "alpha": float(alpha),
                "learning_rate_init": float(learning_rate_init),
            }
            for hidden_layer_sizes, alpha, learning_rate_init in product(
                config.NN_HIDDEN_LAYER_GRID,
                config.NN_ALPHA_GRID,
                config.NN_LEARNING_RATE_INIT_GRID,
            )
        ]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if class_type == "multiclass":
        full_grid = []
        for params in base_grid:
            for q in config.MULTICLASS_QUANTILE_GRID:
                full_grid.append({**params, "multiclass_quantile": float(q)})
        return full_grid

    return base_grid


def fit_logistic_regression(X_train, y_train, params, config):
    model = LogisticRegression(
        C=params["C"],
        penalty="l2",
        solver="lbfgs",
        max_iter=config.LOGIT_MAX_ITER,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def fit_neural_network(X_train, y_train, params, config):
    model = MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=config.NN_ACTIVATION,
        solver=config.NN_SOLVER,
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=config.NN_MAX_ITER,
        early_stopping=config.NN_EARLY_STOPPING,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def fit_model(model_name: str, X_train, y_train, params, config):
    if model_name == "logit":
        return fit_logistic_regression(X_train, y_train, params, config)

    if model_name == "nn":
        return fit_neural_network(X_train, y_train, params, config)

    raise ValueError(f"Unknown model_name: {model_name}")