from sklearn.linear_model import LogisticRegression


def fit_logistic_regression(X_train, y_train, random_state: int = 42):
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model