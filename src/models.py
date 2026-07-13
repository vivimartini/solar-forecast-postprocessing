# src/models.py
"""Models. Layer 1: a gradient-boosted residual correction (LightGBM)."""
import lightgbm as lgb

DEFAULT_PARAMS = dict(
    objective="regression",      # squared error -> predicts the mean residual
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=50,
    random_state=42, n_jobs=-1, verbose=-1,
)


def train_residual_model(X_train, y_train, params=None):
    model = lgb.LGBMRegressor(**(params or DEFAULT_PARAMS))
    model.fit(X_train, y_train)
    return model