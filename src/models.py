# src/models.py
"""Models. Layer 1: a gradient-boosted residual correction (LightGBM).
Regularised + early stopping so it keeps only strong, repeatable patterns.
An XGBoost variant with a matched config is provided for a like-for-like comparison.
"""
import lightgbm as lgb
import xgboost as xgb

DEFAULT_PARAMS = dict(
    objective="regression",
    n_estimators=3000,          # upper bound; early stopping picks the real number
    learning_rate=0.03,
    num_leaves=15,              # shallower trees -> less freedom to memorise
    min_child_samples=200,      # needs 200+ examples before committing to a pattern
    subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=42, n_jobs=-1, verbose=-1,
)

# XGBoost params chosen to mirror DEFAULT_PARAMS as closely as the two libraries allow:
# max_depth=4 -> up to 16 leaves (~num_leaves=15); min_child_weight=200 == min_child_samples
# for the squared-error objective (hessian is 1 per row); same lr/subsample/colsample/lambda.
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=200,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=42, n_jobs=-1, verbosity=0,
)


def train_residual_model(X_train, y_train, X_val=None, y_val=None, params=None):
    p = dict(DEFAULT_PARAMS); p.update(params or {})
    model = lgb.LGBMRegressor(**p)
    if X_val is not None:                        # early stopping: watch a held-out slice
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X_train, y_train)
    return model

def train_residual_model_xgb(X_train, y_train, X_val=None, y_val=None, params=None):
    """XGBoost counterpart of train_residual_model, matched config + early stopping."""
    p = dict(XGB_PARAMS); p.update(params or {})
    if X_val is not None:
        p["early_stopping_rounds"] = 50
        model = xgb.XGBRegressor(**p)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = xgb.XGBRegressor(**p)
        model.fit(X_train, y_train)
    return model


def train_quantile_model(X_train, y_train, tau, X_val=None, y_val=None, params=None):
    """Quantile regression at level tau (pinball objective)."""
    p = dict(DEFAULT_PARAMS); p.update(objective="quantile", alpha=tau); p.update(params or {})
    model = lgb.LGBMRegressor(**p)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X_train, y_train)
    return model