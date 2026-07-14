# src/models.py
"""LightGBM residual/quantile models, plus an XGBoost twin for the library comparison.

First attempt used near-default params (num_leaves=31, min_child_samples=50, no early
stopping) and overfit badly -- see DECISIONS.md. Current params are deliberately heavy
on regularisation: this dataset rewards caution, not capacity.
"""
import lightgbm as lgb
import xgboost as xgb

DEFAULT_PARAMS = dict(
    objective="regression",
    n_estimators=3000,          # just a ceiling, early stopping decides
    learning_rate=0.03,
    num_leaves=15,              # was 31 -> memorised the training folds
    min_child_samples=200,      # don't trust a split seen on <200 rows
    subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=42, n_jobs=-1, verbose=-1,
)

# Matched to DEFAULT_PARAMS as closely as the two APIs allow: max_depth=4 gives up to
# 16 leaves (~num_leaves 15), and for squared error the hessian is 1/row so
# min_child_weight=200 is the same constraint as min_child_samples=200.
# Not a perfect match (leaf-wise vs depth-wise growth) -- noted in DECISIONS.md.
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
    if X_val is not None:
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