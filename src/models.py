# src/models.py
"""Models. Layer 1: a gradient-boosted residual correction (LightGBM).
Regularised + early stopping so it keeps only strong, repeatable patterns.
"""
import lightgbm as lgb

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


def train_residual_model(X_train, y_train, X_val=None, y_val=None, params=None):
    p = dict(DEFAULT_PARAMS); p.update(params or {})
    model = lgb.LGBMRegressor(**p)
    if X_val is not None:                        # early stopping: watch a held-out slice
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X_train, y_train)
    return model