"""Block-bootstrap CI for point skill + revision-direction AUC."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.data import load_config, build_dataset
from src.online_bias import add_online_bias
from src.splits import train_test_seal

N_BOOT = 2000


def block_bootstrap_skill(te):
    te = te.copy()
    te["corrected"] = np.clip(te.fc_mw + te.online_bias, 0, None)
    te["eb2"] = (te.actual_mw - te.fc_mw) ** 2
    te["em2"] = (te.actual_mw - te.corrected) ** 2
    g = (te.assign(iday=te.issued_at.dt.floor("D"))
           .groupby("iday")
           .agg(eb=("eb2", "sum"), em=("em2", "sum"), n=("eb2", "size")))

    point = (1 - np.sqrt(g.em.sum()) / np.sqrt(g.eb.sum())) * 100
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(g), len(g), replace=True)
        boots.append(1 - np.sqrt(g.em.values[idx].sum()) / np.sqrt(g.eb.values[idx].sum()))
    boots = np.array(boots) * 100
    day_skill = 1 - np.sqrt(g.em / g.n) / np.sqrt(g.eb / g.n)
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5), 100 * (day_skill > 0).mean()


def direction_auc(day, test_mask):
    d = day.dropna(subset=["next_revision", "disp_mw"]).copy()
    d["up"] = (d.next_revision > 0).astype(int)
    d["month"] = d.step.dt.month
    d["hour"] = d.step.dt.hour
    cut = day.loc[test_mask, "issued_at"].min()
    feats = ["disp_mw", "fc_mw", "op_lead_h", "hour", "month"]
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        min_child_samples=200, reg_lambda=5, random_state=0, verbose=-1,
    )
    tr = d[d.issued_at < cut]
    ts = d[d.issued_at >= cut]
    clf.fit(tr[feats], tr["up"])
    return roc_auc_score(ts["up"], clf.predict_proba(ts[feats])[:, 1])


def main():
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].reset_index(drop=True)
    ob = cfg["online_bias"]
    day = add_online_bias(day, window_days=ob["window_days"], per_hour=ob["per_hour"])

    _, test_mask = train_test_seal(day, cfg["validation"]["sealed_test_frac"])
    te = day[test_mask]

    skill, lo, hi, pct_days = block_bootstrap_skill(te)
    auc = direction_auc(day, test_mask)

    print("=== walk-forward evaluation period ===")
    print(f"point skill (RMSE): {skill:+.2f}%")
    print(f"block-bootstrap 95% CI: [{lo:+.2f}%, {hi:+.2f}%]  (issue-day resampling, n={N_BOOT})")
    print(f"days with positive skill: {pct_days:.0f}%")
    print(f"revision direction AUC: {auc:.3f}  (0.50 = unpredictable)")


if __name__ == "__main__":
    main()
