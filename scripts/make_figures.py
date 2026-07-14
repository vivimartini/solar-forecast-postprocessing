import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = pd.read_csv("outputs/predictions.csv", parse_dates=["issued_at", "step"])
p["date"] = p.step.dt.floor("D")
p["abserr"] = (p.actual_mw - p.corrected).abs()
p["width"] = p.p90 - p.p10

dd = p.groupby("date")["disp"].mean()
days = [(dd.idxmin(), "a settled day"), (dd.idxmax(), "a volatile day")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, (d, title) in zip(axes, days):
    g = p[p.date == d].sort_values("step")
    ax.fill_between(g.step, g.p10, g.p90, color="#5ea3a3", alpha=.28, label="P10–P90 band")
    ax.plot(g.step, g.corrected, color="#2c6e6e", lw=2, label="model forecast")
    ax.plot(g.step, g.fc_mw, "--", color="#e8a33d", lw=1.4, label="raw forecast")
    ax.plot(g.step, g.actual_mw, "o", color="#c0392b", ms=4, label="actual")
    ax.set_title(f"{title} — {d.date()}"); ax.grid(alpha=.25); ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("generation (MW)"); axes[0].legend(fontsize=8)
fig.suptitle("The forecast and its confidence band — tight when settled, wide when volatile")
fig.tight_layout(); fig.savefig("outputs/fig_fan_charts.png", dpi=120)

cov10, cov90 = (p.actual_mw <= p.p10).mean(), (p.actual_mw <= p.p90).mean()
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.plot([0, 1], [0, 1], "--", color="#888", label="perfectly calibrated")
ax.plot([0.1, 0.9], [cov10, cov90], "o-", color="#2c6e6e", ms=8)
for x, y in [(0.1, cov10), (0.9, cov90)]: ax.annotate(f"{y:.0%}", (x, y), (x+.03, y-.05), fontsize=9)
ax.set(xlabel="stated (nominal) level", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1),
       title="Calibration: the model's stated confidence ≈ reality")
ax.legend(fontsize=9); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig("outputs/fig_reliability.png", dpi=120)

p["band"] = pd.qcut(p.width, 3, labels=["narrow\n(confident)", "medium", "wide\n(uncertain)"])
by = p.groupby("band", observed=True)["abserr"].mean()
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.bar(range(3), by.values, color=["#4b8f8f", "#6b8cce", "#c0392b"])
ax.set_xticks(range(3)); ax.set_xticklabels(by.index, fontsize=9)
for i, v in enumerate(by.values): ax.text(i, v, f"{v:.0f} MW", ha="center", va="bottom", fontsize=10)
ax.set(ylabel="actual forecast error (MW)",
       title="The uncertainty is meaningful:\nwhen the model's band is wide, the forecast really is less accurate")
ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig("outputs/fig_uncertainty_informative.png", dpi=120)

p["ym"] = p.step.dt.to_period("M").astype(str)
monthly = []
for ym, g in p.groupby("ym"):
    base = np.sqrt(np.mean((g.actual_mw - g.fc_mw) ** 2))
    mod = np.sqrt(np.mean((g.actual_mw - g.corrected) ** 2))
    monthly.append((ym, 100 * (1 - mod / base), len(g)))
monthly.sort()
fig, ax = plt.subplots(figsize=(7, 3.8))
xs = np.arange(len(monthly))
skills = [m[1] for m in monthly]
colors = ["#2c6e6e" if s >= 0 else "#c0392b" for s in skills]
ax.bar(xs, skills, color=colors)
ax.axhline(0, color="#888", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([m[0] for m in monthly], rotation=35, ha="right")
for i, (_, sk, n) in enumerate(monthly):
    ax.text(i, sk + (1.5 if sk >= 0 else -1.5), f"{sk:+.1f}%", ha="center", va="bottom" if sk >= 0 else "top", fontsize=8)
ax.set(ylabel="point skill vs raw (%)",
       title="Sealed-test skill by month — positive early, drifts negative through 2026")
ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig("outputs/fig_monthly_skill.png", dpi=120)

print("saved 4 figures to outputs/")
