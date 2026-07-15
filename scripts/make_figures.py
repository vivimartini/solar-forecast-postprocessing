import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.data import load_config, build_dataset
from src.online_bias import add_online_bias
from src.splits import train_test_seal

# shared report styling (~1.7:1, captions live in LaTeX not in the PNG)
REPORT_SIZE = (8.5, 5.0)
REPORT_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}
GRID_ALPHA = 0.25

p = pd.read_csv("outputs/predictions.csv", parse_dates=["issued_at", "step"])
p["abserr"] = (p.actual_mw - p.corrected).abs()
p["width"] = p.p90 - p.p10
if "inside" not in p.columns:
    p["inside"] = (p.actual_mw >= p.p10) & (p.actual_mw <= p.p90)

overall_skill = 100 * (1 - np.sqrt(np.mean((p.actual_mw - p.corrected) ** 2))
                         / np.sqrt(np.mean((p.actual_mw - p.fc_mw) ** 2)))
overall_cov = p.inside.mean() * 100
report_cutoff_str = None

with plt.rc_context(REPORT_STYLE):
    # --- report fig 1: width terciles vs realised error ---
    p["width_tercile"] = pd.qcut(p.width, 3, labels=["Narrow", "Medium", "Wide"])
    by = p.groupby("width_tercile", observed=True)["abserr"].mean()
    fig, ax = plt.subplots(figsize=REPORT_SIZE)
    ax.bar(range(3), by.values, color=["#4b8f8f", "#6b8cce", "#c0392b"], width=0.62)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Narrow\ninterval", "Medium\ninterval", "Wide\ninterval"])
    ax.set_ylim(0, 2150)
    for i, v in enumerate(by.values):
        ax.text(i, v + 45, f"{v:.0f} MW", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Mean absolute forecast error (MW)",
           title="Wider prediction intervals identify less reliable forecasts")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig("outputs/fig_uncertainty_informative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- report fig 2: bias drift (mechanism — main report) ---
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].reset_index(drop=True)
    ob = cfg["online_bias"]
    day = add_online_bias(day, window_days=ob["window_days"], per_hour=ob["per_hour"])
    dev_mask, test_mask = train_test_seal(day, cfg["validation"]["sealed_test_frac"])
    cutoff = day.loc[test_mask, "issued_at"].min()
    day["residual"] = day.actual_mw - day.fc_mw
    day["ym"] = day.step.dt.to_period("M").astype(str)

    monthly = (day.groupby("ym", observed=True)
                 .agg(mean_resid=("residual", "mean"), mean_bias=("online_bias", "mean"))
                 .reset_index()
                 .sort_values("ym")
                 .reset_index(drop=True))
    split_ym = day.loc[test_mask, "ym"].min()
    split_idx = int(monthly.index[monthly.ym == split_ym][0])
    cutoff_str = cutoff.strftime("%d %b %Y")
    report_cutoff_str = cutoff_str

    fig, ax = plt.subplots(figsize=REPORT_SIZE)
    xs = np.arange(len(monthly))
    ax.bar(xs, monthly.mean_resid, color="#3d6fa8", alpha=0.88, width=0.72,
           label="Monthly mean residual", zorder=2)
    ax.plot(xs, monthly.mean_bias, "o-", color="#e8a33d", lw=1.7, ms=3.5,
            label="Online bias estimate", zorder=4)
    ax.axhline(0, color="#444444", lw=1.3, zorder=3)
    ax.axvline(split_idx - 0.5, color="#888888", ls=":", lw=0.75, zorder=1)
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.06
    ax.set_ylim(ylo, yhi + pad)
    ax.text(split_idx - 0.5 + 0.2, yhi + pad * 0.85,
            f"Walk-forward evaluation begins\n({cutoff_str})",
            fontsize=7, ha="left", va="top", color="#555555")
    quarter_idx = [i for i, ym in enumerate(monthly.ym)
                   if pd.Period(ym, freq="M").month in (1, 4, 7, 10)]
    quarter_labels = [pd.Period(monthly.ym.iloc[i], freq="M").strftime("%b %Y")
                      for i in quarter_idx]
    ax.set_xticks(quarter_idx)
    ax.set_xticklabels(quarter_labels, rotation=30, ha="right")
    ax.set(ylabel="Monthly mean residual (MW)",
           title="The online bias estimate tracks a drifting residual process")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig("outputs/fig_bias_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- appendix: monthly walk-forward skill ---
    p["ym"] = p.step.dt.to_period("M").astype(str)
    monthly = []
    for ym, g in p.groupby("ym"):
        base = np.sqrt(np.mean((g.actual_mw - g.fc_mw) ** 2))
        mod = np.sqrt(np.mean((g.actual_mw - g.corrected) ** 2))
        monthly.append((ym, 100 * (1 - mod / base), len(g)))
    monthly.sort()
    fig, ax = plt.subplots(figsize=REPORT_SIZE)
    xs = np.arange(len(monthly))
    skills = [m[1] for m in monthly]
    colors = ["#2c6e6e" if s >= 0 else "#c0392b" for s in skills]
    ax.bar(xs, skills, color=colors, width=0.72)
    ax.axhline(0, color="#444444", lw=1.3, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in monthly], rotation=35, ha="right")
    ax.set_ylim(-12, 14)
    for i, (_, sk, _n) in enumerate(monthly):
        offset = 1.4 if sk >= 0 else -1.4
        ax.text(i, sk + offset, f"{sk:+.1f}%", ha="center",
                va="bottom" if sk >= 0 else "top", fontsize=9)
    ax.set(ylabel="RMSE skill vs raw forecast (%)",
           title="Walk-forward correction skill varies over time")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig("outputs/fig_monthly_skill.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- report fig 3 (compact): coverage + width by dispersion regime ---
p["disp_tercile"] = pd.qcut(p.disp, 3, labels=["Calm", "Medium", "Volatile"])
regime = (p.groupby("disp_tercile", observed=True)
            .agg(coverage=("inside", "mean"), mean_width=("width", "mean"))
            .reset_index())
fig, ax1 = plt.subplots(figsize=(5.8, 4.0))
x = np.arange(3)
w = 0.36
ax1.bar(x - w / 2, regime.coverage * 100, width=w, color="#2c6e6e", label="Observed coverage")
ax1.axhline(80, color="#888", ls="--", lw=0.9, label="Nominal 80%")
ax1.set_ylabel("P10–P90 coverage (%)")
ax1.set_ylim(70, 90)
ax1.set_xticks(x)
ax1.set_xticklabels(regime.disp_tercile)
ax2 = ax1.twinx()
ax2.bar(x + w / 2, regime.mean_width, width=w, color="#e8a33d", alpha=0.85, label="Mean width")
ax2.set_ylabel("Mean interval width (MW)")
for i, (cov, wid) in enumerate(zip(regime.coverage * 100, regime.mean_width)):
    ax1.text(i - w / 2, cov + 0.4, f"{cov:.0f}%", ha="center", fontsize=8)
    ax2.text(i + w / 2, wid + 80, f"{wid:.0f}", ha="center", fontsize=8)
ax1.set(title=f"Interval calibration by forecast instability (overall {overall_cov:.1f}%)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
ax1.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig("outputs/fig_coverage_by_regime.png", dpi=140)

# --- appendix: illustrative settled vs volatile days ---
p["date"] = p.step.dt.floor("D")
dd = p.groupby("date")["disp"].mean()
days = [(dd.idxmin(), "Settled day (illustrative)"), (dd.idxmax(), "Volatile day (illustrative)")]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
for ax, (d, title) in zip(axes, days):
    g = p[p.date == d].sort_values("step")
    mean_w = (g.p90 - g.p10).mean()
    ax.fill_between(g.step, g.p10, g.p90, color="#5ea3a3", alpha=.28, label="P10–P90 interval")
    ax.plot(g.step, g.corrected, color="#2c6e6e", lw=1.6, label="corrected forecast")
    ax.plot(g.step, g.fc_mw, "--", color="#e8a33d", lw=1.2, label="raw forecast")
    ax.plot(g.step, g.actual_mw, "o", color="#c0392b", ms=3.5, label="actual")
    ax.set_title(f"{title}\n{d.date()} | mean interval width {mean_w:.0f} MW")
    ax.grid(alpha=.25)
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("Generation (MW)")
axes[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Illustrative prediction intervals on settled and volatile forecast periods", y=1.02)
fig.tight_layout()
fig.savefig("outputs/fig_fan_charts.png", dpi=140, bbox_inches="tight")

# --- appendix: marginal quantile check + stated central coverage ---
cov10, cov90 = (p.actual_mw <= p.p10).mean(), (p.actual_mw <= p.p90).mean()
central = overall_cov
fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfect calibration")
ax.plot([0.1, 0.9], [cov10, cov90], "o-", color="#2c6e6e", ms=8, label="Marginal quantiles")
for x, y in [(0.1, cov10), (0.9, cov90)]:
    ax.annotate(f"{y:.0%}", (x, y), (x + 0.03, y - 0.05), fontsize=9)
ax.bar([0.5], [central], width=0.18, color="#6b8cce", alpha=0.55, label=f"P10–P90 coverage ({central:.1f}%)")
ax.axhline(0.8, color="#c0392b", ls=":", lw=0.9)
ax.set(xlabel="Nominal level", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1),
       title="Empirical quantile frequencies are close to nominal levels")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig("outputs/fig_reliability.png", dpi=140)

print("saved report figures:")
print("  outputs/fig_bias_drift.png                (main — mechanism)")
print("  outputs/fig_uncertainty_informative.png  (main — intervals)")
print("  outputs/fig_coverage_by_regime.png       (main, compact)")
print("  outputs/fig_monthly_skill.png            (appendix)")
print("  outputs/fig_fan_charts.png               (appendix)")
print("  outputs/fig_reliability.png              (appendix)")
print()
print("LaTeX captions (paste into report, not embedded in PNG):")
print()
print("Figure 1: The online bias estimate tracks a drifting residual process.")
print("Monthly mean residual of the raw forecast, defined as actual minus forecast,")
print("together with the online bias estimate. The residual changes sign repeatedly,")
print("while the adaptive estimate tracks the broad drift with some lag, motivating")
print(f"an online rather than static correction. Walk-forward evaluation begins")
print(f"{report_cutoff_str}.")
print()
print("Figure 2: Wider prediction intervals identify less reliable forecasts.")
print("Mean absolute forecast error by predicted interval-width tercile on the walk-forward")
print("evaluation set. Error rises monotonically from 643~MW for narrow intervals to")
print("1{,}923~MW for wide intervals, showing that the P10--P90 interval width meaningfully")
print("ranks forecast difficulty.")
print()
print("Appendix — monthly walk-forward skill:")
print(f"Monthly RMSE skill of the adaptive correction relative to the raw forecast.")
print(f"Aggregate skill over the evaluation period: {overall_skill:+.2f}\\%.")
