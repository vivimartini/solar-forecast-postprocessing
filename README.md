# Solar forecast post-processing — Germany (0–36h)

Post-processing a solar generation forecast for Germany against ENTSO-E actuals: correct remaining
systematic error where possible, and attach honest **predictive intervals** (metric a + b).

**Headline:** a static GBDT correction looks fine in CV then scores **−15.2%** on a chronological
walk-forward evaluation — bias drift. An adaptive per-hour bias tracker recovers that to **+1.28%**
vs the raw forecast, but a block bootstrap shows that gain is **not statistically significant**
(95% CI includes zero). The robust result is **avoiding the static-model failure**; the uncertainty
layer performs more robustly in the walk-forward backtest, reaching **79.1%** P10–P90 coverage.

## Scope

**0–36h operational lead** for point correction. Metric (b) — bounds on the next forecast revision —
as extension.

Post-processing only removes *systematic* error. At 7–15 days the raw forecast still beats causal
historical climatology and retains correlations of approximately **0.89–0.92** with actuals, although
these correlations are partly inflated by shared diurnal and seasonal structure. A learned correction
**hurts** at that range because there is no stable transferable structure left to learn:

| lead | correction skill (walk-forward eval) |
|------|--------------------------------------|
| 0–36h | +1.28% (marginal; CI includes 0) |
| 168–360h | −1.81% (hurts) |

## Definitions

- **Operational lead:** `step − issued_at` (when the forecast is usable), not model init time.
- **Daytime:** solar elevation > 5° (geometry-based, not outcome-dependent).
- **Climatology baseline:** expanding month × hour mean of actuals at valid times **strictly before
  issue day** (same latency rule as online bias; see `add_climatology_baseline` in `src/data.py`).
- **Actuals latency:** online bias uses only fully realised past valid days (`shift(1)` on valid time);
  a forecast issued on day D sees residuals through D−1.

## Data handling

- Forecasts GW, actuals MW → ×1000.
- Actuals are 15-min snapshots; hourly mean to match forecast definition.
- The empirical upper generation envelope grows from roughly **41 to 56 GW** over the sample (99.9th-pct
  actual by year), consistent with an increasing effective generation scale. Skill is reported against
  the contemporaneous raw forecast alongside absolute MW errors.

## Evaluation

Rolling-origin CV on issue time with embargo, plus a **chronological walk-forward backtest** on the
final 20% of issue dates (from **24 Sep 2025** through **31 May 2026**, 14,284 daytime rows). That
period informed the final method choice — it is not an untouched holdout. For a clean final number,
freeze the procedure and evaluate once on a later slice (not done here).

## The main result

Regularised LightGBM on the residual: **+0.9% CV, −15.2% walk-forward** — bias flips from **+152 MW**
(dev) to **−320 MW** (eval). See `outputs/fig_bias_drift.png`.

**Adaptive per-hour bias (60d window):** +1.28% RMSE skill on the walk-forward period.
Block-bootstrap 95% CI: **[−0.72%, +3.23%]**; positive on **54%** of issue-days — **not distinguishable
from zero**. The value is not overclaiming a point gain; it is that adaptivity **avoids −15.2%**.

GBDT on the de-biased residual: **−2.53%** — still hurts OOS. Point deliverable = online bias only.

**Predictive intervals:** the quantile-GBDT base layer under-covers. Static offline conformal restores
near-nominal coverage on the walk-forward evaluation (**81.2%**) but produces wider intervals and a
worse interval score. Online ACI achieves comparable near-nominal coverage (**79.1%** against an 80%
target) with a substantially lower interval score, giving the better calibration–sharpness trade-off
under drift. With dispersion scaling, mean width is **4,452 MW**; interval width also ranks realised
difficulty, with MAE of 643 / 1,327 / 1,923 MW across narrow / medium / wide terciles
(see `fig_uncertainty_informative.png`).

## Results (walk-forward evaluation)

| | |
|---|---|
| point RMSE skill vs raw | +1.28% (95% CI [−0.72%, +3.23%]) |
| days with positive skill | 54% |
| by lead band (0–6 / 6–12 / 12–24 / 24–36h) | +1.72 / +0.89 / +1.33 / +1.24% |
| P10–P90 coverage | 79.1% |
| coverage by regime (calm / mid / volatile) | 82.1% / 78.5% / 76.8% |
| revision direction | weakly predictable (AUC 0.554); magnitude is the useful signal |
| revision quantiles (P10–P90 of next update, base+disp) | 78.0% coverage, 940 MW width |
| raw forecast \|error\| by instability (calm → volatile) | 801 → 1,412 → 1,725 MW |

## Metric (b) — two uncertainty targets

**A. Outcome uncertainty** (mean P10–P90 width **4.45 GW** on walk-forward eval): where will actual
generation land?

**B. Revision uncertainty** (P10–P90 width **0.94 GW**, base+dispersion): how much will the *next
issuance* move the forecast? Direction only weakly predictable (AUC 0.554); dispersion predicts
magnitude (Spearman ρ = 0.44; partial ρ ≈ 0.29 controlling for level).

## How to run

```bash
pip install -r requirements.txt
# put forecasts.parquet + actuals.parquet in data/

PYTHONPATH=. python scripts/10_final_test.py     # walk-forward eval + predictions.csv
PYTHONPATH=. python scripts/16_significance.py   # bootstrap CI + direction AUC
PYTHONPATH=. python scripts/make_figures.py      # report figures
PYTHONPATH=. python -m pytest tests/ -q

./scripts/run_submission.sh
```

Experiment trail: `scripts/03`–`16`, `metric_b.py`. `DECISIONS.md` is the lab notebook.

## Repo map

```
src/          data loading, features, models, splits, online_bias, dispersion
scripts/      experiment scripts + 10_final_test + 16_significance + make_figures
tests/        metrics + leak-safety
config.yaml   paths, lead band, validation, online_bias
outputs/      figures + predictions.csv (gitignored)
```

## What I'd do next

- Genuinely untouched final holdout (~2 months) with frozen procedure.
- Recency-weighted conformal to close the last ~1pt of coverage.
- State-space / Kalman bias tracker instead of rolling mean.

