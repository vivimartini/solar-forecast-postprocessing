# DECISIONS — working log

Data first. Forecasts peak ~55, actuals ~55,000, so forecasts are GW and actuals MW. ×1000. Peaks line
up after that.

Actuals are 15-min snapshots, forecast is per-hour, so averaged actuals to hourly (mean).

Two lead columns — init_time (weather model) and issued_at (published). Used `step − issued_at` since
that's when you can actually act on it. Some rows come out negative (valid time already gone), dropped.

Baseline is the raw forecast, not climatology. It's already good, I'm just shaving the residual.

The empirical upper generation envelope grows from roughly 41 to 56 GW over the sample (99.9th-pct
actual by year), consistent with an increasing effective generation scale. Raw MW errors aren't
comparable across time — same size error looks tiny early and huge late, just from scale. Reporting
relative RMSE skill vs the contemporaneous forecast instead of raw MW.

Scope: brief wants 0–36h or 168–360h. Ran the correction on both — helps at short (+1.28%), hurts at long
(−1.81%). The tested adaptive correction does not find stable transferable residual structure at long
range, so I scoped the main analysis to 0–36h.

Daytime = solar elevation > 5° (pvlib). Wanted something geometric so nothing from the actuals leaks in.

Eval setup. CV is rolling-origin split on issue time — train on the past, predict forward. Random k-fold
would leak the future into the past, useless here. Put a 36h embargo between train and validation so a
lagged feature can't straddle the split (sized to the max feature lookback). Then held the last 20% of
issue dates as a chronological walk-forward, only looked at it near the end. Being honest though — it did
inform the final method choice, so it's an operational backtest, not a clean untouched holdout.
For the online bias I assume yesterday's ENTSO-E actuals are in by the next issue day — that's the latency
the shift(1) encodes.

Tried LightGBM on the residual. +0.9% in CV, looked fine. Then −15.2% on the walk-forward. Ouch.
Took a while to see why. The temporal CV was leakage-safe, but the later failure is dominated by target
drift: the mean residual changes from +152 MW in development to −320 MW in evaluation, so the frozen
correction acts in the wrong direction. Frozen model learns "forecast runs low, add MW", then the
forecast starts running high and it corrects the wrong way. That's basically the whole finding.

So dropped the frozen model, just track the recent bias instead — trailing 60d mean of residuals,
shift(1) so today doesn't see itself. Did it per hour-of-day since the drift isn't uniform through the
day. Per-hour beats a single global term (+1.28 vs +1.15). −15.2% → +1.28% just from making it causal +
adaptive.

Why 60d? Judgement call. Later swept 14/30/45/60/90/120 — short too noisy (14d goes negative), long lags
the drift, 30–60 all fine. Kept 60 for stability. Would rather learn the window (EWMA / state-space),
noting for later.

Is +1.28% even real? Block bootstrap over issue-days, 95% CI [−0.72%, +3.23%], positive on 54% of days.
So no, not significant. Not selling it as a point win. The actual win is not blowing up like the static
model.

Tried to get more out of the point forecast, nothing worked:
- GBDT on the de-biased residual: −2.53%. Leftover structure drifts too.
- Capacity-normalising the residual first (divide by a capacity proxy — running upper envelope of
  generation): no help.
- Physics features (clear-sky, geometry): did not add robust out-of-sample value. Plausible because the
  supplied NWP-driven forecast already encodes much of the deterministic solar structure, while the
  remaining residual is dominated in this sample by non-stationary bias and scale effects.
So point deliverable is just the online bias.

Physical bounds: the additive correction can push the point forecast slightly below 0 at dawn/dusk
(~0.02% of rows), and P10 more often. Clip everything to [0, capacity].

Uncertainty (metric b, and the better half). Quantile GBDT for P10/P90 (80% central interval); the base
layer under-covers. Static offline conformal restores near-nominal coverage on walk-forward eval
(81.2%) but wider intervals and worse interval score. Online ACI hits 79.1% with a substantially lower
interval score — better calibration–sharpness trade-off under drift. Dispersion scaling on top. Width
~4.45 GW. Sort quantiles so they don't cross. Coverage by regime 82 / 79 / 77, roughly even. Width is
informative: bucket by width and MAE goes 643 → 1327 → 1923. Tells you when to distrust the forecast.

Metric (b) target: for each valid time, revision = next issuance − current issuance. Modelled the
absolute magnitude of that and put a symmetric band around the currently issued forecast.

Dispersion / lagged ensemble — successive runs for the same hour disagree, maybe a reliability signal.
Tested three ways:
- point-correction feature: no.
- direct feature in the interval model: redundant (mostly just generation level).
- scaling signal in calibration + revision magnitude: yes, this is where it works.
For metric (b) it predicts revision magnitude (Spearman 0.44, ~0.29 after controlling for level).
Direction barely predictable (AUC 0.554) so dropped that, only ship the magnitude band (78% cover,
0.94 GW).

Climatology — caught a leak. Was computing it full-sample, which uses future actuals. Redid it expanding,
only valid times before each issue day, same latency rule as the bias. Skill vs the causal version: +73%
at 0–36h, +26 / +17 at the long bands. `scripts/climatology.py`.

LightGBM vs XGBoost: +0.88 vs +0.38, kept LightGBM. Minor.
Unit-tested the metrics + split leak-safety.

Where it landed: point gain marginal and not significant, real story is avoiding the static failure, the
uncertainty layer is the stronger deliverable. All reproducible from the scripts.

Still open / will do:
- proper untouched holdout (walk-forward informed the method choice, so it's not clean)
- recency-weighted conformal for the last point of coverage
- state-space / Kalman bias tracker instead of the fixed 60d mean