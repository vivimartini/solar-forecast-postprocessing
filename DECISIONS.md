# Decisions & experiment log

## Framing
- Treating this as forecast POST-PROCESSING: correct the supplied forecast, baseline = raw forecast.
- Scope: metric (a) RMSE correction, operational lead 0–36h. Metric (b) as extension.

## Data
- Forecast is GW, actuals MW → ×1000 to reconcile (peaks line up ~53–56 GW). Verified, not assumed.
- Lead = step − issued_at (operational), NOT step − init_time. Found some negative-lead rows → dropped.

## Baseline (to beat)
- Daytime RMSE by lead band: 0–6h ~1478 · 6–12h ~1562 · 12–24h ~1705 · 24–36h ~1841 MW.

## EDA diagnostics (scripts/01_eda.py → outputs/eda_diagnostics.png)
- Coverage: 80,745 daytime rows, op-lead cleanly 0–36h, valid times 2023-01 → 2026-06.
- Fleet growing: 99.9th-pct actual 41.0→46.4→50.5→55.9 GW (2023→2026). Report SKILL vs baseline, not raw MW.
- Small positive bias everywhere (+39→+84 MW) → forecast slightly under-predicts (fc/actual ratio ~0.99). Calibration gain available.
- Error structure to exploit: (1) horizon — monotonic 1478→1841; (2) hour — peaks 10–11h UTC ~2270 MW (morning ramp); (3) season — spring worst (Feb–Apr ~1900–2020), summer calmest (Jul–Aug ~1340).

## Layer 1 — residual GBDT (scripts/03_train.py, src/models.py)
- Target = residual (actual − forecast, MW); LightGBM on issue-time-safe features; rolling-origin folds.
- HONEST FIRST RESULT: mean skill −11.3% (std 9.3%), positive in only 1/5 folds. Model makes it WORSE.
  Per fold: 1 −27.9% · 2 −10.8% · 3 −10.1% · 4 −8.3% · 5 +0.7%.
- Not leakage (opposite of the +30% red flag). Skill improves MONOTONICALLY with train size
  (10.7k→52.6k rows) → non-stationarity, not a bug: MW-level residual mapping learned on a smaller
  fleet mis-sizes corrections on a larger one; season mismatch across folds; no early-stopping overfit
- FIX = regularise + early stopping (num_leaves 31→15, min_child_samples 50→200, lr 0.05→0.03,
  reg_lambda 5, subsample/colsample 0.7; inner val = most-recent 15% of each fold's train, patience 50).
  → mean skill +0.9% (std 2.4%), positive 3/5 folds: 1 −0.4 · 2 +1.4 · 3 −2.2 · 4 +0.6 · 5 +5.1.
  Overfit was most of the earlier damage; fold-to-fold std collapsed 9.3→2.4%. First real positive skill.
- STILL NEXT: capacity-normalised target (residual as fraction of capacity/clear-sky proxy) so the
  mapping is stationary as the fleet grows; still-negative folds 1 & 3 are season/fleet mismatches.
- GBDT choice (scripts/06_xgb_compare.py): LightGBM vs XGBoost on a matched config (max_depth 4 ≈
  num_leaves 15, min_child_weight 200 == min_child_samples, same lr/subsample/colsample/lambda, ES 50).
  LightGBM +0.88% vs XGBoost +0.38% mean; LightGBM wins ALL 5 folds, same 3/5 positive, similar std.
  Same shape (both stumble on folds 1 & 3). DECISION: keep LightGBM. Caveat: configs matched not
  identical (leaf-wise vs depth-wise) + gap small (<1% abs) → library choice is not the big lever.

## Signature idea — lagged-ensemble dispersion (src/lagged_ensemble.py)
- Prior runs for a valid hour = a time-lagged ensemble; their spread (std of latest k=4 PRIOR members,
  shift(1) so leak-safe) = uncertainty signal. disp_mw covers 80,722/80,745 daytime rows.
- Tested airtight: Spearman(spread, |error|) 0.376 (0.402 daytime, k=4); 0.178 controlling for level; robust across seasons.
- Honest null: revision *direction* weakly predictable (AUC 0.554). Spread variants: simple de-bias didn't beat raw (TODO: variance-normalised).
- RESOLVED (scripts/04_ablation.py): dispersion does NOT help OUT-OF-SAMPLE POINT skill.
  base +0.83% vs base+disp +0.45% mean; negative on 4/5 fold-deltas. Expected: it tracks error
  MAGNITUDE (|residual|), not DIRECTION, so it can't sharpen the mean — pure noise to the point model.
- DECISION: keep POINT model on BASE_FEATURES; route disp_mw to the INTERVAL model (metric b:
  pinball/coverage), where an error-magnitude signal is exactly what sizes the interval width.
- Repro note: add_dispersion sorts by (step, issued_at) before reset_index → changes row order → shifts
  LightGBM's bagged subsample slightly (base drifts +0.9%→+0.83%). Benign; sort to canonical order to pin.

## Layer 2 — intervals (P10/P50/P90) & calibration
- Raw quantile GBDT (scripts/05_intervals.py): pinball 391.7 | P10-90 coverage 71.9% | width 2933 (base).
  Intervals TOO NARROW — 71.9% vs 80% nominal (undercover ~8pts). disp_mw again null (391.7→390.2, cov/width flat).
- CQR conformal (scripts/06_conformal.py; fit/ES/cal = 70/15/15% of each fold's train by issue time):
  pinball 393.5 | coverage 74.1% | width 3254 (base); base+disp 390.0 | 74.0% | 3247.
- CQR helps but UNDER-DELIVERS: coverage 71.9→74.1% (still <80), width +11%, pinball slightly worse
  (sharpness↔coverage tradeoff). Root cause = SAME non-stationarity: calibration slice is PAST relative
  to the val block, so time drift breaks conformal exchangeability → Q too small for the future.
- disp_mw STILL null on calibrated intervals too (pinball −0.9%, coverage/width flat) → drop it for good.
- NEXT for 80% coverage: weighted/adaptive conformal (recency-weight cal residuals) or capacity-normalised
  target (aligns past/future dists, helps point skill AND exchangeability). Drift inflation = band-aid.

## Capacity normalisation (scripts/08_capacity.py)
- Fleet proxy = trailing 60d peak of the forecast (shift(1), leak-safe). residual_norm = residual/cap.
- CV: MW target +0.88% → normalised +1.09%. Helps the fleet-growth folds (1,2,4); fold 3 (season) still negative.
- Walk-forward eval: normalised −14.79% vs raw-MW −9.79% — normalisation amplifies but doesn't cause the failure.

## Rich features (scripts/11_features.py)
- Shape: fc_ramp, fc_curv, fc_over_cs + clearsky_ghi. CV: base +1.09% → rich +1.30%.
- add_forecast_shape audited: 0 mismatches vs time-based truth on 1.74M rows (positional shift is fine here).
- Walk-forward eval: rich vs base only 0.4pp apart on the static GBDT — not the −15.2% culprit.

## Metric (b) — revision / predictive intervals (scripts/metric_b.py)
- next_revision = next issuance's fc − current fc, same valid hour. Direction weakly predictable (AUC ≈ 0.55 via scripts/16_significance.py), not a pure coin flip.
- Spearman(disp, |next revision|) = 0.443. Dispersion helps |revision| MAE: 222.7 → 219.9 MW.
- **Revision quantiles (P10/P50/P90 of next_revision):** base pinball 107.9, coverage 77.4%, width 919 MW;
  base+disp 107.3, coverage 78.0%, width 940 MW. Dispersion helps here (unlike actuals intervals).
- Answer to brief: symmetric predictive intervals on the next update, sized by instability; direction only weakly predictable.

## Walk-forward evaluation — final deliverable (scripts/10_final_test.py)
- Window: 2025-09-24 → 2026-05-31, 14,284 daytime rows. Chronological walk-forward backtest — informed final design; not an untouched holdout.
- Static GBDT (rich, normalised): −15.18% skill. Diagnosis: bias flips +152 MW (dev) → −320 MW (eval). See outputs/fig_bias_drift.png.
- Online per-hour bias (60d window): **+1.28%** skill. Block-bootstrap 95% CI **[−0.72%, +3.23%]** — not statistically significant (scripts/16_significance.py). Positive 54% of issue-days.
- GBDT on de-biased residual: **−2.53%** (still hurts OOS).
- Intervals (online ACI + dispersion): **79.1%** coverage.
- DECISION: ship online bias only for point forecast. Robust claim = avoiding static −15.2%, not a significant point gain over raw.
- Conditional coverage calm/mid/volatile: 82.1% / 78.5% / 76.8%. |error| by instability: 801 → 1412 → 1725 MW.

## GBDT on de-biased residual (scripts/15_debias_gbdt.py)
- Target = residual_mw − online_bias. CV: mixed (+0.3–1.8pp on some folds over bias alone).
- Walk-forward eval: bias +1.28% → bias+GBDT −2.53%. Conditional structure still doesn't transfer OOS.
- DECISION: point deliverable = online bias only; GBDT layer kept as documented negative result.

## Significance (scripts/16_significance.py)
- Block-bootstrap (issue-day resampling, n=2000): skill +1.28%, 95% CI [−0.72%, +3.23%], positive 54% of days. CI includes zero.
- Revision direction classifier AUC 0.554 (weakly predictable, not unpredictable).

## Conformal arc (scripts/06, 09, 13, 13b, 14)
- Raw quantile intervals: 71.9% coverage (undercover). Fleet-normalised target alone → 77.1% without conformal.
- Offline CQR: 74.1% — cal slice in the past, drift breaks exchangeability. 13b: Q < 0 on folds 1-2 (narrows when should widen).
- Dispersion-scaled offline conformal (13): made conditional coverage WORSE (volatile 0.725 → 0.651).
- Online ACI (14): 83.2% marginal; dispersion-scaled 82.9% with better conditional balance (volatile 0.776 → 0.796).