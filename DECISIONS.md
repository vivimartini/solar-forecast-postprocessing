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

## Signature idea — lagged-ensemble dispersion (src/lagged_ensemble.py)
- Prior runs for a valid hour = a time-lagged ensemble; their spread (std of latest k=4 PRIOR members,
  shift(1) so leak-safe) = uncertainty signal. disp_mw covers 80,722/80,745 daytime rows.
- Tested airtight: Spearman(spread, |error|) 0.376 (0.402 daytime, k=4); 0.178 controlling for level; robust across seasons.
- Honest null: revision *direction* ≈ 0. Spread variants: simple de-bias didn't beat raw (TODO: variance-normalised).
- RESOLVED (scripts/04_ablation.py): dispersion does NOT help OUT-OF-SAMPLE POINT skill.
  base +0.83% vs base+disp +0.45% mean; negative on 4/5 fold-deltas. Expected: it tracks error
  MAGNITUDE (|residual|), not DIRECTION, so it can't sharpen the mean — pure noise to the point model.
- DECISION: keep POINT model on BASE_FEATURES; route disp_mw to the INTERVAL model (metric b:
  pinball/coverage), where an error-magnitude signal is exactly what sizes the interval width.
- Repro note: add_dispersion sorts by (step, issued_at) before reset_index → changes row order → shifts
  LightGBM's bagged subsample slightly (base drifts +0.9%→+0.83%). Benign; sort to canonical order to pin.