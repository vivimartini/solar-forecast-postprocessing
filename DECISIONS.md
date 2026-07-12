# Decisions & experiment log

## Framing
- Treating this as forecast POST-PROCESSING: correct the supplied forecast, baseline = raw forecast.
- Scope: metric (a) RMSE correction, operational lead 0–36h. Metric (b) as extension.

## Data
- Forecast is GW, actuals MW → ×1000 to reconcile (peaks line up ~53–56 GW). Verified, not assumed.
- Lead = step − issued_at (operational), NOT step − init_time. Found some negative-lead rows → dropped.

## Baseline (to beat)
- Daytime RMSE by lead band: 0–6h ~1478 · 6–12h ~1562 · 12–24h ~1705 · 24–36h ~1841 MW.

## Signature idea — lagged-ensemble dispersion
- Prior runs for a valid hour = a time-lagged ensemble; their spread = uncertainty signal.
- Tested airtight: Spearman(spread, |error|) 0.376; 0.178 controlling for level; robust across seasons.
- Honest null: revision *direction* ≈ 0. Spread variants: simple de-bias didn't beat raw (TODO: variance-normalised).
- OPEN QUESTION: does spread improve OUT-OF-SAMPLE pinball? 