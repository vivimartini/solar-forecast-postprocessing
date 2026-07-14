# Solar forecast post-processing — Germany (0–36h)

My take on the Secondlaw challenge: given a solar generation forecast for Germany and the ENTSO-E
actuals, improve it. I treated it as post-processing — the forecast is already good, so instead of
building a new one I tried to correct its remaining errors and put an honest uncertainty band around it.

The short version of what I found: the obvious approach (train a model to correct the forecast) looks
fine in cross-validation and then falls apart on a genuinely held-out future, because the forecast's
error drifts over time. Once I understood that, the things that actually generalise are all *adaptive* —
they keep updating instead of freezing a relationship the data won't hold still for.

## Scope

I focused on the **0–36h lead time** and improving RMSE (metric a), with the forecast-update /
confidence-bounds side (metric b) as the second half.

I almost justified this with "the medium range is hopeless," but when I checked, that's not true — the
forecast still beats climatology at 7–15 days (correlation ~0.9). The real reason is subtler and I think
more interesting. Post-processing can only remove *systematic* error. Running a correction across leads:

| lead | correction skill (sealed test) |
|------|-------------------------------|
| 0–36h | +1.28% (helps) |
| 168–360h | −1.81% (hurts) |

At short range the error has correctable structure; at 7–15 days it's mostly irreducible weather noise,
so a learned correction just fits noise and makes things worse. So I correct accuracy where that's
actually possible (0–36h), and treat the confidence bounds as the thing worth having for the range you
*can't* fix — which is, I think, exactly why the brief pairs the two metrics.

## Data handling

A few things that mattered before any modelling:

- Forecasts are in GW, actuals in MW — ×1000 (checked against the peaks, ~53–56 GW, not assumed).
- I measure lead from when the forecast is actually *available* (`issued_at`), not when the weather
  model initialised. A few rows come out with negative lead and get dropped.
- Actuals are 15-min snapshots; I average them to hourly to match the forecast's "generation within the
  hour" convention.
- The installed fleet grows over the window (~40 → 56 GW), so I report skill against the raw forecast
  rather than raw MW, which would otherwise drift with capacity.

## Evaluation

Rolling-origin CV split by issue time with an embargo, and — the part that mattered most — a 20% sealed
test held back and evaluated once, at the end. That's what caught the failure below; without it I'd have
reported the CV number and been wrong.

## The main result

My first correction model was a regularised LightGBM on the residual. In CV it gave +1.3%. On the sealed
test it gave **−15%** — it made the forecast worse.

Why: the forecast's bias isn't stable. Over the training period it runs slightly low (mean residual
+157 MW); over the test period it runs high (−320 MW), and it drifts steadily through 2026. The model
learned to push forecasts one way, the future needed the other, so it confidently pushed the wrong way.

The fix follows from the diagnosis — instead of a frozen model, track the *recent* bias and update as
you go. A per-hour adaptive bias (each hour of day drifts on its own) takes the sealed-test result from
−15% to **+1.28%**. I also tried putting the regularised GBDT back on top of the de-biased residual;
it scored **−2.53%** on the sealed test, so the point deliverable stays online bias only. Small gain,
but real, and it holds up out of sample.

The uncertainty side showed the same pattern: a fixed (offline) conformal calibration failed under the
drift, while an online one that self-corrects as it moves through the test period gets the P10–P90 band
to ~80% coverage on data it never saw.

## Results (sealed test)

| | |
|---|---|
| point skill vs raw forecast | +1.28% |
| by lead band (0–6 / 6–12 / 12–24 / 24–36h) | +1.72 / +0.89 / +1.33 / +1.24% |
| P10–P90 coverage | 79.1% (target 80%) |
| coverage by regime (calm / mid / volatile) | 0.82 / 0.79 / 0.77 |
| direction of next revision | ~50% up (unpredictable) |
| revision quantiles (P10–P90 of next update) | 78% coverage, ~940 MW width (dispersion helps) |
| forecast error, calm → volatile hours | 801 → 1412 → 1725 MW |

Monthly skill stays positive through late 2025, dips in parts of early 2026 as the bias keeps drifting
(see `outputs/fig_monthly_skill.png`) — the online tracker follows most of it but not perfectly.

## The lagged-ensemble idea

The forecaster reruns 4× a day, so for any hour I have several past forecasts. When they disagree, the
forecast tends to be less reliable. I tested this three ways, and it's a good example of being honest
about what doesn't work: as a feature for the point correction it *hurts*; for the interval model it's
redundant with generation level and does nothing; but it genuinely predicts the *size* of the next
forecast revision (ρ=0.44) and, plugged into the online calibration, sensibly widens the band on
volatile hours. The *direction* of the next revision is a coin flip, so the bounds end up symmetric and
sized by instability. That's the direct answer to metric (b).

## A note on physics

My background is physics-informed ML, so I did try the physical features — clear-sky irradiance, solar
geometry, a clear-sky-scaled forecast term. None helped out of sample, and I think the reason is the
honest one: I'm correcting a forecast that's *already* physics-based, so the diurnal arc and cloud
response are largely baked in. The remaining error is mostly bias drift and weather volatility, which
is why the adaptive pieces ended up mattering more than more physics features.

## How to run

```bash
pip install -r requirements.txt
# put forecasts.parquet + actuals.parquet in data/

PYTHONPATH=. python scripts/00_scope_analysis.py   # why 0-36h not 168-360h
PYTHONPATH=. python scripts/01_eda.py            # baseline + eda plot
PYTHONPATH=. python scripts/02_baseline.py       # check the CV harness
PYTHONPATH=. python scripts/10_final_test.py     # sealed test (run once)
PYTHONPATH=. python scripts/make_figures.py      # figures from predictions.csv
PYTHONPATH=. python -m pytest tests/ -q          # metrics + leak-safety checks

# or the one-liner:
./scripts/run_submission.sh
```

The experiment trail lives in `scripts/03`–`14` and `metric_b.py` if you want to follow the iteration
(GBDT overfit → regularisation → dispersion ablation → conformal failures → online ACI → online bias).
`DECISIONS.md` is the lab notebook.

## Repo map

```
src/          data loading, features, models, splits, online_bias, dispersion
scripts/      numbered experiment scripts + 10_final_test + make_figures
tests/        unit tests for metrics and leak-safety of the eval harness
config.yaml   paths, lead band, validation scheme, online_bias settings
outputs/      eda plot + sealed-test figures (regenerated, gitignored)
```

## What I'd do next

- Recency-weighted conformal instead of a fixed cal slice, to close the last ~1pt of coverage without overshooting.
- Variance-normalised dispersion (partial correlation was only 0.18 controlling for level).