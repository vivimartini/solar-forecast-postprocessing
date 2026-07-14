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
−15% to **+1.28%**. Small, but real, and it holds up out of sample, which the fancier model didn't.
Putting the static GBDT back on top of the de-biased residual still hurt, so I left it out.

The uncertainty side showed the same pattern: a fixed (offline) conformal calibration failed under the
drift, while an online one that self-corrects as it moves through the test period gets the P10–P90 band
to ~80% coverage on data it never saw.

## Results (sealed test)

| | |
|---|---|
| point skill vs raw forecast | +1.28% |
| P10–P90 coverage | 79.1% (target 80%) |
| coverage by regime (calm / mid / volatile) | 0.82 / 0.79 / 0.77 |
| disagreement vs next-revision size (Spearman) | 0.44 |
| forecast error, calm → volatile hours | 801 → 1412 → 1725 MW |

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
honest one: I'm correcting a forecast that's *already* physics-based, so the diurnal arc and