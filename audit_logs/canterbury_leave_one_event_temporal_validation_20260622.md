# Canterbury leave-one-event temporal validation - 2026-06-22

## Protocol

Each Canterbury earthquake is held out as a complete temporal validation event. Models are trained on the other two earthquakes and evaluated on the held-out event.

## Pooled result

Best pooled Brier model: `M2_nonstationary_groundwater_gradation` with Brier `0.137673`.
Best pooled AUC model: `M2_nonstationary_groundwater_gradation` with AUC `0.833519`.

## Claim boundary

Allowed: formal leave-one-earthquake-out temporal validation within Canterbury: model fitting excludes the held-out event, and groundwater/gradation-aware predictors improve rank discrimination and false-negative operating choices in selected held-out events.

Blocked: do not present this as universal non-stationary superiority; M0 remains competitive on Brier score, especially for the low-prevalence 2016 event.
