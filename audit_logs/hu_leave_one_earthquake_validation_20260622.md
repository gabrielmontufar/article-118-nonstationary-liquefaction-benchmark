# Hu leave-one-earthquake validation - 2026-06-22

## Protocol

Each earthquake in the Hu et al. external case-history table is held out in turn. Screening scores are calibrated on all remaining earthquakes and then predicted on the held-out event.

## Result

Status: `PASS_HU_LEAVE_ONE_EARTHQUAKE_VALIDATION`.
Best Brier model: `combined_instrument_score`.
Best Brier: `0.194009`.
Best AUC: `0.753351`.
Predictions in best model: `232` across `15` held-out earthquakes.

## Claim boundary

Allowed: event-level external validation support for bounded screening scores across held-out earthquakes.

Blocked: do not describe this as site-specific design validation or validation of the non-stationary time-update operator.
