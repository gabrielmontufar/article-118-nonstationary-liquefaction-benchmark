# Cost-sensitive novelty frontier - 2026-06-07

## Action

Added a cost-sensitive georisk decision frontier to strengthen the novelty of Article 118 without claiming universal M2/M3 probabilistic superiority.

## New reproducible assets

- `scripts/run_122_cost_sensitive_decision_frontier.py`
- `outputs/cost_sensitive_decision_frontier.csv`
- `outputs/cost_sensitive_decision_frontier_fixed_threshold.csv`
- `outputs/cost_sensitive_decision_frontier_best_by_domain.csv`
- `outputs/cost_sensitive_decision_frontier_balanced.csv`
- `outputs/cost_sensitive_decision_frontier_summary.json`
- `figures/fig11_cost_sensitive_guardrail_frontier.png`

## Result

- With domain-specific cost-optimized thresholds, M0 remains the lowest-loss baseline across the tested false-negative/false-positive cost ratios. This is now labelled as a diagnostic/oracle frontier because the thresholds are selected from validation labels.
- With the standard fixed 0.50 screening threshold, conservative max(M0..M3) becomes the lowest-loss rule once a false negative is at least twice as costly as a false positive.
- At a 10:1 false-negative/false-positive cost ratio, mean expected loss is:
  - domain-balanced M0 static stationary: 0.465
  - domain-balanced conservative max(M0..M3): 0.189
  - case-weighted M0 static stationary: 0.521
  - case-weighted conservative max(M0..M3): 0.484

## Novelty claim enabled

The manuscript can now claim a reproducible operating-policy diagnostic for georisk screening: calibration-favoured M0 should be retained in the diagnostic/oracle retuned-threshold view, while the conservative max(M0..M3) guardrail becomes defensible for fixed-threshold missed-event-averse screening.

## Claim still blocked

Do not claim probability recalibration, universal M2/M3 superiority, or universal validation. The contribution is a decision frontier under explicit false-negative cost, not a new universally better liquefaction predictor.

## Manuscript update

The abstract, contribution paragraph, Canterbury/guardrail results, discussion, conclusions, data availability text and closure notes were updated. Fig. 11 was inserted in both manuscript copies.
