# Repair-cost ratio plausibility gate - 2026-06-22

## Protocol

DesignSafe PRJ-3126 repair cost/time data were downloaded and inspected. The master table has repair-cost and repair-time P50 values for component damage states, but no CPT/event/geospatial key that links directly to the Canterbury manifestation table.

## Result

Status: `PASS_REPAIR_COST_RATIO_PLAUSIBILITY_GATE`.
Direct Canterbury consequence validation: `blocked` because no shared key exists.

## Claim boundary

Allowed: external New Zealand repair-cost/time data support the plausibility of using false-negative cost ratios greater than one in screening frontiers.

Blocked: do not treat PRJ-3126 as direct Canterbury CPT consequence validation, repair-cost prediction, or design-level validation.
