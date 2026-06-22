# Hydraulic-leverage physical-regime gate - 2026-06-22

## Mechanism

For fixed PGA and depth, CSR is proportional to `sigma_v / sigma'_v`. A shallower water table lowers effective vertical stress and therefore amplifies CSR. The gate tests whether this hydraulic leverage becomes consequential when the CPT profile is also loose and fine-sensitive.

## Result

Status: `FAIL_FIELD_HYDRAULIC_LEVERAGE_NOT_VALIDATED`.
Gate definition: `top quartile hydraulic leverage AND bottom quartile qc10_mpa AND top quartile Ic_median`.

## Claim boundary

Allowed: The field test can be reported as a falsification/guardrail: a simple hydraulic-leverage plus CPT-fabric gate is not sufficient, by itself, to explain Canterbury false negatives.

Blocked: Do not call this a new constitutive law, universal liquefaction triggering equation, or repair-cost validation. It is a field-tested physical regime/gate derived from effective stress and CPT fabric proxies.
