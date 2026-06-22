# Subagent MRNB-100 Georisk recheck - 2026-06-07

## Scope

Applied the academic article skills, subagent orchestration, Q1 evidence gate and MRNB-100-Q1-EA benchmark to Article 118 after adding the Canterbury multi-site gate and the cost-sensitive decision frontier.

Target journal: Georisk: Assessment and Management of Risk for Engineered Systems and Geohazards.

## Subagent findings

| Role | Main finding | Action taken |
|---|---|---|
| Red-Team Reviewer | Novelty improved but remains conditional; avoid selling M2/M3 as universal and make the cost frontier central. | Reframed manuscript as a cost-sensitive operating-policy diagnostic. |
| Technical/Mathematical Validator | Retuned thresholds are diagnostic/oracle, not operational validation; `0.189 vs 0.465` must be labelled as domain-balanced. | Updated `run_122`, README and DOCX to report diagnostic/oracle frontier, fixed-threshold frontier, domain-balanced and case-weighted losses. |
| Literature and Journal Fit Auditor | Fit to Georisk is strong but capped without direct Georisk literature. | Added Georisk citations [51-54] and positioned the gap against risk-informed sampling, spatial variability and geotechnical risk management. |
| Manuscript Integrator | Main DOCX is healthy; supplement and package cleanliness needed correction. | Updated `Supplementary_Material.docx`, moved backup DOCX files out of `manuscript`, removed Python caches. |

## Current MRNB-100-Q1-EA estimate

| Criterion | Score |
|---|---:|
| Specific editorial fit with Georisk | 8.5 / 10 |
| Competitive editorial novelty | 11.0 / 15 |
| Demonstrated scientific gap | 8.5 / 10 |
| Theoretical/mathematical solidity | 8.0 / 10 |
| Methodological rigor | 10.0 / 12 |
| Independent validation/comparison | 12.0 / 15 |
| Reproducibility and auditability | 9.5 / 10 |
| Evidence-conclusion alignment | 7.5 / 8 |
| Critical discussion and limitations | 4.5 / 5 |
| Ethics, format and reporting | 4.5 / 5 |
| Total | 84.0 / 100 |

## Editorial control sentence

The manuscript deserves review in Georisk because it demonstrates that non-stationary liquefaction updating should be evaluated as a cost-sensitive georisk operating-policy diagnostic rather than as a universally superior probability model, and it validates that distinction using Hu et al. case-history discrimination, Nisqually spatial holdouts, the SODO/UW adverse holdout, a 15,890-state Canterbury external transfer and an auditable false-negative-cost frontier.

## Remaining risks

- The result is a decision-policy diagnostic, not probability recalibration.
- The threshold-retuned frontier is an oracle diagnostic because thresholds are selected using validation labels.
- The strongest fixed-threshold benefit depends on false-negative costs exceeding false-positive costs.
- The manuscript is stronger for Georisk after adding Georisk literature, but it is not a 100/100 blind-prediction paper.

## Current status

Finished as a strengthened Georisk-targeted technical manuscript and reproducible package. Not declared 100/100 or upload-ready until the user decides whether to prepare a clean submission ZIP/cover letter and whether to commit or otherwise freeze the new untracked evidence files.
