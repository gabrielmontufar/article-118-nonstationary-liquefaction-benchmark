# Canterbury multi-site evidence gate - 2026-06-07

## Action

Implemented the Canterbury PRJ-2937 multi-site external transfer gate for Article 118.

## New reproducible assets

- `scripts/run_121_canterbury_multi_site_temporal_validation.py`
- `data/canterbury_multi_site_cpt_summary.csv`
- `data/canterbury_multi_site_event_features.csv`
- `outputs/canterbury_multi_site_temporal_prediction_probabilities.csv`
- `outputs/canterbury_multi_site_temporal_validation_metrics.csv`
- `outputs/canterbury_multi_site_temporal_validation_summary.json`
- `outputs/canterbury_multi_site_temporal_policy_cases.csv`
- `outputs/conservative_max_guardrail_canterbury_multi_site_predictions.csv`
- `figures/fig10_canterbury_multi_site_validation.png`

## Results

- Raw source: `raw_designsafe/canterbury_designsafe_PRJ-2937/CANTERBURYDATASET.mat`.
- Records in MAT file: 5,668.
- Usable CPT records: 5,668.
- Scored event states: 15,890.
- Excluded indeterminate manifestation-code-10 states: 1,114.
- Label rule: manifestation codes 1-5 = liquefaction, 0 = no liquefaction, 10 = excluded.

| Model | Brier | AUC | FN | FP |
|---|---:|---:|---:|---:|
| M0 static stationary | 0.318175 | 0.864921 | 224 | 6034 |
| M1 groundwater | 0.319886 | 0.865931 | 203 | 6049 |
| M2 groundwater + gradation | 0.407847 | 0.813245 | 149 | 6995 |
| M3 random-field extension | 0.406719 | 0.826248 | 111 | 6972 |
| conservative max(M0..M3) | 0.428961 | 0.817844 | 43 | 7278 |

## Gate decision

- The old Canterbury `n=3` limitation is overcome.
- Universal M2/M3 Brier superiority is not overcome.
- The defensible Georisk claim is a false-negative-aware screening/guardrail claim with disclosed false-positive and calibration cost.
- The manuscript and README were updated to state this boundary explicitly.

## Verification

- `python -m py_compile` passed for modified/new scripts.
- Full reproducibility chain passed:
  - `run_118_nonstationary_liquefaction_benchmark.py`
  - `run_119_site_calibrated_application.py`
  - `run_120_canterbury_temporal_site_application.py`
  - `run_121_canterbury_multi_site_temporal_validation.py`
  - `explore_nisqually_out_of_sample_protocols.py`
  - `analyze_sodo_uw_adverse_holdout.py`
  - `select_adversarial_guardrail_model.py`
  - `adversarial_threshold_operating_gate.py`
  - `paired_uncertainty_model_selection_gate.py`
- Word COM verification after DOCX update:
  - words: 14,382
  - paragraphs: 1,020
  - tables: 15
  - inline shapes: 10
  - OMath objects: 3
