# Supplementary benchmark for Article 118

This folder contains the reproducible benchmark for the manuscript:

**A reproducible reliability benchmark for non-stationary liquefaction probability under groundwater and gradation-change scenarios**

Public repository: <https://github.com/gabrielmontufar/article-118-nonstationary-liquefaction-benchmark>

Versioned release: <https://github.com/gabrielmontufar/article-118-nonstationary-liquefaction-benchmark/releases/tag/v1.3>

## Purpose

The benchmark evaluates how time-dependent groundwater depth and gradation changes alter the probability of liquefaction in a layered soil profile. It compares:

- deterministic factor-of-safety checks;
- stationary probabilistic assessment;
- non-stationary probabilistic assessment.

## Files

- `scripts/run_118_nonstationary_liquefaction_benchmark.py`: self-contained Python script.
- `scripts/run_119_site_calibrated_application.py`: site-calibrated Nisqually extension using DesignSafe PRJ-3758 CPT case histories.
- `src/`: modular support code for groundwater calibration, stress calculations, vertical random-field diagnostics and validation metrics.
- `data/synthetic_layer_profile.csv`: synthetic layered profile.
- `data/liquefaction_benchmark_results.csv`: layer-time-scenario Monte Carlo results.
- `data/liquefaction_benchmark_summary.csv`: scenario-level summary.
- `data/profile_method_comparison.csv`: stationary versus non-stationary profile comparison.
- `data/standard_model_comparison.csv`: layer-time comparison against the benchmark resistance module, BI14/NCEER-style SPT, and BI14 SPT computed through `liquepy` when available.
- `data/standard_model_comparison_summary.csv`: scenario-level rank-agreement, Pf-error, and FS-scale diagnostics for the standard SPT model-form check.
- `data/model_availability_diagnostics.csv`: implementation-status table for BI14, BI16, Cetin, Moss/CPT, and Kayen/Vs alternatives using the variables available in this supplement.
- `data/vertical_dependence_sensitivity.csv`: system-probability diagnostic under equicorrelated Gaussian-copula layer dependence.
- `data/monte_carlo_convergence_check.csv`: repeated-sample convergence check for the most severe benchmark state.
- `data/global_sensitivity_rank.csv`: rank-correlation sensitivity check for the non-stationary liquefaction probability.
- `data/external_trend_consistency_checks.csv`: traceability table linking benchmark assumptions to external literature trends.
- `data/external_hu_2021_gravelly_liquefaction_cases.xlsx`: open supplementary dataset from Hu et al. (2021), Data in Brief, used only for a secondary external sanity check.
- `data/external_case_history_sanity_check.csv`: parsed case-history variables and screening scores for the external sanity check.
- `data/external_case_history_sanity_summary.csv`: AUC and mean/median checks comparing liquefied and non-liquefied historical cases, including demand-only, groundwater-only, DPT, Vs, and combined-instrument screening scores.
- `data/external_case_history_calibration_metrics.csv`: five-fold Brier score and confusion-matrix diagnostics for the external screening scores.
- `data/external_case_history_calibration_bins.csv`: five-bin probability calibration table for the external screening scores.
- `data/external_static_limit_state_proxy.csv`: static proxy application of the benchmark limit-state variables to Hu et al. cases where CSR, resistance and fines data are available.
- `data/external_static_limit_state_metrics.csv`: AUC, five-fold Brier score and classification metrics for the static proxy check.
- `data/field_validation_cases.csv`: field case-history validation table from Hu et al. (2021), including the Wenchuan subset, groundwater depth, PGA/CSR, N120/Vs, fines/gravel content, and observed liquefied/non-liquefied labels.
- `data/field_validation_metrics.csv`: Hu full-set and Wenchuan-subset AUC, five-fold Brier score, confusion matrix, sensitivity, specificity, and threshold diagnostics.
- `data/field_validation_thresholds.csv`: prespecified and apparent Youden thresholds for the field validation scores.
- `data/site_profile_calibrated.csv`: measured Nisqually CPT profile summaries by site/layer.
- `data/site_groundwater_timeseries.csv`: reported Nisqually water-table depths with dates.
- `data/site_gradation_timeseries.csv`: gradation-status table; FC/D50 are not reported in PRJ-3758 and are therefore not treated as calibrated time-varying quantities.
- `data/site_event_observations.csv`: Nisqually event, PGA and manifestation observations.
- `data/site_groundwater_model_parameters.csv`: calibrated groundwater trajectory coefficients.
- `data/site_gradation_model_parameters.csv`: gradation model status by layer.
- `data/site_vertical_variogram_parameters.csv`: CPT-derived vertical variogram diagnostics.
- `outputs/site_validation_metrics.csv`: out-of-sample M0-M3 validation metrics using a held-out SODO+UW family split.
- `outputs/site_model_form_comparison.csv`: site-calibrated model-form probability comparison.
- `outputs/site_random_field_system_probability.csv`: random-field system-probability diagnostic for the site extension.
- `figures/fig01_pf_time_extreme_accumulation.png`: layer probability histories.
- `figures/fig02_profile_mean_pf_by_scenario.png`: profile-average probability by groundwater scenario.
- `figures/fig03_depth_time_pf_heatmap.png`: depth-time probability map.
- `figures/fig04_global_sensitivity_rank.png`: global rank-correlation sensitivity chart.
- `figures/fig05_external_case_history_sanity_auc.png`: AUC chart for the external case-history sanity check.
- `figures/fig06_site_groundwater_calibration.png`: observed Nisqually WTD points, calibrated mean trajectory, 95% interval and event date.
- `figures/fig07_model_form_comparison_site.png`: M0-M3 out-of-sample Brier comparison for the site application.

SVG versions of the figures are also included for editable vector use.

## Reproducibility

Run from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\scripts\run_118_nonstationary_liquefaction_benchmark.py
python .\scripts\run_119_site_calibrated_application.py
```

Python dependencies:

- numpy
- pandas
- Pillow
- openpyxl
- liquepy

The random seed is fixed as `1182026`.

The main results file includes approximate 95% binomial confidence intervals for both stationary and non-stationary liquefaction probabilities. The convergence file repeats the most severe state with 1,000, 3,000, 6,000, and 12,000 samples to document Monte Carlo stability.

## Scope and interpretation note

This benchmark is synthetic. The reported `Pf(t)` values are conditional probabilities under the specified stochastic assumptions, groundwater scenarios, gradation-change scenarios, triggering equations, and Monte Carlo input distributions. They should not be interpreted as site-validated annual failure probabilities or as calibrated predictions for a real deposit. The external case-history exercise is a directional sanity check only; it evaluates whether simple screening indices rank liquefied cases above non-liquefied cases, but it does not validate or calibrate the synthetic benchmark.

## Main output columns

- `pf_stationary`: stationary probability computed for the initial groundwater and gradation state, reused across years for the same scenario, gradation and layer.
- `pf_nonstationary`: scenario-conditioned probability computed for the groundwater and gradation state at the reported year.
- `delta_pf`: difference between `pf_nonstationary` and `pf_stationary`.
- `pf_*_ci_low` and `pf_*_ci_high`: approximate 95% binomial confidence interval bounds.
- `beta_proxy`: diagnostic index based on the sampled safety-margin distribution; it is not a formal reliability index.
- `beta_equivalent`: equivalent normal reliability index computed as `-Phi^-1(Pf)` from the non-stationary layer probability.
- `fs_deterministic_nonstationary`: deterministic factor of safety for the non-stationary state.

The profile comparison file also reports `max_layer_pf`, `psys_independent_layers`, `frechet_lower`, and `frechet_upper`. These are diagnostic system-level summaries derived from the layer probabilities. `psys_independent_layers` assumes independent layer events; the Frechet bounds show the possible system-failure probability range under arbitrary dependence. The manuscript keeps the profile-average value as a screening index, not as a calibrated system-failure probability.

The vertical-dependence sensitivity file extends this point using an equicorrelated Gaussian copula for layer events with rho values of 0.0, 0.3, 0.6, and 0.9. It is a diagnostic for dependence sensitivity, not a calibrated random-field model.

The standard model comparison files are model-form checks. They replace the benchmark resistance module with a BI14/NCEER-style SPT clean-sand CRR curve and, when `liquepy` is installed, the matching `liquepy.trigger.boulanger_and_idriss_2014.calc_crr_m7p5_from_n1_60cs` implementation. The comparison keeps the same synthetic layer states, groundwater depths, effective stresses and demand terms so the reported Spearman rank agreement, Pf error, and FS scale ratios isolate resistance-module effects. BI16, Cetin, Moss/CPT and Kayen/Vs are documented in `data/model_availability_diagnostics.csv`; they are not silently substituted when exact inputs or callables are absent.

## External sanity check

The benchmark includes a secondary external check using the open case-history dataset of Hu et al. (2021), Data in Brief, DOI: `10.1016/j.dib.2021.107104`. This check does not calibrate the synthetic benchmark to a field site. It tests whether simple demand/resistance and groundwater screening indices rank historical liquefied cases above non-liquefied cases in the expected direction.

The external check also reports five-fold calibrated Brier scores, confusion matrices and a static limit-state proxy for the subset of Hu et al. cases with CSR, resistance and fines-content data. These diagnostics are compatibility checks only. They do not validate the non-stationary benchmark or replace calibration against a site-specific field dataset.

The Hu et al. and Wenchuan tables are external case-history compatibility checks. They evaluate whether demand, resistance and groundwater indicators rank observed liquefied cases above non-liquefied cases. They do not constitute a calibrated non-stationary field application.

## Site-calibrated extension

The Nisqually extension uses the DesignSafe PRJ-3758 CPT case-history dataset. It separates the reproducible synthetic benchmark from a documented-site application. The available data include CPT depth profiles, coordinates, CPT dates, reported water-table depth, conditional PGA and liquefaction manifestation. FC, D50 and laboratory gradation histories are not reported in the downloaded PRJ-3758 XLSX files; gradation is therefore treated as baseline uncertainty/proxy information through CPT-derived Ic, and the synthetic fines-accumulation/fines-washout paths remain scenario sensitivities rather than validated site predictions.

The validation protocol is intentionally out-of-sample: M0-M3 are trained on 16 non-SODO/UW case histories and evaluated on the held-out SODO+UW site families. In the current strict split, M2 and M3 improve model richness but do not improve Brier score relative to the simple M0 baseline. This negative/mixed result is retained because it is a scientifically useful stress test: the site extension demonstrates data insertion, groundwater calibration and random-field diagnostics, but it does not claim that non-stationarity improves predictive calibration for this small Nisqually holdout.

## Methodological note

The profile is synthetic and transparent. It is intended to test the proposed non-stationary reliability workflow, not to claim calibration to a specific field site. Quantitative site calibration requires field or laboratory data and should be treated as future work unless a site dataset is added.

The file `data/global_sensitivity_rank.csv` reports rank-correlation sensitivity screening using Pearson and Spearman correlations. It should not be interpreted as a Sobol or variance-decomposition sensitivity analysis.
