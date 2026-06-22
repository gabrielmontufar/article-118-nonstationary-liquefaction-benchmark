# Supplementary benchmark for Article 118

This folder contains the reproducible georisk reliability-updating benchmark for the manuscript:

**A reproducible reliability benchmark for non-stationary liquefaction probability under groundwater and gradation-change scenarios**

Public repository: <https://github.com/gabrielmontufar/article-118-nonstationary-liquefaction-benchmark>

Current physics-forward manuscript title:

**Critical Water-Table Crossing as a Physical Switch for False-Negative-Aware Liquefaction Screening**

The main physical result is now isolated by `scripts/run_131_critical_water_table_crossing_mechanism.py`: when a non-stationary water table crosses a shallow liquefiable layer, effective stress drops, CSR rises, FS falls and Pf can jump rather than drift. In the extreme benchmark scenario, the L1 crossing from 2.21 m to 1.25 m raises CSR by 25.3%, lowers deterministic FS by 19.7% and increases Pf by 0.149. This is a bounded physical mechanism claim, not a new constitutive law or universal field classifier.

The package also contains a CIVIL-NOVELTY-Q1-100 manuscript audit. The current evidence-bound CIVIL novelty score is 81/100 after adding a condensed close-literature frontier, explicit evidence-class separation, a falsifiability paragraph, and a control sentence stating what the community learns after this manuscript. The score is capped below 85 until the physical switch is directly confirmed by monitored temporal field evolution or property/site-linked consequence validation.

Versioned MRNB-100 Georisk tag: <https://github.com/gabrielmontufar/article-118-nonstationary-liquefaction-benchmark/releases/tag/v2.0-mrnb100-georisk>

Frozen commit: `2981604`

## Purpose

The benchmark evaluates how time-dependent groundwater depth and gradation changes alter the probability of liquefaction in a layered soil profile. It compares:

- deterministic factor-of-safety checks;
- stationary probabilistic assessment;
- non-stationary probabilistic assessment.

## Files

- `scripts/run_118_nonstationary_liquefaction_benchmark.py`: self-contained Python script.
- `scripts/run_131_critical_water_table_crossing_mechanism.py`: extracts the critical water-table crossing mechanism and writes Fig. 15 plus crossing/jump summaries.
- `scripts/run_132_aggregate_eqc_cost_ratio_validation.py`: checks the decision-cost ratios against public Canterbury/EQC residential repair-cost evidence. Published regional loss ratios of 0.013, 0.066 and 0.171 for MMI 6, 7 and 8 imply empirical multipliers of about 5.1x (MMI7/MMI6) and 13.2x (MMI8/MMI6), supporting FN:FP ratios of 5-10 as economically plausible.
- `scripts/run_133_public_design_consequence_proxy_validation.py`: joins Article 118 CPT predictions to public Christchurch/Canterbury hazard, vulnerability, foundation-design and technical-category layers. The result is a claim-boundary check: the strongest public proxy signal is weakly positive for CCC moderate-to-severe liquefaction hazard, while technical-category proxies do not provide strong discrimination. It is not monetary cost validation or design certification.
- `scripts/run_130_hydraulic_leverage_breakthrough_gate.py`: field-regime guardrail; the simple hydraulic-leverage+CPT-fabric gate is not sufficient by itself to explain Canterbury false negatives.
- `scripts/run_129_official_spatial_consequence_validation.py`: reproducible spatial join to public official Canterbury liquefaction occurrence/vulnerability layers; occurrence validation is direct spatial evidence, while the queried ECan vulnerability layer is not informative for design discrimination in this CPT subset.
- `scripts/run_119_site_calibrated_application.py`: site-calibrated Nisqually extension using DesignSafe PRJ-3758 CPT case histories.
- `scripts/run_120_canterbury_temporal_site_application.py`: Canterbury PRJ-2937 temporal site application for 100 Osbourne St CPT01.
- `scripts/run_121_canterbury_multi_site_temporal_validation.py`: Canterbury PRJ-2937 external transfer validation across all usable CPT event states in `CANTERBURYDATASET.mat`.
- `scripts/run_122_cost_sensitive_decision_frontier.py`: cost-sensitive georisk decision frontier for false-negative-weighted screening decisions.
- `scripts/run_123_hu_leave_one_earthquake_validation.py`: leave-one-earthquake-out external case-history validation on the Hu et al. dataset.
- `scripts/run_124_locked_threshold_external_decision_validation.py`: locked-threshold external decision validation; thresholds are selected on Nisqually only and transferred unchanged to SODO/UW and Canterbury.
- `scripts/run_125_canterbury_leave_one_event_temporal_validation.py`: formal Canterbury leave-one-earthquake-out temporal validation; each event is held out as a complete validation event.
- `scripts/run_126_reader_oriented_figures.py`: reader-oriented figure generator for the compact Feynman-style main manuscript.
- `scripts/run_127_canterbury_severity_triage_validation.py`: action-style Canterbury severity-triage validation using observed manifestation-code severity.
- `scripts/run_128_repair_cost_ratio_plausibility_gate.py`: DesignSafe PRJ-3126 repair-cost/time ratio plausibility gate for the false-negative cost frontier.
- `scripts/select_adversarial_guardrail_model.py`: prespecified adversarial model-selection and conservative max(M0..M3) screening guardrail for the site-validation protocols.
- `src/`: modular support code for groundwater calibration, gradation-status handling, stress calculations, triggering-model registry, vertical random-field diagnostics and validation metrics.
- `manuscript/A_reproducible_reliability_benchmark.docx`: final corrected manuscript file used as the source for the repository PNG figures.
- `manuscript/Supplementary_Material.docx`: supplementary material Word file containing Tables S1-S4 and expanded explanations.
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
- `data/external_case_history_compatibility_cases.csv`: external Hu et al. (2021) case-history compatibility table, including the Wenchuan subset, groundwater depth, PGA/CSR, N120/Vs, fines/gravel content, and observed liquefied/non-liquefied labels.
- `data/external_case_history_compatibility_metrics.csv`: Hu full-set and Wenchuan-subset AUC, five-fold Brier score, confusion matrix, sensitivity, specificity, and threshold diagnostics. These checks are not the calibrated site validation.
- `data/external_case_history_compatibility_thresholds.csv`: prespecified and apparent Youden thresholds for the external compatibility scores.
- `data/site_profile_calibrated.csv`: measured Nisqually CPT profile summaries by site/layer.
- `data/site_groundwater_timeseries.csv`: reported Nisqually water-table depths with dates.
- `data/site_gradation_timeseries.csv`: gradation-status table; FC/D50 are not reported in PRJ-3758 and are therefore not treated as calibrated time-varying quantities.
- `data/site_event_observations.csv`: Nisqually event, PGA and manifestation observations.
- `data/site_groundwater_model_parameters.csv`: calibrated groundwater trajectory coefficients.
- `data/site_gradation_model_parameters.csv`: gradation model status by layer.
- `data/site_vertical_variogram_parameters.csv`: CPT-derived vertical variogram diagnostics.
- `data/triggering_model_uncertainty_treatment.csv`: triggering-model registry, activation status and uncertainty treatment.
- `data/canterbury_100_osbourne_cpt01_profile_points.csv`: point CPT profile for the temporal site application.
- `data/canterbury_100_osbourne_site_profile_calibrated.csv`: layer summaries for the Canterbury CPT profile.
- `data/canterbury_100_osbourne_groundwater_timeseries.csv`: event-specific groundwater depths for 2010, 2011 and 2016.
- `data/canterbury_100_osbourne_event_observations.csv`: event-specific Mw, PGA, groundwater and manifestation observations.
- `data/canterbury_multi_site_cpt_summary.csv`: CPT-level summary for all usable Canterbury PRJ-2937 records.
- `data/canterbury_multi_site_event_features.csv`: scored Canterbury event-state feature table; manifestation code 10 is excluded as indeterminate.
- `outputs/site_validation_metrics.csv`: pooled out-of-sample M0-M3 validation metrics from leave-one-site-family-out spatial validation.
- `outputs/site_validation_leave_one_family_metrics.csv`: fold-level leave-one-family-out metrics.
- `outputs/site_validation_paired_family_metrics.csv`: pooled sensitivity metrics from exhaustive paired-family spatial holdouts.
- `outputs/site_validation_paired_family_fold_metrics.csv`: fold-level paired-family holdout metrics.
- `outputs/site_validation_sodo_uw_sensitivity.csv`: strict SODO+UW holdout sensitivity check.
- `outputs/adversarial_model_selection_guardrail_policy.csv`: domain-specific model-selection policy that keeps M0 as the conservative SODO/UW guardrail while allowing M2/M3 only where they win the declared Brier protocol.
- `outputs/adversarial_model_selection_guardrail_summary.json`: machine-readable summary of the bounded claim enabled by the adversarial guardrail and the universal claim it blocks.
- `outputs/conservative_max_guardrail_metrics.csv`: metrics for the separate conservative max(M0..M3) screening rule across existing validation protocols.
- `outputs/conservative_max_guardrail_site_predictions.csv`, `outputs/conservative_max_guardrail_protocol_predictions.csv`, and `outputs/conservative_max_guardrail_canterbury_predictions.csv`: site/event-level conservative screening predictions used to compute the guardrail metrics.
- `outputs/paired_uncertainty_model_selection_metrics.csv`: paired Brier-delta and bootstrap uncertainty audit for M0-M3 and conservative max(M0..M3) across pooled Nisqually, SODO/UW and Canterbury domains.
- `outputs/paired_uncertainty_model_selection_summary.json`: machine-readable uncertainty-bounded claim boundary for model-selection evidence.
- `outputs/site_model_form_comparison.csv`: site-calibrated model-form probability comparison.
- `outputs/site_random_field_system_probability.csv`: random-field system-probability diagnostic for the site extension.
- `outputs/site_triggering_stress_profile.csv`: layer-wise total stress, pore pressure, effective stress, depth-only rd and CSR seed terms for the site application.
- `outputs/canterbury_temporal_prediction_probabilities.csv`: M0-M3 probability predictions for the Canterbury event sequence.
- `outputs/canterbury_temporal_validation_metrics.csv`: Brier, log-loss, AUC, calibration, sensitivity and specificity for the Canterbury temporal transfer.
- `outputs/canterbury_multi_site_temporal_prediction_probabilities.csv`: M0-M3 probability predictions for the expanded Canterbury transfer domain.
- `outputs/canterbury_multi_site_temporal_validation_metrics.csv`: Brier, log-loss, AUC, sensitivity, specificity and confusion counts for the expanded Canterbury transfer.
- `outputs/canterbury_multi_site_temporal_validation_summary.json`: machine-readable gate showing that the n=3 limitation is removed but universal M2/M3 Brier superiority remains blocked.
- `outputs/conservative_max_guardrail_canterbury_multi_site_predictions.csv`: conservative max(M0..M3) screening predictions for the expanded Canterbury transfer.
- `outputs/cost_sensitive_decision_frontier.csv`: optimized-threshold expected screening loss for M0-M3 and conservative max(M0..M3) across false-negative/false-positive cost ratios.
- `outputs/cost_sensitive_decision_frontier_fixed_threshold.csv`: fixed-threshold 0.50 expected screening loss across the same cost ratios.
- `outputs/cost_sensitive_decision_frontier_case_weighted_fixed_threshold.csv`: case-weighted fixed-threshold 0.50 decision frontier, reported separately from the domain-balanced view.
- `outputs/cost_sensitive_decision_frontier_summary.json`: machine-readable summary of the operating-policy split.
- `outputs/hu_leave_one_earthquake_predictions.csv`: held-out-earthquake predictions for the Hu et al. external case-history validation.
- `outputs/hu_leave_one_earthquake_metrics.csv`: pooled leave-one-earthquake-out metrics by screening model.
- `outputs/hu_leave_one_earthquake_fold_metrics.csv`: fold-level metrics for each held-out earthquake.
- `outputs/hu_leave_one_earthquake_summary.json`: machine-readable summary of the event-level external validation claim boundary.
- `outputs/locked_threshold_selection_rules.csv`: Nisqually-selected thresholds for the locked external decision-rule validation.
- `outputs/locked_threshold_external_decision_validation.csv`: external-domain losses, false negatives and false positives after applying the locked thresholds.
- `outputs/locked_threshold_external_case_weighted_summary.csv`: case-weighted external summary, including coverage flags for partial/full external-domain comparisons.
- `outputs/locked_threshold_external_decision_summary.json`: machine-readable summary of the locked decision-rule validation and claim boundary.
- `outputs/canterbury_leave_one_event_temporal_predictions.csv`: Canterbury predictions from models trained on two earthquakes and evaluated on the held-out earthquake.
- `outputs/canterbury_leave_one_event_temporal_fold_metrics.csv`: event-wise leave-one-earthquake-out metrics.
- `outputs/canterbury_leave_one_event_temporal_pooled_metrics.csv`: pooled temporal holdout metrics across the three held-out Canterbury events.
- `outputs/canterbury_leave_one_event_temporal_locked_thresholds.csv`: training-event-selected thresholds applied to each held-out Canterbury event.
- `outputs/canterbury_leave_one_event_temporal_summary.json`: machine-readable summary of the formal temporal holdout and claim boundary.
- `figures/fig12_canterbury_leave_one_event_temporal_validation.png`: reader-facing temporal holdout summary.
- `figures/fig13_locked_threshold_fn_fp_tradeoff.png`: reader-facing false-negative/false-positive trade-off graphic.
- `figures/fig14_evidence_claim_boundary.png`: reader-facing evidence and claim-boundary map.
- `outputs/canterbury_severity_triage_validation.csv`: top-10% and top-20% severe-manifestation capture metrics by held-out event and model.
- `outputs/canterbury_severity_triage_best_by_event.csv`: best action-style triage model by event and action fraction.
- `outputs/canterbury_severity_triage_summary.json`: machine-readable summary of the severity-triage claim boundary.
- `outputs/repair_cost_ratio_plausibility_components.csv`: component-level DS3/DS4 versus DS1 repair-cost/time ratios from DesignSafe PRJ-3126.
- `outputs/repair_cost_ratio_plausibility_summary.csv`: aggregate repair-cost/time ratio statistics.
- `outputs/repair_cost_ratio_plausibility_summary.json`: machine-readable claim-boundary summary for cost-ratio plausibility.
- `figures/fig01_pf_time_extreme_accumulation.png`: final manuscript Fig. 1, layer probability histories.
- `figures/fig02_profile_mean_pf_by_scenario.png`: final manuscript Fig. 2, profile-average probability by groundwater scenario.
- `figures/fig03_depth_time_pf_heatmap.png`: final manuscript Fig. 3, depth-time probability map.
- `figures/fig04_global_sensitivity_rank.png`: final manuscript Fig. 4, global rank-correlation sensitivity chart.
- `figures/fig05_external_case_history_sanity_auc.png`: final manuscript Fig. 5, AUC chart for the external case-history sanity check.
- `figures/fig06_site_groundwater_calibration.png`: final manuscript Fig. 6, observed Nisqually WTD points, calibrated mean trajectory, 95% interval and event date.
- `figures/fig07_model_form_comparison_site.png`: final manuscript Fig. 7, M0-M3 out-of-sample Brier comparison for the site application.
- `figures/fig08_vertical_variogram_fit.png`: final manuscript Fig. 8, empirical and fitted vertical variogram for the CPT log(qc) diagnostic.
- `figures/fig09_canterbury_temporal_pf.png`: final manuscript Fig. 9, Canterbury 100 Osbourne Pf(t) over the three documented events.
- `figures/fig10_canterbury_multi_site_validation.png`: expanded Canterbury external-transfer Brier and false-negative summary.
- `figures/fig11_cost_sensitive_guardrail_frontier.png`: cost-sensitive georisk decision frontier.

The PNG figures in `figures/` were extracted from the final manuscript DOCX so that the repository figure set matches the submitted document.

## Reproducibility

Run from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\scripts\download_designsafe_raw_118.py
python .\scripts\run_118_nonstationary_liquefaction_benchmark.py
python .\scripts\run_119_site_calibrated_application.py
python .\scripts\run_120_canterbury_temporal_site_application.py
python .\scripts\run_121_canterbury_multi_site_temporal_validation.py
python .\scripts\run_122_cost_sensitive_decision_frontier.py
python .\scripts\run_123_hu_leave_one_earthquake_validation.py
python .\scripts\run_124_locked_threshold_external_decision_validation.py
python .\scripts\run_125_canterbury_leave_one_event_temporal_validation.py
python .\scripts\run_126_reader_oriented_figures.py
python .\scripts\run_127_canterbury_severity_triage_validation.py
python .\scripts\run_128_repair_cost_ratio_plausibility_gate.py
python .\scripts\explore_nisqually_out_of_sample_protocols.py
python .\scripts\analyze_sodo_uw_adverse_holdout.py
python .\scripts\select_adversarial_guardrail_model.py
python .\scripts\adversarial_threshold_operating_gate.py
python .\scripts\paired_uncertainty_model_selection_gate.py
```

The synthetic benchmark is fully self-contained. The two site-extension scripts
use public DesignSafe source files: Nisqually PRJ-3758
(`10.17603/ds2-nsf8-7944`) and Canterbury PRJ-2937
(`10.17603/ds2-tygh-ht91`). The Q1 working supplement includes
`scripts/download_designsafe_raw_118.py`, which downloads the required public
raw files into `raw_designsafe/` and writes `raw_designsafe/download_manifest.json`
with file sizes and SHA256 hashes. The scripts still honor the original
Google Drive paths when mounted, but fall back to the local `raw_designsafe/`
folder when those paths are unavailable.

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

The leave-one-earthquake-out Hu validation adds a stricter external screen: each earthquake is held out in turn, simple screening scores are calibrated on the other earthquakes, and the held-out event is predicted. The best pooled model is the combined screening score, with Brier 0.194 and AUC 0.753 across 232 predictions from 15 held-out earthquakes. This supports event-level external screening transfer, but it remains separate from validation of the non-stationary time-update operator.

The locked-threshold decision validation is stricter than the diagnostic frontier because it selects operating thresholds only on the Nisqually leave-one-family domain and then applies them unchanged to SODO/UW and Canterbury. At FN:FP = 10, the M1 groundwater-only rule selected at threshold 0.075 reduces Canterbury false negatives from 99 under M0 to 9, with additional false positives. The full-domain comparison remains restricted to models available in both external domains and continues to favour M0, so this result supports domain-conditional false-negative management rather than universal non-stationary superiority.

The Canterbury leave-one-event validation adds a formal temporal holdout within the largest external dataset. Each earthquake is removed in turn, models are fitted on the other two earthquakes, and the held-out event is predicted. Across 15,890 unique event states and 63,560 model predictions, M2 is the best pooled model by both Brier score (0.138) and AUC (0.834). Event-wise results remain mixed, especially for the low-prevalence 2016 event, so the manuscript reports this as evidence for temporal transfer and rank discrimination, not universal model dominance.

The compact main manuscript uses a Feynman-style reading path: simple causal language, fewer main-text tables, and three reader-facing figures. Detailed input tables, convergence checks, sensitivity coefficients, secondary site-extension figures and extended claim matrices are kept in the supplementary material and repository outputs so the main manuscript remains under 30 Word-computed pages without losing auditability.

The severity-triage validation adds an action-style proxy. Observed moderate-to-severe Canterbury manifestation is defined as manifestation_code >= 3, and models are evaluated by how many severe held-out event states they capture in the top 10% and top 20% highest predicted-probability inspection set. In the two events with severe manifestations, M2 captures more severe states than M0 in the top-20% set; 2016 has no severe manifestations and is reported as a blocked severity case.

The repair-cost ratio plausibility gate uses DesignSafe PRJ-3126 New Zealand earthquake-damaged component repair costs/times. The PRJ-3126 master table has no shared CPT/event/geospatial key with Canterbury, so it is not direct consequence validation. It does support the plausibility of false-negative cost ratios greater than one: DS3/DS1 median repair-cost ratio is 3.56 and DS4/DS1 median repair-cost ratio is 10.34 among components with available values.

## Site-calibrated extension

The Nisqually extension uses the DesignSafe PRJ-3758 CPT case-history dataset. It separates the reproducible synthetic benchmark from a documented-site application. The available data include CPT depth profiles, coordinates, CPT dates, reported water-table depth, conditional PGA and liquefaction manifestation. FC, D50 and laboratory gradation histories are not reported in the downloaded PRJ-3758 XLSX files; gradation is therefore treated as baseline uncertainty/proxy information through CPT-derived Ic, and the synthetic fines-accumulation/fines-washout paths remain scenario sensitivities rather than validated site predictions.

The validation protocol is intentionally out-of-sample. The primary protocol uses leave-one-site-family-out spatial validation, pools the held-out predictions, and reports calibration diagnostics on those predictions. Under this protocol, M2 (`nonstationary_groundwater_gradation`) improves Brier score relative to M0 (`static_stationary`), from 0.072 to 0.026 in pooled predictions and from 0.088 to 0.025 as mean fold Brier. Calibration remains imperfect because the dataset is small and strongly separable; calibration slopes are reported explicitly rather than hidden. Exhaustive paired-family and strict SODO+UW holdouts are also retained as sensitivity checks; the paired-family protocol also favours M2, while the single SODO+UW split does not.

The adversarial guardrail is prespecified and does not erase that adverse result. It selects M2 for the pooled Nisqually spatial protocol and M3 for the minimal Canterbury temporal transfer, but selects M0 for the strict SODO/UW industrial-waterway holdout because M2 and M3 each introduce one false negative. A separate conservative max(M0..M3) screening rule is reported only as a screening guardrail: it eliminates the SODO/UW false negative and yields Brier 0.036, but it is not used to claim superior probabilistic calibration where it worsens Brier. The enabled claim is therefore cluster-aware non-stationary reliability updating with an explicit conservative adverse-holdout guardrail, not universal superiority of M2/M3.

The paired uncertainty gate keeps this interpretation conservative. In the 24-case pooled Nisqually leave-one-family protocol, M2 improves the point-estimate Brier score against M0 by -0.0466, but the paired bootstrap 95% interval reaches zero (-0.1146, 0.0092). In the 8-case SODO/UW strict holdout, M2 has an adverse point-estimate delta of +0.0213 with an interval spanning zero. The original Canterbury three-event transfer is retained as a minimal temporal stress test.

The expanded Canterbury transfer removes the small-n limitation by applying the Nisqually-trained models to 15,890 scored event states across 5,665 usable CPT records in DesignSafe PRJ-2937 after excluding 1,114 manifestation-code-10 states as indeterminate. This larger external domain does not support a universal non-stationary Brier-superiority claim: M0 is the best Brier model (0.318), while M3 worsens Brier by +0.0885 with a paired bootstrap 95% interval of +0.0846 to +0.0927. The non-stationary and conservative screening forms instead reduce false negatives: M3 reduces false negatives from 224 to 111, and conservative max(M0..M3) reduces them to 43, but at the cost of many additional false positives and worse calibration. These results support a Georisk-style risk-screening/guardrail claim, not a claim of universally better calibrated probabilities.

The Canterbury temporal extension still includes the one-site sequence using DesignSafe PRJ-2937, `100 Osbourne St - CPT01 TabulatedData`. The same CPT profile is evaluated for the 2010, 2011 and 2016 earthquake states using documented Mw, PGA, groundwater depth and manifestation codes. A transfer model trained on the Nisqually case histories is then applied to the Canterbury sequence. This remains a transparent three-event stress test; the expanded multi-site transfer is the stronger external validation gate.

The cost-sensitive decision frontier converts that bounded evidence into an explicit georisk operating-policy diagnostic. The domain-retuned frontier is labelled as an oracle diagnostic because it chooses thresholds from the validation labels; under that diagnostic, M0 remains the lowest-loss baseline across the tested false-negative/false-positive cost ratios. When the standard 0.50 threshold is retained, conservative max(M0..M3) becomes the lowest-loss fixed-threshold screen once a false negative is at least twice as costly as a false positive. At a 10:1 cost ratio, its domain-balanced mean expected loss is 0.189 compared with 0.465 for M0; the case-weighted global mean is 0.484 compared with 0.521 for M0. This is the manuscript's stronger novelty claim: a transparent decision frontier for choosing between calibration-favoured and missed-event-averse screening policies under explicit risk-management preferences, not probability recalibration.

## Methodological note

The profile is synthetic and transparent. It is intended to test the proposed non-stationary reliability workflow, not to claim calibration to a specific field site. Quantitative site calibration requires field or laboratory data and should be treated as future work unless a site dataset is added.

The file `data/global_sensitivity_rank.csv` reports rank-correlation sensitivity screening using Pearson and Spearman correlations. It should not be interpreted as a Sobol or variance-decomposition sensitivity analysis.

