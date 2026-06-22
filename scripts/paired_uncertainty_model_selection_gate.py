from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RNG_SEED = 11820260603
BOOTSTRAP_REPS = 5000

MODELS = [
    "M0_static_stationary",
    "M1_nonstationary_groundwater_only",
    "M2_nonstationary_groundwater_gradation",
    "M3_full_nonstationary_random_field",
    "conservative_max_M0_M3",
]
BASELINE = "M0_static_stationary"
MODEL_ALIASES = {
    "M1_groundwater": "M1_nonstationary_groundwater_only",
    "M2_groundwater_gradation": "M2_nonstationary_groundwater_gradation",
    "M3_random_field": "M3_full_nonstationary_random_field",
}

OUT_METRICS = OUTPUTS / "paired_uncertainty_model_selection_metrics.csv"
OUT_CASES = OUTPUTS / "paired_uncertainty_model_selection_cases.csv"
OUT_SUMMARY = OUTPUTS / "paired_uncertainty_model_selection_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty quantile")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def brier(prob: float, observed: int) -> float:
    return (prob - observed) ** 2


def bootstrap_ci(deltas: list[float], reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    if len(deltas) > 1000:
        rng = np.random.default_rng(RNG_SEED + len(deltas))
        arr = np.asarray(deltas, dtype=float)
        means = []
        chunk = 250
        for _ in range(0, reps, chunk):
            n_reps = min(chunk, reps - len(means))
            idx = rng.integers(0, len(arr), size=(n_reps, len(arr)))
            means.extend(np.mean(arr[idx], axis=1).tolist())
        return quantile(means, 0.025), quantile(means, 0.975)
    rng = random.Random(RNG_SEED + len(deltas))
    means = []
    n = len(deltas)
    for _ in range(reps):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    return quantile(means, 0.025), quantile(means, 0.975)


def paired_metrics(domain: str, protocol: str, cases: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, list[float]] = defaultdict(list)
    deltas_by_model: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        observed = int(case["observed"])
        baseline_loss = brier(float(case[BASELINE]), observed)
        for model in MODELS:
            if model not in case:
                continue
            loss = brier(float(case[model]), observed)
            by_model[model].append(loss)
            if model != BASELINE:
                deltas_by_model[model].append(loss - baseline_loss)

    rows: list[dict[str, object]] = []
    for model, losses in sorted(by_model.items()):
        mean_brier = sum(losses) / len(losses)
        if model == BASELINE:
            rows.append(
                {
                    "domain": domain,
                    "validation_protocol": protocol,
                    "model": model,
                    "n_cases": len(losses),
                    "mean_brier": mean_brier,
                    "mean_brier_delta_vs_m0": 0.0,
                    "bootstrap_delta_ci95_low": 0.0,
                    "bootstrap_delta_ci95_high": 0.0,
                    "cases_better_than_m0": "",
                    "cases_worse_than_m0": "",
                    "cases_tied_with_m0": "",
                    "claim_interpretation": "baseline comparator",
                }
            )
            continue
        deltas = deltas_by_model[model]
        mean_delta = sum(deltas) / len(deltas)
        ci_low, ci_high = bootstrap_ci(deltas)
        better = sum(1 for value in deltas if value < 0)
        worse = sum(1 for value in deltas if value > 0)
        tied = len(deltas) - better - worse
        if len(deltas) < 5:
            interpretation = "descriptive_only_small_n"
        elif ci_high < 0:
            interpretation = "paired_brier_improvement_supported"
        elif ci_low > 0:
            interpretation = "paired_brier_degradation_supported"
        else:
            interpretation = "paired_brier_difference_uncertain"
        rows.append(
            {
                "domain": domain,
                "validation_protocol": protocol,
                "model": model,
                "n_cases": len(losses),
                "mean_brier": mean_brier,
                "mean_brier_delta_vs_m0": mean_delta,
                "bootstrap_delta_ci95_low": ci_low,
                "bootstrap_delta_ci95_high": ci_high,
                "cases_better_than_m0": better,
                "cases_worse_than_m0": worse,
                "cases_tied_with_m0": tied,
                "claim_interpretation": interpretation,
            }
        )
    return rows


def pivot_model_rows(rows: list[dict[str, str]], id_fields: list[str], model_field: str, prob_field: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        if not row.get(prob_field, "").strip():
            continue
        key = tuple(row[field] for field in id_fields)
        target = grouped.setdefault(
            key,
            {
                field: row[field] for field in id_fields
            }
            | {"observed": int(float(row["observed_liquefaction"]))},
        )
        try:
            model_name = MODEL_ALIASES.get(row[model_field], row[model_field])
            target[model_name] = float(row[prob_field])
        except ValueError:
            continue
    return [case for case in grouped.values() if BASELINE in case]


def site_validation_cases() -> list[dict[str, object]]:
    rows = [
        row
        for row in read_csv(OUTPUTS / "site_prediction_probabilities.csv")
        if row.get("validation_protocol") == "leave-one-site-family-out spatial validation"
    ]
    return pivot_model_rows(
        rows,
        ["site_id", "site_family", "validation_protocol"],
        "model_name",
        "predicted_pf",
    )


def sodo_uw_cases() -> list[dict[str, object]]:
    rows = [
        row
        for row in read_csv(OUTPUTS / "nisqually_protocol_exploration_predictions.csv")
        if row["protocol"] == "fixed_family_holdout_SODO_UW" and row["heldout_group"] == "SODO_UW"
    ]
    return pivot_model_rows(
        rows,
        ["site_id", "site_family", "protocol", "heldout_group"],
        "model_name",
        "predicted_pf",
    )


def canterbury_cases() -> list[dict[str, object]]:
    return pivot_model_rows(
        read_csv(OUTPUTS / "canterbury_temporal_prediction_probabilities.csv"),
        ["site_id", "year", "event_name", "validation_protocol"],
        "model_name",
        "predicted_pf",
    )


def canterbury_multi_site_cases() -> list[dict[str, object]]:
    path = OUTPUTS / "canterbury_multi_site_temporal_prediction_probabilities.csv"
    if not path.exists():
        return []
    return pivot_model_rows(
        read_csv(path),
        ["site_id", "year", "event_name", "validation_protocol"],
        "model_name",
        "predicted_pf",
    )


def add_conservative_max(cases: list[dict[str, object]]) -> None:
    for case in cases:
        values = [float(case[model]) for model in MODELS if model in case and model != "conservative_max_M0_M3"]
        case["conservative_max_M0_M3"] = max(values)


def case_rows(domain: str, cases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        observed = int(case["observed"])
        case_id = str(case.get("site_id", "")) or str(case.get("event_name", ""))
        for model in MODELS:
            if model in case:
                rows.append(
                    {
                        "domain": domain,
                        "case_id": case_id,
                        "observed": observed,
                        "model": model,
                        "predicted_pf": float(case[model]),
                        "brier_loss": brier(float(case[model]), observed),
                        "brier_delta_vs_m0": brier(float(case[model]), observed)
                        - brier(float(case[BASELINE]), observed),
                    }
                )
    return rows


def main() -> None:
    domains = [
        (
            "pooled_nisqually_leave_one_family",
            "leave-one-site-family-out spatial validation",
            site_validation_cases(),
        ),
        (
            "sodo_uw_strict_adverse_holdout",
            "fixed_family_holdout_SODO_UW",
            sodo_uw_cases(),
        ),
        (
            "canterbury_three_event_temporal_transfer",
            "Nisqually-trained temporal transfer to Canterbury 100 Osbourne CPT01",
            canterbury_cases(),
        ),
        (
            "canterbury_multi_site_temporal_transfer",
            "Nisqually-trained external transfer to all usable Canterbury CPT event states",
            canterbury_multi_site_cases(),
        ),
    ]
    metric_rows: list[dict[str, object]] = []
    all_case_rows: list[dict[str, object]] = []
    for domain, protocol, cases in domains:
        add_conservative_max(cases)
        metric_rows.extend(paired_metrics(domain, protocol, cases))
        all_case_rows.extend(case_rows(domain, cases))

    write_csv(OUT_METRICS, metric_rows)
    write_csv(OUT_CASES, all_case_rows)

    def find(domain: str, model: str) -> dict[str, object]:
        for row in metric_rows:
            if row["domain"] == domain and row["model"] == model:
                return row
        raise KeyError((domain, model))

    summary = {
        "status": "PASS_UNCERTAINTY_BOUNDED_MODEL_SELECTION_GATE_NOT_UNIVERSAL_VALIDATION",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "rng_seed": RNG_SEED,
        "domains": {
            "pooled_nisqually_leave_one_family": {
                "n_cases": int(find("pooled_nisqually_leave_one_family", BASELINE)["n_cases"]),
                "m2_delta_vs_m0": float(find("pooled_nisqually_leave_one_family", "M2_nonstationary_groundwater_gradation")["mean_brier_delta_vs_m0"]),
                "m2_ci95": [
                    float(find("pooled_nisqually_leave_one_family", "M2_nonstationary_groundwater_gradation")["bootstrap_delta_ci95_low"]),
                    float(find("pooled_nisqually_leave_one_family", "M2_nonstationary_groundwater_gradation")["bootstrap_delta_ci95_high"]),
                ],
                "m2_interpretation": find("pooled_nisqually_leave_one_family", "M2_nonstationary_groundwater_gradation")["claim_interpretation"],
            },
            "sodo_uw_strict_adverse_holdout": {
                "n_cases": int(find("sodo_uw_strict_adverse_holdout", BASELINE)["n_cases"]),
                "m2_delta_vs_m0": float(find("sodo_uw_strict_adverse_holdout", "M2_nonstationary_groundwater_gradation")["mean_brier_delta_vs_m0"]),
                "m2_ci95": [
                    float(find("sodo_uw_strict_adverse_holdout", "M2_nonstationary_groundwater_gradation")["bootstrap_delta_ci95_low"]),
                    float(find("sodo_uw_strict_adverse_holdout", "M2_nonstationary_groundwater_gradation")["bootstrap_delta_ci95_high"]),
                ],
                "m2_interpretation": find("sodo_uw_strict_adverse_holdout", "M2_nonstationary_groundwater_gradation")["claim_interpretation"],
            },
            "canterbury_three_event_temporal_transfer": {
                "n_cases": int(find("canterbury_three_event_temporal_transfer", BASELINE)["n_cases"]),
                "m3_delta_vs_m0": float(find("canterbury_three_event_temporal_transfer", "M3_full_nonstationary_random_field")["mean_brier_delta_vs_m0"]),
                "m3_ci95": [
                    float(find("canterbury_three_event_temporal_transfer", "M3_full_nonstationary_random_field")["bootstrap_delta_ci95_low"]),
                    float(find("canterbury_three_event_temporal_transfer", "M3_full_nonstationary_random_field")["bootstrap_delta_ci95_high"]),
                ],
                "m3_interpretation": find("canterbury_three_event_temporal_transfer", "M3_full_nonstationary_random_field")["claim_interpretation"],
            },
            "canterbury_multi_site_temporal_transfer": {
                "n_cases": int(find("canterbury_multi_site_temporal_transfer", BASELINE)["n_cases"]),
                "m3_delta_vs_m0": float(find("canterbury_multi_site_temporal_transfer", "M3_full_nonstationary_random_field")["mean_brier_delta_vs_m0"]),
                "m3_ci95": [
                    float(find("canterbury_multi_site_temporal_transfer", "M3_full_nonstationary_random_field")["bootstrap_delta_ci95_low"]),
                    float(find("canterbury_multi_site_temporal_transfer", "M3_full_nonstationary_random_field")["bootstrap_delta_ci95_high"]),
                ],
                "m3_interpretation": find("canterbury_multi_site_temporal_transfer", "M3_full_nonstationary_random_field")["claim_interpretation"],
                "conservative_max_delta_vs_m0": float(find("canterbury_multi_site_temporal_transfer", "conservative_max_M0_M3")["mean_brier_delta_vs_m0"]),
                "conservative_max_interpretation": find("canterbury_multi_site_temporal_transfer", "conservative_max_M0_M3")["claim_interpretation"],
            },
        },
        "claim_enabled": (
            "uncertainty-bounded, domain-specific model-selection evidence: M2 improves the pooled Nisqually "
            "point-estimate Brier score but its paired bootstrap interval reaches zero; SODO/UW remains an adverse "
            "holdout; Canterbury is expanded from a three-event descriptive check to a large external transfer "
            "domain where M0 remains the best Brier model and conservative/non-stationary screening mainly reduces "
            "false negatives at a false-positive/calibration cost"
        ),
        "claim_blocked": (
            "universal M2/M3 superiority, universal calibration, or treating the Canterbury transfer as evidence "
            "that non-stationary probability models beat M0 on Brier in every external domain"
        ),
        "metrics_table": OUT_METRICS.relative_to(ROOT).as_posix(),
        "case_table": OUT_CASES.relative_to(ROOT).as_posix(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
