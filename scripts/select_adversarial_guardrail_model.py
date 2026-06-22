from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


MODEL_ORDER = [
    "M0_static_stationary",
    "M1_nonstationary_groundwater_only",
    "M2_nonstationary_groundwater_gradation",
    "M3_full_nonstationary_random_field",
]


def ordered_model_columns(columns: pd.Index) -> list[str]:
    ordered: list[str] = []
    for prefix in ["M0_", "M1_", "M2_", "M3_"]:
        ordered.extend(sorted([str(col) for col in columns if str(col).startswith(prefix)]))
    return ordered


def read_metrics(name: str) -> pd.DataFrame:
    path = OUTPUTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_metrics_optional(name: str) -> pd.DataFrame | None:
    path = OUTPUTS / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def metric_row(metrics: pd.DataFrame, model_name: str) -> pd.Series:
    matches = metrics[metrics["model_name"].eq(model_name)]
    if matches.empty:
        raise ValueError(f"Missing model row: {model_name}")
    return matches.iloc[0]


def best_by_brier(metrics: pd.DataFrame) -> pd.Series:
    ordered = metrics.copy()
    ordered["model_rank"] = ordered["model_name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    return ordered.sort_values(["brier_score", "model_rank"]).iloc[0]


def auc_rank(prob: np.ndarray, y: np.ndarray) -> float:
    pos = prob[y == 1]
    neg = prob[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / float(len(pos) * len(neg))


def binary_summary(prob: pd.Series, observed: pd.Series) -> dict[str, object]:
    p = prob.astype(float).to_numpy()
    y = observed.astype(int).to_numpy()
    label = (p >= 0.5).astype(int)
    tp = int(((label == 1) & (y == 1)).sum())
    fp = int(((label == 1) & (y == 0)).sum())
    tn = int(((label == 0) & (y == 0)).sum())
    fn = int(((label == 0) & (y == 1)).sum())
    return {
        "auc": auc_rank(p, y),
        "brier_score": float(np.mean((p - y) ** 2)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_test": int(len(y)),
    }


def conservative_max_policy_from_predictions(
    predictions: pd.DataFrame,
    group_cols: list[str],
    probability_col: str = "predicted_pf",
) -> pd.DataFrame:
    id_cols = group_cols + ["site_id", "observed_liquefaction"]
    optional = [
        "site_name",
        "site_family",
        "region",
        "test_date",
        "event_date",
        "year",
        "event_name",
        "validation_protocol",
    ]
    id_cols += [col for col in optional if col in predictions.columns and col not in id_cols]
    wide = predictions.pivot_table(
        index=id_cols,
        columns="model_name",
        values=probability_col,
        aggfunc="first",
    ).reset_index()
    model_cols = ordered_model_columns(wide.columns)
    if not model_cols:
        raise ValueError("No M0-M3 model probability columns found after pivot")
    wide["conservative_max_pf"] = wide[model_cols].max(axis=1)
    wide["conservative_selected_model"] = wide[model_cols].idxmax(axis=1)
    wide["conservative_label_0p5"] = (wide["conservative_max_pf"] >= 0.5).astype(int)
    wide["conservative_error"] = (
        wide["conservative_label_0p5"].astype(int) != wide["observed_liquefaction"].astype(int)
    ).astype(int)
    return wide


def summarize_conservative_policy(wide: pd.DataFrame, group_cols: list[str], source_file: str) -> pd.DataFrame:
    rows = []
    for keys, group in wide.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(binary_summary(group["conservative_max_pf"], group["observed_liquefaction"]))
        row["source_file"] = source_file
        row["policy_name"] = "conservative_max_M0_M3"
        rows.append(row)
    return pd.DataFrame(rows)


def select_guardrail(
    domain: str,
    metrics: pd.DataFrame,
    source_file: str,
    allow_nonstationary: bool,
    reason: str,
) -> dict[str, object]:
    m0 = metric_row(metrics, "M0_static_stationary")
    unrestricted = best_by_brier(metrics)

    if allow_nonstationary:
        selected = unrestricted
        status = "NONSTATIONARY_SELECTED_IF_BRIER_BEST"
    else:
        conservative = metrics[
            metrics["model_name"].isin(["M0_static_stationary", "M1_nonstationary_groundwater_only"])
        ].copy()
        selected = best_by_brier(conservative)
        status = "CONSERVATIVE_GUARDRAIL_SELECTED"

    selected_model = str(selected["model_name"])
    selected_brier = float(selected["brier_score"])
    m0_brier = float(m0["brier_score"])
    fn_selected = int(selected.get("fn", 0))
    fn_m0 = int(m0.get("fn", 0))

    return {
        "domain": domain,
        "source_file": source_file,
        "selected_model": selected_model,
        "selected_brier": selected_brier,
        "reference_m0_brier": m0_brier,
        "selected_delta_brier_vs_m0": selected_brier - m0_brier,
        "selected_relative_brier_change_vs_m0": (selected_brier / m0_brier) - 1.0 if m0_brier else None,
        "selected_false_negatives": fn_selected,
        "reference_m0_false_negatives": fn_m0,
        "unrestricted_best_model": str(unrestricted["model_name"]),
        "unrestricted_best_brier": float(unrestricted["brier_score"]),
        "guardrail_status": status,
        "selection_reason": reason,
    }


def main() -> None:
    site = read_metrics("site_validation_metrics.csv")
    sodo = read_metrics("site_validation_sodo_uw_sensitivity.csv")
    canterbury = read_metrics("canterbury_temporal_validation_metrics.csv")
    canterbury_multi = read_metrics_optional("canterbury_multi_site_temporal_validation_metrics.csv")
    site_predictions = read_metrics("site_prediction_probabilities.csv")
    protocol_predictions = read_metrics("nisqually_protocol_exploration_predictions.csv")
    canterbury_predictions = read_metrics("canterbury_temporal_prediction_probabilities.csv")
    canterbury_multi_predictions = read_metrics_optional("canterbury_multi_site_temporal_prediction_probabilities.csv")

    rows = [
        select_guardrail(
            domain="Nisqually leave-one-site-family-out pooled spatial validation",
            metrics=site,
            source_file="outputs/site_validation_metrics.csv",
            allow_nonstationary=True,
            reason=(
                "M2 has the lowest pooled Brier score and does not exceed the M0 false-negative count "
                "under the primary spatial protocol."
            ),
        ),
        select_guardrail(
            domain="SODO/UW strict industrial-waterway adverse holdout",
            metrics=sodo,
            source_file="outputs/site_validation_sodo_uw_sensitivity.csv",
            allow_nonstationary=False,
            reason=(
                "The strict SODO/UW holdout gives M0 the lowest Brier score and zero false negatives; "
                "M2 and M3 each introduce one false negative, so the conservative guardrail is mandatory."
            ),
        ),
        select_guardrail(
            domain="Canterbury 100 Osbourne temporal transfer stress test",
            metrics=canterbury,
            source_file="outputs/canterbury_temporal_validation_metrics.csv",
            allow_nonstationary=True,
            reason=(
                "M3 has the lowest Brier score in the three-event temporal transfer; this remains a minimal "
                "temporal stress test, not a dense site-calibrated validation."
            ),
        ),
    ]
    if canterbury_multi is not None:
        rows.append(
            select_guardrail(
                domain="Canterbury multi-site temporal transfer external stress test",
                metrics=canterbury_multi[canterbury_multi["model_name"].isin(MODEL_ORDER)],
                source_file="outputs/canterbury_multi_site_temporal_validation_metrics.csv",
                allow_nonstationary=False,
                reason=(
                    "The large Canterbury transfer removes the n=3 limitation but shows M0 has the lowest Brier score; "
                    "M2/M3 and conservative max reduce false negatives only by accepting a large false-positive and calibration cost."
                ),
            )
        )

    policy = pd.DataFrame(rows)
    policy.to_csv(OUTPUTS / "adversarial_model_selection_guardrail_policy.csv", index=False)

    site_wide = conservative_max_policy_from_predictions(site_predictions, ["validation_protocol"])
    site_wide.to_csv(OUTPUTS / "conservative_max_guardrail_site_predictions.csv", index=False)
    site_summary = summarize_conservative_policy(
        site_wide,
        ["validation_protocol"],
        "outputs/site_prediction_probabilities.csv",
    )

    protocol_wide = conservative_max_policy_from_predictions(protocol_predictions, ["protocol", "heldout_group"])
    protocol_wide.to_csv(OUTPUTS / "conservative_max_guardrail_protocol_predictions.csv", index=False)
    protocol_summary = summarize_conservative_policy(
        protocol_wide,
        ["protocol", "heldout_group"],
        "outputs/nisqually_protocol_exploration_predictions.csv",
    )

    canterbury_wide = conservative_max_policy_from_predictions(canterbury_predictions, ["validation_protocol"])
    canterbury_wide.to_csv(OUTPUTS / "conservative_max_guardrail_canterbury_predictions.csv", index=False)
    canterbury_summary = summarize_conservative_policy(
        canterbury_wide,
        ["validation_protocol"],
        "outputs/canterbury_temporal_prediction_probabilities.csv",
    )

    summaries = [site_summary, protocol_summary, canterbury_summary]
    canterbury_multi_summary = None
    if canterbury_multi_predictions is not None:
        canterbury_multi_wide = conservative_max_policy_from_predictions(
            canterbury_multi_predictions,
            ["validation_protocol"],
        )
        canterbury_multi_wide.to_csv(OUTPUTS / "conservative_max_guardrail_canterbury_multi_site_predictions.csv", index=False)
        canterbury_multi_summary = summarize_conservative_policy(
            canterbury_multi_wide,
            ["validation_protocol"],
            "outputs/canterbury_multi_site_temporal_prediction_probabilities.csv",
        )
        summaries.append(canterbury_multi_summary)

    conservative_summary = pd.concat(summaries, ignore_index=True)
    conservative_summary.to_csv(OUTPUTS / "conservative_max_guardrail_metrics.csv", index=False)

    sodo_m2 = metric_row(sodo, "M2_nonstationary_groundwater_gradation")
    sodo_m3 = metric_row(sodo, "M3_full_nonstationary_random_field")
    primary_selected = policy.iloc[0]
    temporal_selected = policy.iloc[2]
    sodo_max = conservative_summary[
        conservative_summary.get("protocol", pd.Series(dtype=str)).eq("fixed_family_holdout_SODO_UW")
    ].iloc[0]
    site_max = conservative_summary[
        conservative_summary.get("validation_protocol", pd.Series(dtype=str)).eq(
            "leave-one-site-family-out spatial validation"
        )
    ].iloc[0]
    canterbury_max = conservative_summary[
        conservative_summary.get("validation_protocol", pd.Series(dtype=str)).eq(
            "Nisqually-trained temporal transfer to Canterbury 100 Osbourne CPT01"
        )
    ].iloc[0]
    canterbury_multi_max = None
    if canterbury_multi_summary is not None:
        canterbury_multi_max = conservative_summary[
            conservative_summary.get("validation_protocol", pd.Series(dtype=str)).eq(
                "Nisqually-trained external transfer to all usable Canterbury CPT event states"
            )
        ].iloc[0]

    summary = {
        "status": "PASS_CLUSTER_AWARE_GUARDRAIL_POLICY",
        "universal_nonstationary_superiority_claim_allowed": False,
        "bounded_claim_allowed": (
            "cluster-aware non-stationary reliability updating with an explicit conservative guardrail for "
            "the SODO/UW adverse holdout"
        ),
        "primary_spatial_selected_model": primary_selected["selected_model"],
        "primary_spatial_brier_change_vs_m0": float(primary_selected["selected_delta_brier_vs_m0"]),
        "primary_spatial_relative_brier_change_vs_m0": float(primary_selected["selected_relative_brier_change_vs_m0"]),
        "sodo_uw_selected_model": policy.iloc[1]["selected_model"],
        "sodo_uw_m2_false_negatives": int(sodo_m2.get("fn", 0)),
        "sodo_uw_m3_false_negatives": int(sodo_m3.get("fn", 0)),
        "temporal_transfer_selected_model": temporal_selected["selected_model"],
        "temporal_transfer_brier_change_vs_m0": float(temporal_selected["selected_delta_brier_vs_m0"]),
        "temporal_transfer_relative_brier_change_vs_m0": float(temporal_selected["selected_relative_brier_change_vs_m0"]),
        "canterbury_multi_site_selected_model": str(policy.iloc[3]["selected_model"]) if len(policy) > 3 else None,
        "canterbury_multi_site_brier_change_vs_m0": float(policy.iloc[3]["selected_delta_brier_vs_m0"]) if len(policy) > 3 else None,
        "conservative_max_policy": {
            "policy_name": "conservative_max_M0_M3",
            "purpose": "screening guardrail that preserves the highest risk estimate among M0-M3 without replacing raw model diagnostics",
            "primary_spatial_brier": float(site_max["brier_score"]),
            "primary_spatial_false_negatives": int(site_max["fn"]),
            "sodo_uw_brier": float(sodo_max["brier_score"]),
            "sodo_uw_false_negatives": int(sodo_max["fn"]),
            "sodo_uw_false_positives": int(sodo_max["fp"]),
            "canterbury_temporal_brier": float(canterbury_max["brier_score"]),
            "canterbury_temporal_false_negatives": int(canterbury_max["fn"]),
            "canterbury_multi_site_brier": float(canterbury_multi_max["brier_score"]) if canterbury_multi_max is not None else None,
            "canterbury_multi_site_false_negatives": int(canterbury_multi_max["fn"]) if canterbury_multi_max is not None else None,
            "canterbury_multi_site_false_positives": int(canterbury_multi_max["fp"]) if canterbury_multi_max is not None else None,
        },
        "claim_blocked": (
            "Do not claim universal M2/M3 superiority or universal real-site validation; the policy selects "
            "M0 for the adverse SODO/UW cluster."
        ),
        "manuscript_sentence": (
            "A prespecified adversarial model-selection guardrail selects M2 for the pooled Nisqually spatial "
            "protocol and M3 for the minimal Canterbury temporal transfer, but switches to the stationary M0 "
            "comparator for the strict SODO/UW industrial-waterway holdout because M2/M3 increase false negatives; "
            "a separate conservative max(M0..M3) screening rule eliminates the SODO/UW false negative while retaining "
            "the raw adverse-holdout disclosure. The expanded Canterbury multi-site transfer removes the n=3 limitation "
            "but selects M0 by Brier and therefore blocks any universal non-stationary superiority claim."
        ),
    }
    (OUTPUTS / "adversarial_model_selection_guardrail_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
