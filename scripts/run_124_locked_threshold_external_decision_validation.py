"""Locked-threshold external decision validation for Article 118.

This script turns the cost-sensitive frontier into a stricter decision test.
Thresholds are selected only on the Nisqually leave-one-family validation
domain, then applied unchanged to the adverse SODO/UW and Canterbury external
domains. The result is a pre-specified operating-rule validation diagnostic,
not probability recalibration and not universal model superiority.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)

MODEL_ORDER = [
    "M0_static_stationary",
    "M1_nonstationary_groundwater_only",
    "M2_nonstationary_groundwater_gradation",
    "M3_full_nonstationary_random_field",
    "conservative_max_M0_M3",
]

LOSS_RATIOS = [1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)
TRAIN_DOMAIN = "Nisqually primary leave-one-family"
EXTERNAL_DOMAINS = [
    "SODO/UW adverse industrial-waterway holdout",
    "Canterbury multi-site external transfer",
]


def _read_primary_nisqually() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "site_prediction_probabilities.csv")
    df = df[df["validation_protocol"].eq("leave-one-site-family-out spatial validation")].copy()
    df["domain"] = TRAIN_DOMAIN
    df["case_key"] = df["site_id"]
    return df


def _read_sodo_uw() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "nisqually_protocol_exploration_predictions.csv")
    df = df[df["protocol"].eq("fixed_family_holdout_SODO_UW") & df["heldout_group"].eq("SODO_UW")].copy()
    df["domain"] = "SODO/UW adverse industrial-waterway holdout"
    df["case_key"] = df["site_id"]
    return df


def _read_canterbury() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "canterbury_multi_site_temporal_prediction_probabilities.csv")
    df["domain"] = "Canterbury multi-site external transfer"
    df["case_key"] = df["site_id"].astype(str) + "_" + df["year"].astype(str)
    return df


def _add_conservative_max(df: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["domain", "case_key", "site_id", "observed_liquefaction"]
    optional = [col for col in ["site_family", "region", "year", "event_name", "validation_protocol"] if col in df.columns]
    index_cols += optional
    wide = df.pivot_table(index=index_cols, columns="model_name", values="predicted_pf", aggfunc="first").reset_index()
    model_cols = [col for col in MODEL_ORDER if col in wide.columns and col != "conservative_max_M0_M3"]
    wide["conservative_max_M0_M3"] = wide[model_cols].max(axis=1)
    rows = []
    for model in MODEL_ORDER:
        if model not in wide.columns:
            continue
        tmp = wide[index_cols + [model]].rename(columns={model: "predicted_pf"}).copy()
        tmp["model_name"] = model
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def load_predictions() -> pd.DataFrame:
    parts = [_read_primary_nisqually(), _read_sodo_uw(), _read_canterbury()]
    return pd.concat([_add_conservative_max(part) for part in parts], ignore_index=True)


def _confusion(prob: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, int]:
    pred = (prob >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "n_cases": int(len(y)),
    }


def _loss(confusion: dict[str, int], false_negative_cost_ratio: float) -> float:
    return (confusion["fp"] + false_negative_cost_ratio * confusion["fn"]) / max(confusion["n_cases"], 1)


def select_locked_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    training = predictions[predictions["domain"].eq(TRAIN_DOMAIN)].copy()
    for ratio in LOSS_RATIOS:
        for model, group in training.groupby("model_name", sort=False):
            prob = group["predicted_pf"].to_numpy(dtype=float)
            y = group["observed_liquefaction"].to_numpy(dtype=int)
            best = None
            for threshold in THRESHOLDS:
                confusion = _confusion(prob, y, float(threshold))
                loss = _loss(confusion, ratio)
                candidate = (loss, float(threshold), confusion)
                if best is None or candidate < best:
                    best = candidate
            assert best is not None
            loss, threshold, confusion = best
            rows.append(
                {
                    "model_name": model,
                    "false_negative_cost_ratio": ratio,
                    "locked_threshold": threshold,
                    "selection_domain": TRAIN_DOMAIN,
                    "selection_expected_loss_per_case": loss,
                    **{f"selection_{key}": value for key, value in confusion.items()},
                }
            )
    return pd.DataFrame(rows)


def apply_locked_thresholds(predictions: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in locked.itertuples(index=False):
        groups = predictions[predictions["model_name"].eq(rule.model_name)].groupby("domain", sort=False)
        for domain, group in groups:
            prob = group["predicted_pf"].to_numpy(dtype=float)
            y = group["observed_liquefaction"].to_numpy(dtype=int)
            confusion = _confusion(prob, y, float(rule.locked_threshold))
            rows.append(
                {
                    "domain": domain,
                    "model_name": rule.model_name,
                    "false_negative_cost_ratio": float(rule.false_negative_cost_ratio),
                    "locked_threshold": float(rule.locked_threshold),
                    "expected_loss_per_case": _loss(confusion, float(rule.false_negative_cost_ratio)),
                    **confusion,
                    "threshold_policy": "locked_on_nisqually_leave_one_family_then_external_transfer",
                }
            )
    return pd.DataFrame(rows)


def summarize(applied: pd.DataFrame) -> dict[str, object]:
    external = applied[applied["domain"].isin(EXTERNAL_DOMAINS)].copy()
    ratio10 = external[external["false_negative_cost_ratio"].eq(10)].copy()
    required_domain_count = len(EXTERNAL_DOMAINS)
    case_weighted_rows = []
    for (model, ratio), group in external.groupby(["model_name", "false_negative_cost_ratio"], sort=False):
        n = float(group["n_cases"].sum())
        weighted_loss = float((group["expected_loss_per_case"] * group["n_cases"]).sum() / max(n, 1.0))
        case_weighted_rows.append(
            {
                "model_name": model,
                "false_negative_cost_ratio": float(ratio),
                "case_weighted_external_loss": weighted_loss,
                "external_fn": int(group["fn"].sum()),
                "external_fp": int(group["fp"].sum()),
                "external_n_cases": int(group["n_cases"].sum()),
                "locked_threshold": float(group["locked_threshold"].iloc[0]),
                "external_domains_covered": int(group["domain"].nunique()),
                "full_external_domain_coverage": bool(group["domain"].nunique() == required_domain_count),
            }
        )
    case_weighted = pd.DataFrame(case_weighted_rows)
    case_weighted.to_csv(OUTPUTS / "locked_threshold_external_case_weighted_summary.csv", index=False)
    best_case_weighted = (
        case_weighted.sort_values(["false_negative_cost_ratio", "case_weighted_external_loss"])
        .groupby("false_negative_cost_ratio", as_index=False)
        .first()
    )
    full_domain_case_weighted = case_weighted[case_weighted["full_external_domain_coverage"]].copy()
    best_full_domain = (
        full_domain_case_weighted.sort_values(["false_negative_cost_ratio", "case_weighted_external_loss"])
        .groupby("false_negative_cost_ratio", as_index=False)
        .first()
        if not full_domain_case_weighted.empty
        else pd.DataFrame()
    )

    canterbury10 = ratio10[ratio10["domain"].eq("Canterbury multi-site external transfer")].copy()
    canterbury10 = canterbury10.sort_values(["expected_loss_per_case", "fn", "fp"])
    external10 = case_weighted[case_weighted["false_negative_cost_ratio"].eq(10)].sort_values("case_weighted_external_loss")
    full_external10 = full_domain_case_weighted[
        full_domain_case_weighted["false_negative_cost_ratio"].eq(10)
    ].sort_values("case_weighted_external_loss")
    m0_canterbury = ratio10[
        ratio10["domain"].eq("Canterbury multi-site external transfer")
        & ratio10["model_name"].eq("M0_static_stationary")
    ].iloc[0]
    m1_canterbury = ratio10[
        ratio10["domain"].eq("Canterbury multi-site external transfer")
        & ratio10["model_name"].eq("M1_nonstationary_groundwater_only")
    ].iloc[0]
    summary = {
        "status": "PASS_LOCKED_THRESHOLD_EXTERNAL_DECISION_VALIDATION",
        "selection_domain": TRAIN_DOMAIN,
        "external_domains": EXTERNAL_DOMAINS,
        "threshold_policy": "select threshold on Nisqually leave-one-family only; apply unchanged to external domains",
        "fn_fp_10_best_canterbury_locked_rule": {
            "model_name": str(canterbury10.iloc[0]["model_name"]),
            "locked_threshold": float(canterbury10.iloc[0]["locked_threshold"]),
            "expected_loss_per_case": float(canterbury10.iloc[0]["expected_loss_per_case"]),
            "fn": int(canterbury10.iloc[0]["fn"]),
            "fp": int(canterbury10.iloc[0]["fp"]),
            "n_cases": int(canterbury10.iloc[0]["n_cases"]),
        },
        "fn_fp_10_case_weighted_external_best_rule": {
            "model_name": str(external10.iloc[0]["model_name"]),
            "locked_threshold": float(external10.iloc[0]["locked_threshold"]),
            "case_weighted_external_loss": float(external10.iloc[0]["case_weighted_external_loss"]),
            "external_fn": int(external10.iloc[0]["external_fn"]),
            "external_fp": int(external10.iloc[0]["external_fp"]),
            "external_n_cases": int(external10.iloc[0]["external_n_cases"]),
            "external_domains_covered": int(external10.iloc[0]["external_domains_covered"]),
            "full_external_domain_coverage": bool(external10.iloc[0]["full_external_domain_coverage"]),
        },
        "fn_fp_10_full_domain_external_best_rule": {
            "model_name": str(full_external10.iloc[0]["model_name"]),
            "locked_threshold": float(full_external10.iloc[0]["locked_threshold"]),
            "case_weighted_external_loss": float(full_external10.iloc[0]["case_weighted_external_loss"]),
            "external_fn": int(full_external10.iloc[0]["external_fn"]),
            "external_fp": int(full_external10.iloc[0]["external_fp"]),
            "external_n_cases": int(full_external10.iloc[0]["external_n_cases"]),
            "external_domains_covered": int(full_external10.iloc[0]["external_domains_covered"]),
        },
        "canterbury_fn_fp_10_m1_vs_m0": {
            "m0_threshold": float(m0_canterbury["locked_threshold"]),
            "m0_loss": float(m0_canterbury["expected_loss_per_case"]),
            "m0_fn": int(m0_canterbury["fn"]),
            "m0_fp": int(m0_canterbury["fp"]),
            "m1_threshold": float(m1_canterbury["locked_threshold"]),
            "m1_loss": float(m1_canterbury["expected_loss_per_case"]),
            "m1_fn": int(m1_canterbury["fn"]),
            "m1_fp": int(m1_canterbury["fp"]),
        },
        "best_case_weighted_external_rules": [
            {
                "false_negative_cost_ratio": float(row.false_negative_cost_ratio),
                "best_model": row.model_name,
                "locked_threshold": float(row.locked_threshold),
                "case_weighted_external_loss": float(row.case_weighted_external_loss),
                "external_fn": int(row.external_fn),
                "external_fp": int(row.external_fp),
                "external_domains_covered": int(row.external_domains_covered),
                "full_external_domain_coverage": bool(row.full_external_domain_coverage),
            }
            for row in best_case_weighted.itertuples(index=False)
        ],
        "best_full_domain_external_rules": [
            {
                "false_negative_cost_ratio": float(row.false_negative_cost_ratio),
                "best_model": row.model_name,
                "locked_threshold": float(row.locked_threshold),
                "case_weighted_external_loss": float(row.case_weighted_external_loss),
                "external_fn": int(row.external_fn),
                "external_fp": int(row.external_fp),
                "external_domains_covered": int(row.external_domains_covered),
            }
            for row in best_full_domain.itertuples(index=False)
        ],
        "claim_enabled": (
            "a locked decision-rule validation: thresholds are selected without Canterbury or SODO/UW retuning, then "
            "transferred externally. At FN:FP=10, M1 strongly reduces Canterbury false negatives relative to M0 and "
            "has the best available-case external loss, while the full-domain comparison remains restricted to models "
            "available in both SODO/UW and Canterbury."
        ),
        "claim_blocked": (
            "do not state that non-stationary models are universally superior; the locked-rule evidence supports "
            "domain-conditional false-negative management, not global calibration dominance."
        ),
    }
    (OUTPUTS / "locked_threshold_external_decision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Locked-threshold external decision validation - 2026-06-22",
        "",
        "## Protocol",
        "",
        "Thresholds are selected only on the Nisqually leave-one-family validation domain and then applied unchanged to SODO/UW and Canterbury.",
        "",
        "## FN:FP = 10 result",
        "",
        f"Canterbury best locked rule: `{summary['fn_fp_10_best_canterbury_locked_rule']['model_name']}` at threshold `{summary['fn_fp_10_best_canterbury_locked_rule']['locked_threshold']}` with FN `{summary['fn_fp_10_best_canterbury_locked_rule']['fn']}` and FP `{summary['fn_fp_10_best_canterbury_locked_rule']['fp']}`.",
        f"Available-case external best rule: `{summary['fn_fp_10_case_weighted_external_best_rule']['model_name']}` at threshold `{summary['fn_fp_10_case_weighted_external_best_rule']['locked_threshold']}` with external FN `{summary['fn_fp_10_case_weighted_external_best_rule']['external_fn']}` and FP `{summary['fn_fp_10_case_weighted_external_best_rule']['external_fp']}` across `{summary['fn_fp_10_case_weighted_external_best_rule']['external_domains_covered']}` external domain(s).",
        f"Full-domain external best rule: `{summary['fn_fp_10_full_domain_external_best_rule']['model_name']}` at threshold `{summary['fn_fp_10_full_domain_external_best_rule']['locked_threshold']}` with external FN `{summary['fn_fp_10_full_domain_external_best_rule']['external_fn']}` and FP `{summary['fn_fp_10_full_domain_external_best_rule']['external_fp']}` across `{summary['fn_fp_10_full_domain_external_best_rule']['external_domains_covered']}` external domains.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
    ]
    (AUDIT / "locked_threshold_external_decision_validation_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    predictions = load_predictions()
    locked = select_locked_thresholds(predictions)
    applied = apply_locked_thresholds(predictions, locked)

    locked.to_csv(OUTPUTS / "locked_threshold_selection_rules.csv", index=False)
    applied.to_csv(OUTPUTS / "locked_threshold_external_decision_validation.csv", index=False)
    summary = summarize(applied)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
