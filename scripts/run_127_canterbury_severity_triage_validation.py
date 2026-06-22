"""Canterbury severity-triage validation.

This script tests an action-style question rather than another probability
score: if an engineer can only inspect or prioritise the highest-risk 10% or
20% of held-out Canterbury event states, which model captures more observed
moderate-to-severe manifestations?

The observed action proxy is Canterbury manifestation_code >= 3. The threshold
is deliberately simple and auditable. It is not a repair-cost model and not a
design-level consequence model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)

TOP_FRACTIONS = [0.10, 0.20]
SEVERE_THRESHOLD = 3


def _triage_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    predictions = predictions.copy()
    predictions["severe_manifestation"] = predictions["manifestation_code"].ge(SEVERE_THRESHOLD).astype(int)
    for (event_key, model_name), group in predictions.groupby(["heldout_event_key", "model_name"], sort=False):
        event_name = str(group["event_name"].iloc[0])
        total = int(len(group))
        severe_total = int(group["severe_manifestation"].sum())
        prevalence = severe_total / max(total, 1)
        ranked = group.sort_values("predicted_pf", ascending=False)
        for frac in TOP_FRACTIONS:
            n_action = max(1, int(round(total * frac)))
            action = ranked.head(n_action)
            severe_captured = int(action["severe_manifestation"].sum())
            precision = severe_captured / n_action
            capture = severe_captured / severe_total if severe_total else None
            lift = precision / prevalence if prevalence > 0 else None
            rows.append(
                {
                    "heldout_event_key": event_key,
                    "heldout_event_name": event_name,
                    "model_name": model_name,
                    "top_fraction": frac,
                    "n_event_states": total,
                    "n_action_states": n_action,
                    "severe_threshold": f"manifestation_code >= {SEVERE_THRESHOLD}",
                    "n_severe_event_states": severe_total,
                    "severe_prevalence": prevalence,
                    "severe_captured": severe_captured,
                    "severe_capture_rate": capture,
                    "action_precision": precision,
                    "lift_over_event_prevalence": lift,
                    "action_rule": "inspect_or_prioritise_highest_predicted_pf_states",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    pred_path = OUTPUTS / "canterbury_leave_one_event_temporal_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    predictions = pd.read_csv(pred_path)
    triage = _triage_rows(predictions)
    triage.to_csv(OUTPUTS / "canterbury_severity_triage_validation.csv", index=False)

    valid = triage[triage["n_severe_event_states"].gt(0)].copy()
    best = (
        valid.sort_values(
            ["heldout_event_key", "top_fraction", "severe_capture_rate", "lift_over_event_prevalence"],
            ascending=[True, True, False, False],
        )
        .groupby(["heldout_event_key", "top_fraction"], as_index=False)
        .first()
    )
    best.to_csv(OUTPUTS / "canterbury_severity_triage_best_by_event.csv", index=False)

    top20 = valid[valid["top_fraction"].eq(0.20)].copy()
    m0 = top20[top20["model_name"].eq("M0_static_stationary")].set_index("heldout_event_key")
    best_nonstationary = (
        top20[~top20["model_name"].eq("M0_static_stationary")]
        .sort_values(["heldout_event_key", "severe_capture_rate", "lift_over_event_prevalence"], ascending=[True, False, False])
        .groupby("heldout_event_key", as_index=False)
        .first()
        .set_index("heldout_event_key")
    )
    contrasts = []
    for event_key, row in best_nonstationary.iterrows():
        base = m0.loc[event_key]
        contrasts.append(
            {
                "heldout_event_key": event_key,
                "heldout_event_name": row["heldout_event_name"],
                "best_nonstationary_model": row["model_name"],
                "top_fraction": 0.20,
                "m0_severe_capture_rate": float(base["severe_capture_rate"]),
                "m0_action_precision": float(base["action_precision"]),
                "m0_lift": float(base["lift_over_event_prevalence"]),
                "nonstationary_severe_capture_rate": float(row["severe_capture_rate"]),
                "nonstationary_action_precision": float(row["action_precision"]),
                "nonstationary_lift": float(row["lift_over_event_prevalence"]),
                "n_severe_event_states": int(row["n_severe_event_states"]),
            }
        )

    summary = {
        "status": "PASS_CANTERBURY_SEVERITY_TRIAGE_VALIDATION",
        "source": "outputs/canterbury_leave_one_event_temporal_predictions.csv",
        "severity_definition": f"manifestation_code >= {SEVERE_THRESHOLD}",
        "action_rule": "inspect or prioritise top 10% and top 20% by held-out predicted probability",
        "events_with_severe_manifestations": sorted(valid["heldout_event_key"].unique().tolist()),
        "events_without_severe_manifestations": sorted(
            set(triage["heldout_event_key"].unique()) - set(valid["heldout_event_key"].unique())
        ),
        "top20_nonstationary_vs_m0_by_event": contrasts,
        "claim_enabled": (
            "action-style triage evidence: in Canterbury events with observed severe manifestations, groundwater/gradation-aware models can improve "
            "capture of severe manifestation states in the top-20% inspection set."
        ),
        "claim_blocked": (
            "do not describe this as repair-cost validation, design consequence validation, or a universal severity model; 2016 contains no severe manifestations."
        ),
    }
    (OUTPUTS / "canterbury_severity_triage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Canterbury severity-triage validation - 2026-06-22",
        "",
        "## Protocol",
        "",
        f"Observed severe manifestation is defined as `{summary['severity_definition']}`. The action proxy is to inspect or prioritise the top 10% and top 20% of held-out event states ranked by predicted probability.",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        f"Events with severe manifestations: `{', '.join(summary['events_with_severe_manifestations'])}`.",
        f"Events without severe manifestations: `{', '.join(summary['events_without_severe_manifestations'])}`.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
    ]
    (AUDIT / "canterbury_severity_triage_validation_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
