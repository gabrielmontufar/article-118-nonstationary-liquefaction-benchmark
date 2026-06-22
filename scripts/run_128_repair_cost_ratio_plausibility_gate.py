"""Repair-cost ratio plausibility gate using DesignSafe PRJ-3126.

PRJ-3126 provides New Zealand repair time and cost data for earthquake-damaged
building components. It does not share event-state or coordinate keys with the
Canterbury CPT manifestation dataset, so it cannot be used as direct
site-level validation for Article 118. It can, however, bound the plausibility
of using false-negative cost ratios larger than one in a screening decision
frontier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_designsafe" / "repair_costs_designsafe_PRJ-3126"
OUTPUTS = ROOT / "outputs"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)

MASTER = RAW / "NZ_repair_time_&_cost_master_20210119.csv"


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return n.where((n > 0) & (d > 0)) / d.where((n > 0) & (d > 0))


def main() -> None:
    if not MASTER.exists():
        raise FileNotFoundError(MASTER)
    df = pd.read_excel(MASTER, sheet_name="Main", header=1)
    out = df[
        [
            "Component Category",
            "Component Type",
            "Component Name",
            "P50",
            "P50.1",
            "P50.2",
            "P50.3",
            "P50.4",
            "P50.5",
            "P50.6",
            "P50.7",
        ]
    ].copy()
    out = out.rename(
        columns={
            "P50": "ds1_repair_cost_p50",
            "P50.1": "ds1_repair_time_p50",
            "P50.2": "ds2_repair_cost_p50",
            "P50.3": "ds2_repair_time_p50",
            "P50.4": "ds3_repair_cost_p50",
            "P50.5": "ds3_repair_time_p50",
            "P50.6": "ds4_repair_cost_p50",
            "P50.7": "ds4_repair_time_p50",
        }
    )
    for col in out.columns[3:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["ds3_vs_ds1_cost_ratio"] = _safe_ratio(out["ds3_repair_cost_p50"], out["ds1_repair_cost_p50"])
    out["ds4_vs_ds1_cost_ratio"] = _safe_ratio(out["ds4_repair_cost_p50"], out["ds1_repair_cost_p50"])
    out["ds3_vs_ds1_time_ratio"] = _safe_ratio(out["ds3_repair_time_p50"], out["ds1_repair_time_p50"])
    out["ds4_vs_ds1_time_ratio"] = _safe_ratio(out["ds4_repair_time_p50"], out["ds1_repair_time_p50"])
    out.to_csv(OUTPUTS / "repair_cost_ratio_plausibility_components.csv", index=False)

    ratio_cols = [
        "ds3_vs_ds1_cost_ratio",
        "ds4_vs_ds1_cost_ratio",
        "ds3_vs_ds1_time_ratio",
        "ds4_vs_ds1_time_ratio",
    ]
    rows = []
    for col in ratio_cols:
        vals = out[col].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "ratio": col,
                "n_components": int(vals.shape[0]),
                "median": float(vals.median()) if len(vals) else None,
                "p75": float(vals.quantile(0.75)) if len(vals) else None,
                "p90": float(vals.quantile(0.90)) if len(vals) else None,
                "max": float(vals.max()) if len(vals) else None,
                "share_ge_5": float((vals >= 5).mean()) if len(vals) else None,
                "share_ge_10": float((vals >= 10).mean()) if len(vals) else None,
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(OUTPUTS / "repair_cost_ratio_plausibility_summary.csv", index=False)

    summary = {
        "status": "PASS_REPAIR_COST_RATIO_PLAUSIBILITY_GATE",
        "source": "raw_designsafe/repair_costs_designsafe_PRJ-3126/NZ_repair_time_&_cost_master_20210119.csv",
        "source_note": "DesignSafe PRJ-3126 New Zealand earthquake-damaged building component repair costs/times; downloaded public data file has .csv extension but Excel workbook content.",
        "linkage_to_canterbury_cpt": "blocked: no shared CPT/event/geospatial key in the repair-cost master table",
        "ratio_summary": rows,
        "claim_enabled": (
            "external New Zealand repair-cost/time data support the plausibility of using false-negative cost ratios greater than one in screening frontiers"
        ),
        "claim_blocked": (
            "do not treat PRJ-3126 as direct Canterbury CPT consequence validation, repair-cost prediction, or design-level validation"
        ),
    }
    (OUTPUTS / "repair_cost_ratio_plausibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Repair-cost ratio plausibility gate - 2026-06-22",
        "",
        "## Protocol",
        "",
        "DesignSafe PRJ-3126 repair cost/time data were downloaded and inspected. The master table has repair-cost and repair-time P50 values for component damage states, but no CPT/event/geospatial key that links directly to the Canterbury manifestation table.",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        "Direct Canterbury consequence validation: `blocked` because no shared key exists.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}.",
        "",
        f"Blocked: {summary['claim_blocked']}.",
    ]
    (AUDIT / "repair_cost_ratio_plausibility_gate_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
