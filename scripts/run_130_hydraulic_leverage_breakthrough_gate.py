"""Hydraulic-leverage physical-regime gate.

This gate tests a physical claim rather than a packaging claim:

    False negatives concentrate where transient groundwater shallowing amplifies
    cyclic stress ratio through effective-stress loss, and where the CPT fabric
    proxy indicates loose/fine-sensitive material.

The analysis deliberately uses held-out Canterbury predictions from the
leave-one-event protocol. The physical index is computed from first principles:

    H(z) = sigma'_v(z, reference water table) / sigma'_v(z, event water table)

because CSR is proportional to sigma_v / sigma'_v for fixed PGA and depth.
If H is high, a small water-table change has large mechanical leverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation_metrics import auc_rank, binary_metrics


DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)

GAMMA_MOIST = 18.0
GAMMA_SAT = 19.0
GAMMA_W = 9.81
DEPTHS_M = [2.0, 3.0, 4.0, 6.0]


def _effective_stress(depth_m: float, water_depth_m: np.ndarray) -> np.ndarray:
    z = float(depth_m)
    wtd = np.clip(np.asarray(water_depth_m, dtype=float), 0.0, 100.0)
    above = wtd >= z
    sigma = np.where(
        above,
        GAMMA_MOIST * z,
        GAMMA_MOIST * wtd + (GAMMA_SAT - GAMMA_W) * (z - wtd),
    )
    return np.maximum(sigma, 1.0)


def _assign_deciles(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 10, labels=False, duplicates="drop") + 1


def main() -> None:
    features = pd.read_csv(DATA / "canterbury_multi_site_event_features.csv")
    preds = pd.read_csv(OUTPUTS / "canterbury_leave_one_event_temporal_predictions.csv")

    ref_wtd = (
        features.groupby("site_id", as_index=False)["wtd_event_site_adjusted_m"]
        .median()
        .rename(columns={"wtd_event_site_adjusted_m": "site_reference_wtd_m"})
    )
    phys = features.merge(ref_wtd, on="site_id", how="left", validate="many_to_one")
    for depth in DEPTHS_M:
        ref = _effective_stress(depth, phys["site_reference_wtd_m"].to_numpy())
        event = _effective_stress(depth, phys["wtd_event_site_adjusted_m"].to_numpy())
        phys[f"hydraulic_leverage_z{int(depth)}m"] = ref / event
    leverage_cols = [f"hydraulic_leverage_z{int(d)}m" for d in DEPTHS_M]
    phys["hydraulic_leverage_max_2_6m"] = phys[leverage_cols].max(axis=1)
    phys["hydraulic_leverage_depth_m"] = phys[leverage_cols].idxmax(axis=1).str.extract(r"z([0-9]+)m").astype(float)
    phys["loose_tail"] = phys["qc10_mpa"].le(phys["qc10_mpa"].quantile(0.25)).astype(int)
    phys["fine_sensitive_tail"] = phys["ic_median"].ge(phys["ic_median"].quantile(0.75)).astype(int)
    phys["high_hydraulic_leverage"] = phys["hydraulic_leverage_max_2_6m"].ge(phys["hydraulic_leverage_max_2_6m"].quantile(0.75)).astype(int)
    phys["hydraulic_fabric_gate"] = (
        phys["high_hydraulic_leverage"].eq(1) & phys["loose_tail"].eq(1) & phys["fine_sensitive_tail"].eq(1)
    ).astype(int)
    phys["severe_manifestation"] = phys["manifestation_code"].ge(3).astype(int)
    phys["leverage_decile"] = _assign_deciles(phys["hydraulic_leverage_max_2_6m"])

    joined = preds.merge(
        phys[
            [
                "site_id",
                "event_key",
                "site_reference_wtd_m",
                "hydraulic_leverage_max_2_6m",
                "hydraulic_leverage_depth_m",
                "hydraulic_fabric_gate",
                "high_hydraulic_leverage",
                "loose_tail",
                "fine_sensitive_tail",
                "severe_manifestation",
                "leverage_decile",
            ]
        ],
        on=["site_id", "event_key"],
        how="left",
        validate="many_to_one",
    )

    m0 = joined[joined["model_name"].eq("M0_static_stationary")].copy()
    m2 = joined[joined["model_name"].eq("M2_nonstationary_groundwater_gradation")].copy()
    m0["m0_label_0p5"] = m0["predicted_pf"].ge(0.5).astype(int)
    m0["m0_false_negative_0p5"] = (m0["observed_liquefaction"].eq(1) & m0["m0_label_0p5"].eq(0)).astype(int)

    gate_summary = []
    for label, group in [
        ("inside_hydraulic_fabric_gate", m0[m0["hydraulic_fabric_gate"].eq(1)]),
        ("outside_hydraulic_fabric_gate", m0[m0["hydraulic_fabric_gate"].eq(0)]),
    ]:
        gate_summary.append(
            {
                "regime": label,
                "n_event_states": int(len(group)),
                "observed_liquefaction_rate": float(group["observed_liquefaction"].mean()),
                "severe_manifestation_rate": float(group["severe_manifestation"].mean()),
                "m0_false_negative_rate_0p5": float(group["m0_false_negative_0p5"].mean()),
                "median_hydraulic_leverage": float(group["hydraulic_leverage_max_2_6m"].median()),
                "median_qc10_mpa": float(group["qc10_mpa"].median()),
                "median_ic": float(group["ic_median"].median()),
            }
        )
    gate_summary_df = pd.DataFrame(gate_summary)

    decile = (
        m0.groupby("leverage_decile", as_index=False)
        .agg(
            n_event_states=("site_id", "size"),
            median_hydraulic_leverage=("hydraulic_leverage_max_2_6m", "median"),
            observed_liquefaction_rate=("observed_liquefaction", "mean"),
            severe_manifestation_rate=("severe_manifestation", "mean"),
            m0_false_negative_rate_0p5=("m0_false_negative_0p5", "mean"),
        )
        .sort_values("leverage_decile")
    )

    phys_metrics = []
    for target in ["observed_liquefaction", "severe_manifestation", "m0_false_negative_0p5"]:
        labels = m0[target].astype(int).to_numpy()
        scores = m0["hydraulic_leverage_max_2_6m"].astype(float).to_numpy()
        phys_metrics.append(
            {
                "target": target,
                "n": int(len(m0)),
                "n_positive": int(labels.sum()),
                "auc_hydraulic_leverage": auc_rank(scores, labels),
            }
        )
    phys_metrics_df = pd.DataFrame(phys_metrics)

    model_gate_metrics = []
    for model_name, group in joined.groupby("model_name"):
        for regime_name, regime in [
            ("inside_hydraulic_fabric_gate", group[group["hydraulic_fabric_gate"].eq(1)]),
            ("outside_hydraulic_fabric_gate", group[group["hydraulic_fabric_gate"].eq(0)]),
        ]:
            if regime["observed_liquefaction"].nunique() < 2:
                continue
            metrics = binary_metrics(
                regime["predicted_pf"].to_numpy(dtype=float),
                regime["observed_liquefaction"].to_numpy(dtype=int),
            )
            model_gate_metrics.append(
                {
                    "model_name": model_name,
                    "regime": regime_name,
                    "n": int(len(regime)),
                    "n_liquefied": int(regime["observed_liquefaction"].sum()),
                    **metrics,
                }
            )
    model_gate_metrics_df = pd.DataFrame(model_gate_metrics)

    m2_wide = m2[["site_id", "event_key", "predicted_pf"]].rename(columns={"predicted_pf": "m2_predicted_pf"})
    contrast = m0.merge(m2_wide, on=["site_id", "event_key"], how="inner", validate="one_to_one")
    contrast["m2_minus_m0_pf"] = contrast["m2_predicted_pf"] - contrast["predicted_pf"]
    contrast_summary = (
        contrast.groupby("hydraulic_fabric_gate", as_index=False)
        .agg(
            n=("site_id", "size"),
            median_m2_minus_m0_pf=("m2_minus_m0_pf", "median"),
            mean_m2_minus_m0_pf=("m2_minus_m0_pf", "mean"),
            m0_false_negative_rate_0p5=("m0_false_negative_0p5", "mean"),
            observed_liquefaction_rate=("observed_liquefaction", "mean"),
        )
        .replace({"hydraulic_fabric_gate": {0: "outside", 1: "inside"}})
    )

    phys.to_csv(OUTPUTS / "hydraulic_leverage_event_states.csv", index=False)
    gate_summary_df.to_csv(OUTPUTS / "hydraulic_fabric_gate_summary.csv", index=False)
    decile.to_csv(OUTPUTS / "hydraulic_leverage_decile_response.csv", index=False)
    phys_metrics_df.to_csv(OUTPUTS / "hydraulic_leverage_physical_auc.csv", index=False)
    model_gate_metrics_df.to_csv(OUTPUTS / "hydraulic_fabric_gate_model_metrics.csv", index=False)
    contrast_summary.to_csv(OUTPUTS / "hydraulic_fabric_gate_model_contrast.csv", index=False)

    best_inside = (
        model_gate_metrics_df[model_gate_metrics_df["regime"].eq("inside_hydraulic_fabric_gate")]
        .sort_values(["auc", "brier_score"], ascending=[False, True])
        .head(1)
        .to_dict("records")
    )
    fn_auc = float(phys_metrics_df.loc[phys_metrics_df["target"].eq("m0_false_negative_0p5"), "auc_hydraulic_leverage"].iloc[0])
    liq_auc = float(phys_metrics_df.loc[phys_metrics_df["target"].eq("observed_liquefaction"), "auc_hydraulic_leverage"].iloc[0])
    status = "PASS_HYDRAULIC_LEVERAGE_PHYSICAL_REGIME_GATE" if fn_auc > 0.6 and liq_auc > 0.6 else "FAIL_FIELD_HYDRAULIC_LEVERAGE_NOT_VALIDATED"
    summary = {
        "status": status,
        "physical_mechanism": (
            "CSR scales with sigma_v/sigma'_v; transient groundwater shallowing lowers sigma'_v and creates a hydraulic leverage factor. "
            "The effect is physically consequential when paired with loose low-qc and fine-sensitive high-Ic CPT fabric."
        ),
        "hydraulic_leverage_definition": "max_z sigma'_v(z, site-median water table) / sigma'_v(z, event water table), z in 2,3,4,6 m",
        "fabric_gate_definition": "top quartile hydraulic leverage AND bottom quartile qc10_mpa AND top quartile Ic_median",
        "gate_summary": gate_summary,
        "physical_auc": phys_metrics_df.to_dict("records"),
        "model_contrast": contrast_summary.to_dict("records"),
        "best_inside_gate_model": best_inside,
        "claim_enabled": (
            "The field test can be reported as a falsification/guardrail: a simple hydraulic-leverage plus CPT-fabric gate is not sufficient, by itself, to explain Canterbury false negatives."
            if status.startswith("FAIL")
            else "The manuscript can claim a physically interpretable regime: shallow-water effective-stress leverage combined with loose/fine-sensitive CPT fabric identifies where stationary screening creates concentrated false-negative risk."
        ),
        "claim_blocked": (
            "Do not call this a new constitutive law, universal liquefaction triggering equation, or repair-cost validation. It is a field-tested physical regime/gate derived from effective stress and CPT fabric proxies."
        ),
    }
    (OUTPUTS / "hydraulic_leverage_breakthrough_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Hydraulic-leverage physical-regime gate - 2026-06-22",
        "",
        "## Mechanism",
        "",
        "For fixed PGA and depth, CSR is proportional to `sigma_v / sigma'_v`. A shallower water table lowers effective vertical stress and therefore amplifies CSR. The gate tests whether this hydraulic leverage becomes consequential when the CPT profile is also loose and fine-sensitive.",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        f"Gate definition: `{summary['fabric_gate_definition']}`.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
    ]
    (AUDIT / "hydraulic_leverage_physical_regime_gate_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
