from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    predictions = pd.read_csv(OUTPUTS / "nisqually_protocol_exploration_predictions.csv")
    metrics = pd.read_csv(OUTPUTS / "site_validation_sodo_uw_sensitivity.csv")
    site = pd.read_csv(OUTPUTS / "site_calibrated_results.csv")
    site["site_family"] = site["site_name"].str.replace(r"-\d+$", "", regex=True)

    fixed = predictions[predictions["protocol"].eq("fixed_family_holdout_SODO_UW")].copy()
    fixed["predicted_label_0p5"] = (fixed["predicted_pf"] >= 0.5).astype(int)
    fixed["classification_error"] = (fixed["predicted_label_0p5"] != fixed["observed_liquefaction"]).astype(int)

    wide = fixed.pivot_table(
        index=["site_id", "site_name", "site_family", "observed_liquefaction"],
        columns="model_name",
        values="predicted_pf",
    ).reset_index()
    for col in ["M2_groundwater_gradation", "M3_random_field"]:
        if col in wide.columns and "M0_static_stationary" in wide.columns:
            wide[f"{col}_minus_M0"] = wide[col] - wide["M0_static_stationary"]
    features = site[
        [
            "site_id",
            "pga_g",
            "wtd_event_site_adjusted_m",
            "qc10_mpa",
            "qc25_mpa",
            "ic_median",
            "friction_ratio_median_pct",
            "theta_z_m",
            "variogram_rmse",
        ]
    ]
    diagnostic = wide.merge(features, on="site_id", how="left")
    diagnostic.to_csv(OUTPUTS / "sodo_uw_adverse_holdout_diagnostic.csv", index=False)

    best = metrics.sort_values("brier_score").iloc[0]
    worst_nonstationary = metrics[metrics["model_name"].str.contains("M2|M3", regex=True)].sort_values("brier_score").iloc[-1]
    false_negative_rows = fixed[
        fixed["classification_error"].eq(1) & fixed["observed_liquefaction"].eq(1)
    ][["model_name", "site_name", "predicted_pf", "observed_liquefaction"]]

    summary = {
        "status": "ADVERSE_HOLDOUT_GUARDRAIL_REQUIRED",
        "validation_protocol": "strict sensitivity holdout: train on 16 non-SODO/UW cases, test on SODO+UW siblings",
        "best_model_by_brier": best["model_name"],
        "best_brier": float(best["brier_score"]),
        "m2_brier": float(metrics.loc[metrics["model_name"].str.contains("M2"), "brier_score"].iloc[0]),
        "m3_brier": float(metrics.loc[metrics["model_name"].str.contains("M3"), "brier_score"].iloc[0]),
        "m2_false_negative_count": int(
            fixed[
                fixed["model_name"].eq("M2_groundwater_gradation")
                & fixed["classification_error"].eq(1)
                & fixed["observed_liquefaction"].eq(1)
            ].shape[0]
        ),
        "m3_false_negative_count": int(
            fixed[
                fixed["model_name"].eq("M3_random_field")
                & fixed["classification_error"].eq(1)
                & fixed["observed_liquefaction"].eq(1)
            ].shape[0]
        ),
        "false_negative_sites": sorted(false_negative_rows["site_name"].unique().tolist()),
        "interpretation": (
            "The non-stationary gradation/random-field terms improve pooled and several out-of-sample protocols, "
            "but they are not universally conservative under the SODO/UW industrial-waterway holdout. "
            "For this cluster, M0/M1 should be reported as the conservative guardrail and M2/M3 as bounded alternatives."
        ),
        "claim_enabled": "cluster-aware non-stationary reliability updating with explicit adverse-holdout guardrail",
        "claim_blocked": "universal superiority or universal site-validation claim for M2/M3",
    }
    (OUTPUTS / "sodo_uw_adverse_holdout_guardrail_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
