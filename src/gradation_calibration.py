"""Gradation-status helpers for the site-calibrated extension.

The Nisqually PRJ-3758 CPT workbooks used here do not report repeated FC/D50
laboratory measurements. These helpers make that limitation explicit in the
CSV outputs instead of silently converting CPT proxies into measured gradation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_gradation_parameter_table(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in profile.iterrows():
        has_fc = pd.notna(row.get("fc_pct"))
        has_d50 = pd.notna(row.get("d50_mm"))
        status = "baseline_uncertainty_only" if not (has_fc and has_d50) else "calibrated_time_evolution"
        rows.append(
            {
                "site_id": row["site_id"],
                "layer_id": row["layer_id"],
                "fc0_mean": row.get("fc_pct", np.nan),
                "fc0_sd": np.nan,
                "fc_trend_pct_per_year": np.nan,
                "trend_sd": np.nan,
                "d50_mean": row.get("d50_mm", np.nan),
                "d50_sd": np.nan,
                "model_status": status,
                "proxy_available": "CPT Ic proxy" if pd.notna(row.get("ic")) else "none",
            }
        )
    return pd.DataFrame(rows)
