"""Layer stress calculations for site-calibrated liquefaction checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

GAMMA_W = 9.81


def compute_total_vertical_stress(layers: pd.DataFrame, z: float) -> float:
    sigma = 0.0
    for _, row in layers.sort_values("z_top_m").iterrows():
        top = float(row["z_top_m"])
        bot = float(row["z_bot_m"])
        if z <= top:
            continue
        dz = min(z, bot) - top
        if dz > 0:
            sigma += float(row.get("gamma_total_kN_m3", 18.0)) * dz
    return sigma


def compute_pore_pressure(z: float, gw_depth_t: float) -> float:
    return GAMMA_W * max(float(z) - float(gw_depth_t), 0.0)


def compute_effective_vertical_stress(layers: pd.DataFrame, z: float, gw_depth_t: float) -> float:
    return max(compute_total_vertical_stress(layers, z) - compute_pore_pressure(z, gw_depth_t), 5.0)

