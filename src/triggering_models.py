"""Triggering-model registry and lightweight site-proxy calculations.

The synthetic benchmark keeps its smooth CRR function separate from the
site-calibrated extension. The Nisqually extension is CPT based, so the
registered design procedures are documented with activation status rather than
being mixed with incompatible or unavailable inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TriggeringModel:
    model_name: str
    input_type: str
    output_type: str
    uncertainty_source: str
    model_error_distribution: str
    calibration_source: str
    used_in_final_site_application: bool
    activation_status: str


def model_registry() -> pd.DataFrame:
    models = [
        TriggeringModel(
            "Benchmark_smooth",
            "synthetic SPT-like N1,60cs",
            "deterministic FS converted to Pf by external error term",
            "benchmark epsm only",
            "lognormal multiplier in synthetic benchmark",
            "internal reproducibility benchmark",
            False,
            "retained for synthetic benchmark only",
        ),
        TriggeringModel(
            "BI14_SPT",
            "SPT N1,60cs",
            "deterministic FS",
            "procedure-specific corrections plus documented model uncertainty",
            "not applied here because SPT is unavailable",
            "Boulanger and Idriss SPT procedure",
            False,
            "not activated: Nisqually dataset is CPT based",
        ),
        TriggeringModel(
            "BI14_CPT",
            "CPT qc-derived resistance",
            "deterministic FS / probability proxy",
            "procedure-specific CPT corrections",
            "reported through cross-fitted site proxy in this extension",
            "Boulanger and Idriss CPT family",
            True,
            "activated as CPT-compatible site proxy and cross-fitted validation model",
        ),
        TriggeringModel(
            "Cetin_SPT_probabilistic",
            "SPT N1,60 and fines content",
            "direct probability",
            "built-in probabilistic triggering model",
            "not externally inflated",
            "Cetin SPT probabilistic model",
            False,
            "not activated: SPT and FC inputs are unavailable",
        ),
        TriggeringModel(
            "Moss_CPT_probabilistic",
            "CPT qc, fs and stress state",
            "direct probability",
            "built-in probabilistic CPT triggering model",
            "not externally inflated",
            "Moss CPT probabilistic model",
            False,
            "registered for future exact implementation",
        ),
        TriggeringModel(
            "Kayen_Vs_probabilistic",
            "Vs1 and stress state",
            "direct probability",
            "built-in probabilistic Vs triggering model",
            "not externally inflated",
            "Kayen Vs probabilistic model",
            False,
            "not activated: profile Vs is unavailable",
        ),
    ]
    return pd.DataFrame([m.__dict__ for m in models])


def rd_benchmark_depth_only(z_m: float) -> float:
    z = max(float(z_m), 0.0)
    return max(0.45, min(1.0, 1.0 - 0.00765 * z if z <= 9.15 else 1.174 - 0.0267 * z))


def csr_seed_1971(pga_g: float, sigma_v_kpa: float, sigma_v_eff_kpa: float, rd: float) -> float:
    return 0.65 * float(pga_g) * (float(sigma_v_kpa) / max(float(sigma_v_eff_kpa), 1.0)) * float(rd)


def crr_benchmark_smooth(n1cs: float) -> float:
    x = max(float(n1cs), 0.0)
    return 0.05 + 0.004 * x + 0.00025 * x * x


def beta_equivalent_from_pf(pf: float) -> float:
    p = min(max(float(pf), 1e-6), 1.0 - 1e-6)
    return -NormalDist().inv_cdf(p)
