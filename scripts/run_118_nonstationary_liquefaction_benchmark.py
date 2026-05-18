"""Reproducible benchmark for non-stationary liquefaction probability.

The script creates a synthetic but transparent layered soil profile and compares
deterministic, stationary-probabilistic, and non-stationary probabilistic
liquefaction assessments under evolving groundwater and gradation scenarios.
It is intentionally self-contained: only numpy and pandas are required.
It requires numpy, pandas, Pillow, and openpyxl.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from liquepy.trigger import boulanger_and_idriss_2014 as LIQUEPY_BI14
except Exception:  # pragma: no cover - optional dependency fallback for editorial review
    LIQUEPY_BI14 = None


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
DATA.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(1182026)
N_MC = 12000
YEARS = np.arange(0, 51, 2)
GAMMA_TOTAL = 18.2
GAMMA_W = 9.81
MAGNITUDE_SCALING = 1.12
STD_NORMAL = NormalDist()


LAYERS = pd.DataFrame(
    [
        {"layer": "L1", "z_mid_m": 2.0, "thickness_m": 2.0, "n160": 10.0, "fc0_pct": 8.0, "d50_mm": 0.33},
        {"layer": "L2", "z_mid_m": 4.5, "thickness_m": 3.0, "n160": 13.0, "fc0_pct": 14.0, "d50_mm": 0.25},
        {"layer": "L3", "z_mid_m": 8.0, "thickness_m": 4.0, "n160": 18.0, "fc0_pct": 21.0, "d50_mm": 0.18},
        {"layer": "L4", "z_mid_m": 12.5, "thickness_m": 5.0, "n160": 24.0, "fc0_pct": 28.0, "d50_mm": 0.14},
    ]
)

SCENARIOS = [
    {"scenario": "stationary", "trend_m_per_yr": 0.0, "season_amp_m": 0.0, "extreme_drop_m": 0.0},
    {"scenario": "rising", "trend_m_per_yr": -0.035, "season_amp_m": 0.0, "extreme_drop_m": 0.0},
    {"scenario": "seasonal", "trend_m_per_yr": -0.010, "season_amp_m": 0.45, "extreme_drop_m": 0.0},
    {"scenario": "extreme", "trend_m_per_yr": -0.025, "season_amp_m": 0.30, "extreme_drop_m": -1.20},
]

GRADATIONS = [
    {"gradation": "constant", "fc_rate_pct_per_yr": 0.00},
    {"gradation": "fines_accumulation", "fc_rate_pct_per_yr": 0.10},
    {"gradation": "fines_washout", "fc_rate_pct_per_yr": -0.08},
]


def rd_youd_2001(z: np.ndarray | float) -> np.ndarray | float:
    """Stress reduction factor approximation for shallow depths."""
    z_arr = np.asarray(z, dtype=float)
    rd = np.where(z_arr <= 9.15, 1.0 - 0.00765 * z_arr, 1.174 - 0.0267 * z_arr)
    return np.clip(rd, 0.55, 1.0)


def groundwater_depth(year: float, sc: dict) -> float:
    base = 3.2
    seasonal = sc["season_amp_m"] * math.sin(2.0 * math.pi * year / 10.0)
    event = sc["extreme_drop_m"] if year >= 30 else 0.0
    return max(0.6, base + sc["trend_m_per_yr"] * year + seasonal + event)


def fines_content(fc0: float, year: float, grad: dict) -> float:
    return float(np.clip(fc0 + grad["fc_rate_pct_per_yr"] * year, 0.0, 45.0))


def clean_sand_equivalent(n160: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Clean-sand equivalent SPT resistance using a smooth fines correction."""
    n160 = np.asarray(n160, dtype=float)
    fc = np.asarray(fc, dtype=float)
    alpha = np.where(fc <= 5, 0.0, np.where(fc < 35, np.exp(1.76 - 190.0 / (fc**2 + 1e-6)), 5.0))
    beta = np.where(fc <= 5, 1.0, np.where(fc < 35, 0.99 + (fc**1.5) / 1000.0, 1.20))
    out = alpha + beta * n160
    return np.where(np.isnan(n160) | np.isnan(fc), np.nan, out)


def crr_from_n1_60cs(n1cs: np.ndarray) -> np.ndarray:
    """Simplified CRR curve fitted for benchmark use, bounded to practical values."""
    x = np.clip(n1cs, 2.0, 32.0)
    crr = 0.048 + 0.0067 * x + 0.00032 * x**2
    return np.clip(crr, 0.05, 0.55)


def crr_bi14_spt_style(n1cs: np.ndarray | float) -> np.ndarray | float:
    """BI14/NCEER-style clean-sand SPT CRR curve for model-form comparison.

    The comparison is a benchmark diagnostic, not a site-specific design check.
    It uses the same clean-sand equivalent resistance, effective stress state,
    and demand terms as the benchmark so that only the resistance module changes.
    """
    x = np.clip(np.asarray(n1cs, dtype=float), 1.0, 37.0)
    crr = np.exp((x / 14.1) + (x / 126.0) ** 2 - (x / 23.6) ** 3 + (x / 25.4) ** 4 - 2.8)
    return np.clip(crr, 0.03, 0.80)


def crr_bi14_spt_liquepy(n1cs: np.ndarray | float, c_0: float = 2.8) -> np.ndarray | float:
    """BI14 SPT CRR from liquepy when available, with local equation fallback."""
    if LIQUEPY_BI14 is None:
        return crr_bi14_spt_style(n1cs)
    x = np.clip(np.asarray(n1cs, dtype=float), 1.0, 37.0)
    crr = LIQUEPY_BI14.calc_crr_m7p5_from_n1_60cs(x, c_0=c_0)
    return np.clip(np.asarray(crr, dtype=float), 0.03, 0.80)


def csr(pga_g: np.ndarray, z: float, gw_depth: np.ndarray) -> np.ndarray:
    sigma_v0 = GAMMA_TOTAL * z
    below = np.maximum(z - gw_depth, 0.0)
    u = GAMMA_W * below
    sigma_eff = np.maximum(sigma_v0 - u, 8.0)
    return 0.65 * pga_g * (sigma_v0 / sigma_eff) * rd_youd_2001(z)


def pf_from_samples(layer: pd.Series, year: float, sc: dict, grad: dict, mode: str) -> dict:
    gw_mean = groundwater_depth(year if mode == "nonstationary" else 0.0, sc)
    fc_mean = fines_content(layer.fc0_pct, year if mode == "nonstationary" else 0.0, grad)

    n160 = RNG.lognormal(mean=math.log(layer.n160), sigma=0.18, size=N_MC)
    fc = np.clip(RNG.normal(fc_mean, 3.0, size=N_MC), 0.0, 45.0)
    gw = np.clip(RNG.normal(gw_mean, 0.35, size=N_MC), 0.4, 7.0)
    pga = np.clip(RNG.lognormal(mean=math.log(0.28), sigma=0.30, size=N_MC), 0.05, 0.85)
    model_error = RNG.normal(1.0, 0.12, size=N_MC)

    n1cs = clean_sand_equivalent(n160, fc)
    resistance = crr_from_n1_60cs(n1cs) / MAGNITUDE_SCALING
    demand = csr(pga, layer.z_mid_m, gw)
    fs = resistance / np.maximum(demand * model_error, 1e-6)
    pf = float(np.mean(fs < 1.0))
    beta = -float(np.quantile(fs - 1.0, 0.50)) / max(float(np.std(fs - 1.0)), 1e-6)
    pf_clip = min(max(pf, 1e-6), 1.0 - 1e-6)
    beta_equivalent = -STD_NORMAL.inv_cdf(pf_clip)

    det_n1cs = clean_sand_equivalent(np.array([layer.n160]), np.array([fc_mean]))[0]
    det_crr = crr_from_n1_60cs(np.array([det_n1cs]))[0] / MAGNITUDE_SCALING
    det_csr = csr(np.array([0.28]), layer.z_mid_m, np.array([gw_mean]))[0]
    fs_det = float(det_crr / det_csr)

    return {
        "pf": pf,
        "beta_proxy": beta,
        "beta_equivalent": beta_equivalent,
        "fs_deterministic": fs_det,
        "gw_depth_m": gw_mean,
        "fc_pct": fc_mean,
        "n1_60cs": float(det_n1cs),
        "crr": float(det_crr),
        "csr": float(det_csr),
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stationary_cache = {}
    for sc in SCENARIOS:
        for grad in GRADATIONS:
            for _, layer in LAYERS.iterrows():
                cache_key = (sc["scenario"], grad["gradation"], layer.layer)
                stationary_cache[cache_key] = pf_from_samples(layer, 0.0, sc, grad, "stationary")
                for year in YEARS:
                    stationary = stationary_cache[cache_key]
                    nonstationary = pf_from_samples(layer, year, sc, grad, "nonstationary")
                    se_stat = math.sqrt(max(stationary["pf"] * (1.0 - stationary["pf"]), 0.0) / N_MC)
                    se_non = math.sqrt(max(nonstationary["pf"] * (1.0 - nonstationary["pf"]), 0.0) / N_MC)
                    rows.append(
                        {
                            "scenario": sc["scenario"],
                            "gradation": grad["gradation"],
                            "layer": layer.layer,
                            "z_mid_m": layer.z_mid_m,
                            "year": year,
                            "pf_stationary": stationary["pf"],
                            "pf_stationary_ci_low": max(0.0, stationary["pf"] - 1.96 * se_stat),
                            "pf_stationary_ci_high": min(1.0, stationary["pf"] + 1.96 * se_stat),
                            "pf_nonstationary": nonstationary["pf"],
                            "pf_nonstationary_ci_low": max(0.0, nonstationary["pf"] - 1.96 * se_non),
                            "pf_nonstationary_ci_high": min(1.0, nonstationary["pf"] + 1.96 * se_non),
                            "delta_pf": nonstationary["pf"] - stationary["pf"],
                            "fs_deterministic_nonstationary": nonstationary["fs_deterministic"],
                            "gw_depth_m": nonstationary["gw_depth_m"],
                            "fc_pct": nonstationary["fc_pct"],
                            "n1_60cs": nonstationary["n1_60cs"],
                            "crr": nonstationary["crr"],
                            "csr": nonstationary["csr"],
                            "beta_proxy": nonstationary["beta_proxy"],
                            "beta_equivalent": nonstationary["beta_equivalent"],
                        }
                    )
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["scenario", "gradation"], as_index=False)
        .agg(
            max_pf_nonstationary=("pf_nonstationary", "max"),
            mean_pf_nonstationary=("pf_nonstationary", "mean"),
            max_delta_pf=("delta_pf", "max"),
            final_mean_pf=("pf_nonstationary", lambda s: float(s[results.loc[s.index, "year"].eq(YEARS[-1])].mean())),
            min_fs_deterministic=("fs_deterministic_nonstationary", "min"),
        )
        .sort_values(["max_pf_nonstationary", "max_delta_pf"], ascending=False)
    )
    return results, summary


def make_standard_model_comparison(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the benchmark resistance module with a standard SPT triggering curve.

    This is a model-form sensitivity check using the same synthetic profile states.
    It is not field validation and does not use CPT/Vs models without independent
    CPT/Vs measurements.
    """
    comp = results.copy()
    comp["pga_g"] = 0.28
    comp["n160"] = comp["layer"].map(dict(zip(LAYERS.layer, LAYERS.n160)))
    comp["crr_benchmark_module"] = comp["crr"]
    comp["crr_bi14_spt_style"] = crr_bi14_spt_style(comp["n1_60cs"].to_numpy()) / MAGNITUDE_SCALING
    comp["fs_bi14_spt_style"] = comp["crr_bi14_spt_style"] / comp["csr"].clip(lower=1e-6)
    comp["crr_bi14_liquepy_m7p5"] = crr_bi14_spt_liquepy(comp["n1_60cs"].to_numpy())
    comp["crr_bi14_liquepy"] = comp["crr_bi14_liquepy_m7p5"] / MAGNITUDE_SCALING
    comp["fs_bi14_liquepy"] = comp["crr_bi14_liquepy"] / comp["csr"].clip(lower=1e-6)
    comp["crr_bi14_liquepy_median_m7p5"] = crr_bi14_spt_liquepy(comp["n1_60cs"].to_numpy(), c_0=2.6)
    comp["crr_bi14_liquepy_median"] = comp["crr_bi14_liquepy_median_m7p5"] / MAGNITUDE_SCALING
    comp["fs_bi14_liquepy_median"] = comp["crr_bi14_liquepy_median"] / comp["csr"].clip(lower=1e-6)
    comp["delta_fs_bi14_minus_benchmark"] = comp["fs_bi14_spt_style"] - comp["fs_deterministic_nonstationary"]
    comp["delta_fs_liquepy_bi14_minus_benchmark"] = comp["fs_bi14_liquepy"] - comp["fs_deterministic_nonstationary"]
    comp["delta_fs_liquepy_bi14_median_minus_benchmark"] = (
        comp["fs_bi14_liquepy_median"] - comp["fs_deterministic_nonstationary"]
    )
    comp["fs_scale_ratio_liquepy_bi14_to_benchmark"] = (
        comp["fs_bi14_liquepy"] / comp["fs_deterministic_nonstationary"].clip(lower=1e-6)
    )
    comp["fs_scale_ratio_liquepy_bi14_median_to_benchmark"] = (
        comp["fs_bi14_liquepy_median"] / comp["fs_deterministic_nonstationary"].clip(lower=1e-6)
    )
    sigma_ln_fs = math.sqrt(0.25**2 + 0.12**2)
    comp["pf_bi14_spt_style_diagnostic"] = comp["fs_bi14_spt_style"].clip(lower=1e-6).map(
        lambda fs: STD_NORMAL.cdf(-math.log(float(fs)) / sigma_ln_fs)
    )
    comp["pf_bi14_liquepy_diagnostic"] = comp["fs_bi14_liquepy"].clip(lower=1e-6).map(
        lambda fs: STD_NORMAL.cdf(-math.log(float(fs)) / sigma_ln_fs)
    )
    comp["pf_bi14_liquepy_median_diagnostic"] = comp["fs_bi14_liquepy_median"].clip(lower=1e-6).map(
        lambda fs: STD_NORMAL.cdf(-math.log(float(fs)) / sigma_ln_fs)
    )
    comp["delta_pf_bi14_minus_benchmark"] = comp["pf_bi14_spt_style_diagnostic"] - comp["pf_nonstationary"]
    comp["delta_pf_liquepy_bi14_minus_benchmark"] = comp["pf_bi14_liquepy_diagnostic"] - comp["pf_nonstationary"]
    comp["delta_pf_liquepy_bi14_median_minus_benchmark"] = (
        comp["pf_bi14_liquepy_median_diagnostic"] - comp["pf_nonstationary"]
    )
    comp["method_status"] = "direct_spt_model_form_check_with_liquepy_bi14_when_available"
    keep = [
        "scenario",
        "gradation",
        "year",
        "layer",
        "z_mid_m",
        "gw_depth_m",
        "fc_pct",
        "pga_g",
        "n160",
        "n1_60cs",
        "csr",
        "pf_nonstationary",
        "fs_deterministic_nonstationary",
        "crr_benchmark_module",
        "crr_bi14_spt_style",
        "fs_bi14_spt_style",
        "crr_bi14_liquepy_m7p5",
        "crr_bi14_liquepy",
        "fs_bi14_liquepy",
        "crr_bi14_liquepy_median_m7p5",
        "crr_bi14_liquepy_median",
        "fs_bi14_liquepy_median",
        "delta_fs_bi14_minus_benchmark",
        "delta_fs_liquepy_bi14_minus_benchmark",
        "delta_fs_liquepy_bi14_median_minus_benchmark",
        "fs_scale_ratio_liquepy_bi14_to_benchmark",
        "fs_scale_ratio_liquepy_bi14_median_to_benchmark",
        "pf_bi14_spt_style_diagnostic",
        "pf_bi14_liquepy_diagnostic",
        "pf_bi14_liquepy_median_diagnostic",
        "delta_pf_bi14_minus_benchmark",
        "delta_pf_liquepy_bi14_minus_benchmark",
        "delta_pf_liquepy_bi14_median_minus_benchmark",
        "method_status",
    ]
    comp = comp[keep]
    summary_rows = []
    for scenario, gradation in comp[["scenario", "gradation"]].drop_duplicates().itertuples(index=False):
        sub = comp[(comp.scenario == scenario) & (comp.gradation == gradation)]
        same_highest = []
        for _, g in sub.groupby("year"):
            same_highest.append(
                g.loc[g["pf_nonstationary"].idxmax(), "layer"]
                == g.loc[g["pf_bi14_liquepy_diagnostic"].idxmax(), "layer"]
            )
        summary_rows.append(
            {
                "scenario": scenario,
                "gradation": gradation,
                "n_records": int(sub.shape[0]),
                "spearman_pf_current_vs_bi14_style": float(
                    sub["pf_nonstationary"].rank(method="average").corr(
                        sub["pf_bi14_spt_style_diagnostic"].rank(method="average"), method="pearson"
                    )
                ),
                "spearman_pf_current_vs_bi14_liquepy": float(
                    sub["pf_nonstationary"].rank(method="average").corr(
                        sub["pf_bi14_liquepy_diagnostic"].rank(method="average"), method="pearson"
                    )
                ),
                "median_abs_delta_pf_style": float(sub["delta_pf_bi14_minus_benchmark"].abs().median()),
                "median_abs_delta_pf_liquepy": float(sub["delta_pf_liquepy_bi14_minus_benchmark"].abs().median()),
                "median_abs_delta_pf_liquepy_median": float(
                    sub["delta_pf_liquepy_bi14_median_minus_benchmark"].abs().median()
                ),
                "max_abs_delta_pf_style": float(sub["delta_pf_bi14_minus_benchmark"].abs().max()),
                "max_abs_delta_pf_liquepy": float(sub["delta_pf_liquepy_bi14_minus_benchmark"].abs().max()),
                "max_abs_delta_pf_liquepy_median": float(
                    sub["delta_pf_liquepy_bi14_median_minus_benchmark"].abs().max()
                ),
                "median_fs_scale_ratio_liquepy_to_benchmark": float(
                    sub["fs_scale_ratio_liquepy_bi14_to_benchmark"].median()
                ),
                "median_fs_scale_ratio_liquepy_median_to_benchmark": float(
                    sub["fs_scale_ratio_liquepy_bi14_median_to_benchmark"].median()
                ),
                "same_highest_layer_share": float(np.mean(same_highest)),
                "liquepy_bi14_available": bool(LIQUEPY_BI14 is not None),
                "bi16_exact_liquepy_available": False,
                "interpretation": "standard BI14 SPT resistance module ranking and scale check; not site validation",
            }
        )
    return comp, pd.DataFrame(summary_rows)


def make_vertical_dependence_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    """System-probability diagnostic under equicorrelated Gaussian copula layers."""
    rows = []
    rng = np.random.default_rng(1182027)
    n_sim = 50000
    rhos = [0.0, 0.3, 0.6, 0.9]
    for (scenario, gradation, year), group in results.groupby(["scenario", "gradation", "year"]):
        pfs = group.sort_values("layer")["pf_nonstationary"].clip(1e-6, 1.0 - 1e-6).to_numpy()
        frechet_lower = float(np.max(pfs))
        frechet_upper = float(min(1.0, np.sum(pfs)))
        independent = float(1.0 - np.prod(1.0 - pfs))
        thresholds = np.array([STD_NORMAL.inv_cdf(float(p)) for p in pfs])
        for rho in rhos:
            cov = np.full((len(pfs), len(pfs)), rho)
            np.fill_diagonal(cov, 1.0)
            z = rng.multivariate_normal(np.zeros(len(pfs)), cov, size=n_sim)
            psys = float(np.mean((z <= thresholds).any(axis=1)))
            rows.append(
                {
                    "scenario": scenario,
                    "gradation": gradation,
                    "year": int(year),
                    "rho_equicorrelation": rho,
                    "psys_gaussian_copula": psys,
                    "psys_independent": independent,
                    "frechet_lower": frechet_lower,
                    "frechet_upper": frechet_upper,
                    "n_sim": n_sim,
                    "status": "vertical-dependence diagnostic; not calibrated random field",
                }
            )
    return pd.DataFrame(rows)


def make_model_availability_diagnostics() -> pd.DataFrame:
    """Document which established triggering comparisons are executed or constrained by inputs."""
    rows = [
        {
            "model_or_family": "Boulanger and Idriss 2014 SPT",
            "implementation_status": "executed",
            "implementation_detail": "liquepy.trigger.boulanger_and_idriss_2014.calc_crr_m7p5_from_n1_60cs; local equation fallback retained",
            "required_inputs": "N1,60cs, CSR/state demand, magnitude scaling",
            "available_in_synthetic_benchmark": "yes",
            "available_in_hu_field_cases": "proxy via N120, FC, CSR7.5",
            "output_files": "standard_model_comparison.csv; standard_model_comparison_summary.csv; field_validation_metrics.csv",
            "interpretation": "formal model-form and field transferability comparison",
        },
        {
            "model_or_family": "Boulanger and Idriss CPT",
            "implementation_status": "not executed in current package",
            "implementation_detail": "liquepy callable exists, but the manuscript supplement and Hu dataset do not contain qc, fs, u2 CPT profiles",
            "required_inputs": "qc, fs, u2, depth profile, groundwater, PGA, Mw",
            "available_in_synthetic_benchmark": "no CPT profile",
            "available_in_hu_field_cases": "no CPT profile",
            "output_files": "none",
            "interpretation": "candidate extension if DesignSafe CPT case-history files are added",
        },
        {
            "model_or_family": "Boulanger and Idriss 2016 exact",
            "implementation_status": "documented constraint",
            "implementation_detail": "no distinct BI16 callable was present in liquepy 0.6.34; not substituted by a renamed BI14 curve",
            "required_inputs": "published BI16 implementation plus SPT/CPT variables required by that implementation",
            "available_in_synthetic_benchmark": "partial SPT state only",
            "available_in_hu_field_cases": "partial SPT/Vs state only",
            "output_files": "model_availability_diagnostics.csv; standard_model_comparison_summary.csv flag",
            "interpretation": "excluded to avoid a false exact-model claim",
        },
        {
            "model_or_family": "Cetin SPT probabilistic triggering",
            "implementation_status": "documented constraint",
            "implementation_detail": "not silently approximated because exact implementation/calibration coefficients are not bundled in the supplement",
            "required_inputs": "SPT resistance, fines/plasticity terms, stress terms, magnitude/demand terms and calibrated coefficients",
            "available_in_synthetic_benchmark": "partial",
            "available_in_hu_field_cases": "partial",
            "output_files": "model_availability_diagnostics.csv",
            "interpretation": "do not substitute with an undocumented equation in a reproducibility benchmark",
        },
        {
            "model_or_family": "Moss CPT triggering",
            "implementation_status": "documented constraint",
            "implementation_detail": "not executed because neither the synthetic benchmark nor Hu cases include qc/fs CPT records",
            "required_inputs": "CPT qc, fs/friction ratio, vertical stress, groundwater, PGA, Mw",
            "available_in_synthetic_benchmark": "no CPT profile",
            "available_in_hu_field_cases": "no CPT profile",
            "output_files": "model_availability_diagnostics.csv",
            "interpretation": "requires external CPT dataset before defensible execution",
        },
        {
            "model_or_family": "Kayen/Vs-style screening",
            "implementation_status": "executed as field predictor",
            "implementation_detail": "Hu et al. Vs1 values used as an independent resistance-screening score with AUC/Brier/confusion diagnostics",
            "required_inputs": "Vs1, demand term, groundwater/capping descriptors, observed outcome",
            "available_in_synthetic_benchmark": "no Vs profile",
            "available_in_hu_field_cases": "yes",
            "output_files": "field_validation_cases.csv; field_validation_metrics.csv",
            "interpretation": "formal Vs-based field discrimination check, not a full Kayen equation calibration",
        },
    ]
    return pd.DataFrame(rows)


def svg_line(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray]], ylabel: str) -> None:
    width, height = 900, 560
    ml, mr, mt, mb = 90, 30, 70, 70
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = 0.0, max(float(ys.max()) * 1.12, 0.05)

    def sx(x): return ml + (x - xmin) / (xmax - xmin) * (width - ml - mr)
    def sy(y): return height - mb - (y - ymin) / (ymax - ymin) * (height - mt - mb)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Time horizon (years)</text>',
        f'<text x="24" y="{height/2}" transform="rotate(-90 24,{height/2})" text-anchor="middle" font-family="Arial" font-size="14">{ylabel}</text>',
    ]
    for tick in np.linspace(xmin, xmax, 6):
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{height-mb}" x2="{x:.1f}" y2="{height-mb+6}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-mb+24}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.0f}</text>')
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(tick)
        parts.append(f'<line x1="{ml-6}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{ml-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for i, (label, xdata, ydata) in enumerate(series):
        pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xdata, ydata))
        color = colors[i % len(colors)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<rect x="{width-260}" y="{70+i*24}" width="16" height="3" fill="{color}"/>')
        parts.append(f'<text x="{width-238}" y="{76+i*24}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, data: pd.DataFrame, title: str) -> None:
    width, height = 820, 560
    ml, mr, mt, mb = 100, 110, 70, 70
    years = sorted(data.year.unique())
    layers = list(LAYERS.layer)
    val = {(r.layer, r.year): r.pf_nonstationary for r in data.itertuples()}
    maxv = max(val.values()) if val else 1.0
    cw = (width - ml - mr) / len(years)
    ch = (height - mt - mb) / len(layers)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for i, layer in enumerate(layers):
        y = mt + i * ch
        parts.append(f'<text x="{ml-14}" y="{y+ch/2+5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{layer}</text>')
        for j, year in enumerate(years):
            x = ml + j * cw
            v = val.get((layer, year), 0.0)
            r = int(245 - 70 * (v / maxv))
            g = int(248 - 190 * (v / maxv))
            b = int(255 - 220 * (v / maxv))
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw+0.5:.1f}" height="{ch+0.5:.1f}" fill="rgb({r},{g},{b})" stroke="white"/>')
    for j, year in enumerate(years[::5]):
        x = ml + years.index(year) * cw + cw / 2
        parts.append(f'<text x="{x:.1f}" y="{height-mb+22}" text-anchor="middle" font-family="Arial" font-size="12">{year}</text>')
    parts.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Time horizon (years)</text>')
    parts.append(f'<text x="24" y="{height/2}" transform="rotate(-90 24,{height/2})" text-anchor="middle" font-family="Arial" font-size="14">Layer</text>')
    for k in range(6):
        v = k / 5 * maxv
        y = mt + (5-k) * 50
        r = int(245 - 70 * (v / maxv))
        g = int(248 - 190 * (v / maxv))
        b = int(255 - 220 * (v / maxv))
        parts.append(f'<rect x="{width-mr+35}" y="{y}" width="28" height="50" fill="rgb({r},{g},{b})"/>')
        parts.append(f'<text x="{width-mr+70}" y="{y+30}" font-family="Arial" font-size="12">{v:.2f}</text>')
    parts.append(f'<text x="{width-mr+35}" y="{mt-14}" font-family="Arial" font-size="12">Pf</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def png_line(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray]], ylabel: str) -> None:
    width, height = 1400, 850
    ml, mr, mt, mb = 140, 70, 110, 110
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, tick_font = _font(30, True), _font(22), _font(18)
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = 0.0, max(float(ys.max()) * 1.12, 0.05)

    def sx(x): return ml + (x - xmin) / (xmax - xmin) * (width - ml - mr)
    def sy(y): return height - mb - (y - ymin) / (ymax - ymin) * (height - mt - mb)

    draw.line((ml, mt, ml, height - mb, width - mr, height - mb), fill=(40, 40, 40), width=2)
    for tick in np.linspace(xmin, xmax, 6):
        x = sx(tick)
        draw.line((x, height - mb, x, height - mb + 10), fill=(40, 40, 40), width=2)
        draw.text((x, height - mb + 32), f"{tick:.0f}", anchor="mm", font=tick_font, fill=(40, 40, 40))
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(tick)
        draw.line((ml - 10, y, ml, y), fill=(40, 40, 40), width=2)
        draw.line((ml, y, width - mr, y), fill=(225, 225, 225), width=1)
        draw.text((ml - 18, y), f"{tick:.2f}", anchor="rm", font=tick_font, fill=(40, 40, 40))
    for i, (label, xdata, ydata) in enumerate(series):
        pts = [(sx(float(x)), sy(float(y))) for x, y in zip(xdata, ydata)]
        draw.line(pts, fill=colors[i % len(colors)], width=5)
        yleg = 120 + i * 34
        draw.line((width - 360, yleg, width - 320, yleg), fill=colors[i % len(colors)], width=6)
        draw.text((width - 308, yleg), label, anchor="lm", font=tick_font, fill=(20, 20, 20))
    draw.text((width / 2, height - 34), "Time horizon (years)", anchor="mm", font=label_font, fill=(0, 0, 0))
    rotated = Image.new("RGBA", (420, 50), (255, 255, 255, 0))
    rdraw = ImageDraw.Draw(rotated)
    rdraw.text((210, 25), ylabel, anchor="mm", font=label_font, fill=(0, 0, 0))
    img.paste(rotated.rotate(90, expand=True), (26, int(height / 2 - 210)), rotated.rotate(90, expand=True))
    img.save(path, "PNG")


def png_heatmap(path: Path, data: pd.DataFrame, title: str) -> None:
    width, height = 1300, 820
    ml, mr, mt, mb = 150, 170, 110, 110
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, tick_font = _font(30, True), _font(22), _font(18)
    years = sorted(data.year.unique())
    layers = list(LAYERS.layer)
    val = {(r.layer, r.year): r.pf_nonstationary for r in data.itertuples()}
    maxv = max(val.values()) if val else 1.0
    cw = (width - ml - mr) / len(years)
    ch = (height - mt - mb) / len(layers)
    for i, layer in enumerate(layers):
        y = mt + i * ch
        draw.text((ml - 18, y + ch / 2), layer, anchor="rm", font=label_font, fill=(20, 20, 20))
        for j, year in enumerate(years):
            x = ml + j * cw
            v = val.get((layer, year), 0.0)
            r = int(245 - 70 * (v / maxv))
            g = int(248 - 190 * (v / maxv))
            b = int(255 - 220 * (v / maxv))
            draw.rectangle((x, y, x + cw + 1, y + ch + 1), fill=(r, g, b), outline="white")
    for year in years[::5]:
        x = ml + years.index(year) * cw + cw / 2
        draw.text((x, height - mb + 34), str(year), anchor="mm", font=tick_font, fill=(40, 40, 40))
    draw.text((width / 2, height - 34), "Time horizon (years)", anchor="mm", font=label_font, fill=(0, 0, 0))
    for k in range(6):
        v = k / 5 * maxv
        y = mt + (5 - k) * 72
        r = int(245 - 70 * (v / maxv))
        g = int(248 - 190 * (v / maxv))
        b = int(255 - 220 * (v / maxv))
        draw.rectangle((width - mr + 45, y, width - mr + 85, y + 72), fill=(r, g, b))
        draw.text((width - mr + 100, y + 36), f"{v:.2f}", anchor="lm", font=tick_font, fill=(40, 40, 40))
    draw.text((width - mr + 45, mt - 24), "Pf", font=tick_font, fill=(40, 40, 40))
    img.save(path, "PNG")


def svg_bar(path: Path, title: str, labels: list[str], values: list[float], xlabel: str) -> None:
    width, height = 900, 560
    ml, mr, mt, mb = 230, 40, 70, 60
    vmax = max(values) * 1.15 if values else 1.0
    bar_h = (height - mt - mb) / max(len(labels), 1) * 0.68
    gap = (height - mt - mb) / max(len(labels), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#333"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = mt + i * gap + (gap - bar_h) / 2
        w = (value / vmax) * (width - ml - mr)
        parts.append(f'<text x="{ml-12}" y="{y+bar_h/2+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{label}</text>')
        parts.append(f'<rect x="{ml}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="#4c78a8"/>')
        parts.append(f'<text x="{ml+w+8:.1f}" y="{y+bar_h/2+5:.1f}" font-family="Arial" font-size="12">{value:.3f}</text>')
    parts.append(f'<text x="{(ml+width-rm if False else width/2)}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">{xlabel}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def png_bar(path: Path, title: str, labels: list[str], values: list[float], xlabel: str) -> None:
    width, height = 1400, 850
    ml, mr, mt, mb = 360, 90, 110, 90
    vmax = max(values) * 1.15 if values else 1.0
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, tick_font = _font(30, True), _font(21), _font(18)
    draw.line((ml, height - mb, width - mr, height - mb), fill=(40, 40, 40), width=2)
    gap = (height - mt - mb) / max(len(labels), 1)
    bar_h = gap * 0.62
    for i, (label, value) in enumerate(zip(labels, values)):
        y = mt + i * gap + (gap - bar_h) / 2
        w = (value / vmax) * (width - ml - mr)
        draw.text((ml - 16, y + bar_h / 2), label, anchor="rm", font=tick_font, fill=(40, 40, 40))
        draw.rectangle((ml, y, ml + w, y + bar_h), fill=(76, 120, 168))
        draw.text((ml + w + 12, y + bar_h / 2), f"{value:.3f}", anchor="lm", font=tick_font, fill=(40, 40, 40))
    draw.text((width / 2, height - 28), xlabel, anchor="mm", font=label_font, fill=(0, 0, 0))
    img.save(path, "PNG")


def _numeric_column(series: pd.Series) -> pd.Series:
    """Parse numeric columns that contain symbols such as *, <5, ~5, or footnote letters."""
    cleaned = (
        series.astype(str)
        .str.replace("～", "~", regex=False)
        .str.replace("*", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.replace("~", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    """AUC from average ranks, avoiding an extra machine-learning dependency."""
    valid = scores.notna() & labels.notna()
    scores = scores[valid]
    labels = labels[valid].astype(int)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scores.rank(method="average")
    rank_sum_pos = float(ranks[labels.eq(1)].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _fit_logistic_one_score(scores: pd.Series, labels: pd.Series) -> tuple[float, float]:
    """Small Newton solver for a one-score logistic calibration model."""
    valid = scores.notna() & labels.notna()
    x = scores[valid].astype(float).to_numpy()
    y = labels[valid].astype(float).to_numpy()
    x = (x - x.mean()) / max(float(x.std()), 1e-9)
    beta = np.array([0.0, 0.0])
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(50):
        eta = np.clip(X @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        grad = X.T @ (y - p)
        hess = -(X.T * w) @ X
        step = np.linalg.solve(hess - np.eye(2) * 1e-8, grad)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def _logistic_predict(scores_train: pd.Series, labels_train: pd.Series, scores_test: pd.Series) -> np.ndarray:
    intercept, slope = _fit_logistic_one_score(scores_train, labels_train)
    mu = float(scores_train.astype(float).mean())
    sd = max(float(scores_train.astype(float).std()), 1e-9)
    z = (scores_test.astype(float).to_numpy() - mu) / sd
    return 1.0 / (1.0 + np.exp(-(intercept + slope * z)))


def _cross_validated_probabilities(scores: pd.Series, labels: pd.Series, k: int = 5) -> pd.Series:
    valid = scores.notna() & labels.notna()
    out = pd.Series(np.nan, index=scores.index, dtype=float)
    idx = scores[valid].sort_values(kind="mergesort").index.to_list()
    for fold in range(k):
        test_idx = idx[fold::k]
        train_idx = [i for i in idx if i not in set(test_idx)]
        out.loc[test_idx] = _logistic_predict(scores.loc[train_idx], labels.loc[train_idx], scores.loc[test_idx])
    return out


def _confusion_from_threshold(scores: pd.Series, labels: pd.Series, threshold: float, greater_is_liq: bool = True) -> dict:
    valid = scores.notna() & labels.notna()
    s = scores[valid].astype(float)
    y = labels[valid].astype(int)
    pred = (s >= threshold).astype(int) if greater_is_liq else (s <= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((pred == y).mean()),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
    }


def _best_youden_threshold(scores: pd.Series, labels: pd.Series, greater_is_liq: bool = True) -> tuple[float, dict]:
    """Select the threshold that maximises sensitivity + specificity - 1."""
    valid = scores.notna() & labels.notna()
    s = scores[valid].astype(float)
    y = labels[valid].astype(int)
    if s.empty:
        return float("nan"), {}
    best_threshold = float(s.median())
    best_conf = _confusion_from_threshold(s, y, best_threshold, greater_is_liq)
    best_j = best_conf["sensitivity"] + best_conf["specificity"] - 1.0
    for threshold in np.unique(s.to_numpy()):
        conf = _confusion_from_threshold(s, y, float(threshold), greater_is_liq)
        j = conf["sensitivity"] + conf["specificity"] - 1.0
        if j > best_j:
            best_threshold = float(threshold)
            best_conf = conf
            best_j = j
    best_conf["youden_j"] = float(best_j)
    return best_threshold, best_conf


def external_calibration_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """In-sample external discrimination/calibration diagnostics for Hu et al. scores."""
    rows = []
    bins = []
    for score in ["combined_instrument_score", "dpt_screening_score", "vs_screening_score"]:
        valid = df.dropna(subset=[score, "liquefied"]).copy()
        prob = _cross_validated_probabilities(valid[score], valid["liquefied"]).loc[valid.index].to_numpy()
        valid["calibrated_probability"] = prob
        pred = (prob >= 0.5).astype(int)
        y = valid["liquefied"].astype(int).to_numpy()
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rows.append(
            {
                "score": score,
                "n_cases": int(valid.shape[0]),
                "auc": float(_rank_auc(valid[score], valid["liquefied"])),
                "brier_score_5fold": float(np.mean((prob - y) ** 2)),
                "threshold": 0.5,
                "accuracy": float((pred == y).mean()),
                "sensitivity": float(tp / max(tp + fn, 1)),
                "specificity": float(tn / max(tn + fp, 1)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "status": "5-fold calibration sanity check; not field validation",
            }
        )
        valid["bin"] = pd.qcut(valid["calibrated_probability"], q=5, duplicates="drop")
        for interval, b in valid.groupby("bin", observed=False):
            bins.append(
                {
                    "score": score,
                    "probability_bin": str(interval),
                    "n_cases": int(b.shape[0]),
                    "mean_predicted_probability": float(b["calibrated_probability"].mean()),
                    "observed_liquefaction_rate": float(b["liquefied"].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bins)


def external_static_limit_state_proxy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a static proxy of the benchmark limit-state to Hu et al. cases."""
    valid = df.dropna(subset=["liquefied", "csr75", "n120", "fc_pct"]).copy()
    valid["n120_cs"] = clean_sand_equivalent(valid["n120"].to_numpy(), valid["fc_pct"].to_numpy())
    valid["crr_static_benchmark"] = crr_from_n1_60cs(valid["n120_cs"].to_numpy()) / MAGNITUDE_SCALING
    valid["fs_static_proxy"] = valid["crr_static_benchmark"] / valid["csr75"].clip(lower=1e-6)
    valid["g_static_proxy"] = valid["fs_static_proxy"] - 1.0
    valid["minus_g_static_proxy"] = -valid["g_static_proxy"]
    sigma_ln_fs = math.sqrt(0.25**2 + 0.12**2)
    valid["pf_static_proxy"] = valid["fs_static_proxy"].clip(lower=1e-6).map(
        lambda fs: STD_NORMAL.cdf(-math.log(float(fs)) / sigma_ln_fs)
    )
    metrics = []
    for score, threshold, greater in [
        ("minus_g_static_proxy", 0.0, True),
        ("pf_static_proxy", 0.5, True),
    ]:
        prob_cv = _cross_validated_probabilities(valid[score], valid["liquefied"]).loc[valid.index]
        conf = _confusion_from_threshold(valid[score], valid["liquefied"], threshold, greater)
        metrics.append(
            {
                "score": score,
                "n_cases": int(valid.shape[0]),
                "auc": float(_rank_auc(valid[score], valid["liquefied"])),
                "brier_score_5fold": float(np.mean((prob_cv.to_numpy() - valid["liquefied"].astype(int).to_numpy()) ** 2)),
                "raw_threshold": threshold,
                **conf,
                "status": "static limit-state compatibility check; not non-stationary validation",
            }
        )
    return valid, pd.DataFrame(metrics)


def field_case_history_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build field-data validation tables from Hu et al. cases and the Wenchuan subset.

    The full Hu table is a multi-earthquake case-history dataset. The Wenchuan
    subset is treated as a focal documented earthquake subset with recorded
    groundwater depth, shaking demand, resistance proxies, and observed outcome.
    """
    cases = df.copy()
    cases["case_id"] = cases["case_no"].map(lambda x: f"HU-{int(x):03d}" if pd.notna(x) else "")
    cases["source_dataset"] = "Hu et al. (2021) gravelly-soil case histories"
    cases["validation_split"] = np.where(
        cases["earthquake"].astype(str).str.contains("Wenchuan", case=False, na=False),
        "Wenchuan focal field subset",
        "Hu multi-earthquake external set",
    )
    cases["groundwater_table_m"] = cases["dw_m"]
    cases["critical_layer_depth_m"] = cases["ds_m"]
    cases["unsaturated_cap_or_cover_m"] = cases["dn_m"]
    cases["observed_liquefaction"] = cases["liquefied"].astype(int)

    cases["n1_60cs_proxy"] = clean_sand_equivalent(cases["n120"].to_numpy(), cases["fc_pct"].to_numpy())
    cases["crr_static_benchmark"] = crr_from_n1_60cs(cases["n1_60cs_proxy"].to_numpy()) / MAGNITUDE_SCALING
    cases["fs_static_proxy"] = cases["crr_static_benchmark"] / cases["csr75"].clip(lower=1e-6)
    cases["g_static_proxy"] = cases["fs_static_proxy"] - 1.0
    cases["minus_g_static_proxy"] = -cases["g_static_proxy"]
    sigma_ln_fs = math.sqrt(0.25**2 + 0.12**2)
    cases["pf_static_proxy"] = cases["fs_static_proxy"].clip(lower=1e-6).map(
        lambda fs: STD_NORMAL.cdf(-math.log(float(fs)) / sigma_ln_fs)
    )

    metrics = []
    thresholds = []
    groups = [
        ("Hu full field set", cases),
        ("Wenchuan documented earthquake subset", cases[cases["validation_split"].eq("Wenchuan focal field subset")]),
    ]
    predictors = [
        ("combined demand-resistance-groundwater score", "combined_instrument_score", 0.0, True),
        ("DPT/N120 score", "dpt_screening_score", 0.0, True),
        ("Vs score", "vs_screening_score", 0.0, True),
        ("static limit-state proxy", "minus_g_static_proxy", 0.0, True),
        ("static Pf proxy", "pf_static_proxy", 0.5, True),
    ]
    for group_name, group in groups:
        for predictor_name, score, default_threshold, greater in predictors:
            valid = group.dropna(subset=[score, "liquefied"]).copy()
            if valid.empty:
                continue
            prob_cv = _cross_validated_probabilities(valid[score], valid["liquefied"]).loc[valid.index]
            default_conf = _confusion_from_threshold(valid[score], valid["liquefied"], default_threshold, greater)
            best_threshold, best_conf = _best_youden_threshold(valid[score], valid["liquefied"], greater)
            base = {
                "validation_group": group_name,
                "predictor": predictor_name,
                "score_column": score,
                "n_cases": int(valid.shape[0]),
                "n_liquefied": int(valid["liquefied"].sum()),
                "n_non_liquefied": int(valid.shape[0] - valid["liquefied"].sum()),
                "auc": float(_rank_auc(valid[score], valid["liquefied"])),
                "brier_score_5fold": float(np.mean((prob_cv.to_numpy() - valid["liquefied"].astype(int).to_numpy()) ** 2)),
            }
            metrics.append(
                {
                    **base,
                    "threshold_type": "prespecified",
                    "threshold": float(default_threshold),
                    **default_conf,
                    "status": "field case-history validation of ranking and static transferability",
                }
            )
            metrics.append(
                {
                    **base,
                    "threshold_type": "best_youden",
                    "threshold": float(best_threshold),
                    **best_conf,
                    "status": "field case-history validation of ranking and static transferability",
                }
            )
            thresholds.append(
                {
                    "validation_group": group_name,
                    "predictor": predictor_name,
                    "prespecified_threshold": float(default_threshold),
                    "best_youden_threshold": float(best_threshold),
                    "youden_j": float(best_conf.get("youden_j", np.nan)),
                }
            )

    keep = [
        "case_id",
        "source_dataset",
        "validation_split",
        "earthquake",
        "site",
        "mw",
        "pga",
        "csr75",
        "groundwater_table_m",
        "critical_layer_depth_m",
        "unsaturated_cap_or_cover_m",
        "n120",
        "vs1_ms",
        "fc_pct",
        "gc_pct",
        "observed_liquefaction",
        "n1_60cs_proxy",
        "crr_static_benchmark",
        "fs_static_proxy",
        "g_static_proxy",
        "pf_static_proxy",
        "combined_instrument_score",
        "dpt_screening_score",
        "vs_screening_score",
    ]
    return cases[keep], pd.DataFrame(metrics), pd.DataFrame(thresholds)


def external_case_history_sanity_check() -> tuple[pd.DataFrame, pd.DataFrame]:
    """External trend-discrimination check using the open Hu et al. (2021) case-history dataset.

    This is not site calibration. It checks whether simple demand/resistance and groundwater
    indicators rank historical liquefied cases above non-liquefied cases in the expected direction.
    """
    source = DATA / "external_hu_2021_gravelly_liquefaction_cases.xlsx"
    if not source.exists():
        return pd.DataFrame(), pd.DataFrame()

    raw = pd.read_excel(source, sheet_name="Data")
    label = raw["Liqefied ？"].astype(str).str.strip().str.lower()
    df = pd.DataFrame(
        {
            "case_no": _numeric_column(raw["Case No."]),
            "earthquake": raw["Earthquake Name"].astype(str),
            "site": raw["Site location & Borehole name"].astype(str),
            "liquefied": label.map({"yes": 1, "no": 0}),
            "mw": _numeric_column(raw["Mw"]),
            "pga": _numeric_column(raw["PGA"]),
            "csr75": _numeric_column(raw["CSR7.5"]),
            "fc_pct": _numeric_column(raw["FC (%)"]),
            "gc_pct": _numeric_column(raw["GC (%)"]),
            "n120": _numeric_column(raw["N'120"]),
            "vs1_ms": _numeric_column(raw["Vs1 (m/s)"]),
            "dw_m": _numeric_column(raw["Dw (m)"]),
            "ds_m": _numeric_column(raw["Ds (m)"]),
            "hn_m": _numeric_column(raw["Hn (m)"]),
            "dn_m": _numeric_column(raw["Dn (m)"]),
        }
    ).dropna(subset=["liquefied", "csr75", "dw_m"])

    def zscore(s: pd.Series, inverse: bool = False) -> pd.Series:
        x = s.astype(float)
        z = (x - x.mean()) / max(float(x.std(ddof=0)), 1e-9)
        return -z if inverse else z

    df["shallow_groundwater_index"] = zscore(df["dw_m"], inverse=True)
    df["demand_index"] = zscore(df["csr75"])
    df["dpt_resistance_index"] = zscore(df["n120"], inverse=True)
    df["vs_resistance_index"] = zscore(df["vs1_ms"], inverse=True)
    df["capping_index"] = zscore(df["dn_m"].fillna(df["dn_m"].median()), inverse=True)
    df["dpt_screening_score"] = (
        0.45 * df["demand_index"]
        + 0.30 * df["dpt_resistance_index"]
        + 0.20 * df["shallow_groundwater_index"]
        + 0.05 * df["capping_index"]
    )
    df["vs_screening_score"] = (
        0.45 * df["demand_index"]
        + 0.30 * df["vs_resistance_index"]
        + 0.20 * df["shallow_groundwater_index"]
        + 0.05 * df["capping_index"]
    )
    df["combined_instrument_score"] = (
        0.35 * df["demand_index"]
        + 0.40 * ((df["dpt_resistance_index"] + df["vs_resistance_index"]) / 2.0)
        + 0.20 * df["shallow_groundwater_index"]
        + 0.05 * df["capping_index"]
    )

    summary_rows = []
    for score in ["combined_instrument_score", "dpt_screening_score", "vs_screening_score", "demand_index", "shallow_groundwater_index"]:
        valid = df.dropna(subset=[score, "liquefied"])
        liq = valid[valid["liquefied"].eq(1)][score]
        non = valid[valid["liquefied"].eq(0)][score]
        summary_rows.append(
            {
                "check": score,
                "n_cases": int(valid.shape[0]),
                "n_liquefied": int(valid["liquefied"].sum()),
                "n_non_liquefied": int(valid.shape[0] - valid["liquefied"].sum()),
                "mean_liquefied": float(liq.mean()),
                "mean_non_liquefied": float(non.mean()),
                "median_liquefied": float(liq.median()),
                "median_non_liquefied": float(non.median()),
                "rank_auc_liquefied_higher": float(_rank_auc(valid[score], valid["liquefied"])),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return df, summary


def main() -> None:
    results, summary = run()
    results.to_csv(DATA / "liquefaction_benchmark_results.csv", index=False)
    summary.to_csv(DATA / "liquefaction_benchmark_summary.csv", index=False)
    LAYERS.to_csv(DATA / "synthetic_layer_profile.csv", index=False)
    standard_comp, standard_summary = make_standard_model_comparison(results)
    standard_comp.to_csv(DATA / "standard_model_comparison.csv", index=False)
    standard_summary.to_csv(DATA / "standard_model_comparison_summary.csv", index=False)
    make_model_availability_diagnostics().to_csv(DATA / "model_availability_diagnostics.csv", index=False)
    vertical_dep = make_vertical_dependence_sensitivity(results)
    vertical_dep.to_csv(DATA / "vertical_dependence_sensitivity.csv", index=False)

    fig_data = results[(results.scenario == "extreme") & (results.gradation == "fines_accumulation")]
    series = []
    for layer in LAYERS.layer:
        s = fig_data[fig_data.layer == layer].sort_values("year")
        series.append((layer, s.year.to_numpy(), s.pf_nonstationary.to_numpy()))
    svg_line(FIGURES / "fig01_pf_time_extreme_accumulation.svg", "Non-stationary liquefaction probability", series, "Probability of liquefaction")
    png_line(FIGURES / "fig01_pf_time_extreme_accumulation.png", "Non-stationary liquefaction probability", series, "Probability of liquefaction")

    sc_series = []
    for scenario in ["stationary", "rising", "seasonal", "extreme"]:
        s = results[(results.scenario == scenario) & (results.gradation == "fines_accumulation")]
        s = s.groupby("year", as_index=False).pf_nonstationary.mean()
        sc_series.append((scenario, s.year.to_numpy(), s.pf_nonstationary.to_numpy()))
    svg_line(FIGURES / "fig02_profile_mean_pf_by_scenario.svg", "Profile-average probability by groundwater scenario", sc_series, "Mean profile Pf")
    png_line(FIGURES / "fig02_profile_mean_pf_by_scenario.png", "Profile-average probability by groundwater scenario", sc_series, "Mean profile Pf")

    svg_heatmap(FIGURES / "fig03_depth_time_pf_heatmap.svg", fig_data, "Depth-time probability map, extreme + fines accumulation")
    png_heatmap(FIGURES / "fig03_depth_time_pf_heatmap.png", fig_data, "Depth-time probability map, extreme + fines accumulation")

    comp = results.groupby(["scenario", "gradation", "year"], as_index=False).agg(
        stationary=("pf_stationary", "mean"),
        nonstationary=("pf_nonstationary", "mean"),
        max_layer_pf=("pf_nonstationary", "max"),
        frechet_lower=("pf_nonstationary", "max"),
        frechet_upper=("pf_nonstationary", lambda s: min(1.0, float(s.sum()))),
        psys_independent_layers=("pf_nonstationary", lambda s: 1.0 - float(np.prod(1.0 - s))),
    )
    comp["delta"] = comp.nonstationary - comp.stationary
    comp.to_csv(DATA / "profile_method_comparison.csv", index=False)
    sensitivity_vars = ["z_mid_m", "gw_depth_m", "fc_pct", "n1_60cs", "crr", "csr", "fs_deterministic_nonstationary"]
    sens_rows = []
    for var in sensitivity_vars:
        pearson = float(results[var].corr(results["pf_nonstationary"], method="pearson"))
        # Spearman correlation without scipy: Pearson correlation of average ranks.
        spearman = float(results[var].rank(method="average").corr(results["pf_nonstationary"].rank(method="average"), method="pearson"))
        sens_rows.append(
            {
                "variable": var,
                "pearson_with_pf": pearson,
                "spearman_with_pf": spearman,
                "absolute_spearman": abs(spearman),
            }
        )
    sens = pd.DataFrame(sens_rows).sort_values("absolute_spearman", ascending=False)
    sens.to_csv(DATA / "global_sensitivity_rank.csv", index=False)
    labels = sens["variable"].tolist()
    values = sens["absolute_spearman"].tolist()
    svg_bar(FIGURES / "fig04_global_sensitivity_rank.svg", "Global sensitivity rank for non-stationary Pf", labels, values, "|Spearman rho|")
    png_bar(FIGURES / "fig04_global_sensitivity_rank.png", "Global sensitivity rank for non-stationary Pf", labels, values, "|Spearman rho|")

    external = pd.DataFrame(
        [
            {
                "claim": "Groundwater depth is a controlling input in liquefaction probability.",
                "source": "Holzer et al. (2011); Cruz et al. (2024)",
                "benchmark_check": "Pf increases under rising and extreme groundwater scenarios relative to stationary assumptions.",
                "status": "directionally consistent; not calibration",
            },
            {
                "claim": "Fines content affects liquefaction triggering correlations.",
                "source": "Lee et al. (2020); Yang and Wei (2024)",
                "benchmark_check": "Gradation scenarios change CRR, deterministic FS, and Pf rankings.",
                "status": "directionally consistent; not calibration",
            },
            {
                "claim": "Probabilistic regional liquefaction screening propagates uncertainty in subsurface conditions and groundwater.",
                "source": "Greenfield et al. (2024); USGS (2025)",
                "benchmark_check": "Monte Carlo propagation reports Pf(t), confidence intervals, and profile-level comparisons.",
                "status": "directionally consistent; not calibration",
            },
        ]
    )
    external.to_csv(DATA / "external_trend_consistency_checks.csv", index=False)
    ext_cases, ext_summary = external_case_history_sanity_check()
    if not ext_cases.empty:
        ext_cases.to_csv(DATA / "external_case_history_sanity_check.csv", index=False)
        ext_summary.to_csv(DATA / "external_case_history_sanity_summary.csv", index=False)
        ext_cal, ext_bins = external_calibration_metrics(ext_cases)
        ext_cal.to_csv(DATA / "external_case_history_calibration_metrics.csv", index=False)
        ext_bins.to_csv(DATA / "external_case_history_calibration_bins.csv", index=False)
        static_cases, static_metrics = external_static_limit_state_proxy(ext_cases)
        static_cases.to_csv(DATA / "external_static_limit_state_proxy.csv", index=False)
        static_metrics.to_csv(DATA / "external_static_limit_state_metrics.csv", index=False)
        field_cases, field_metrics, field_thresholds = field_case_history_validation(ext_cases)
        field_cases.to_csv(DATA / "field_validation_cases.csv", index=False)
        field_metrics.to_csv(DATA / "field_validation_metrics.csv", index=False)
        field_thresholds.to_csv(DATA / "field_validation_thresholds.csv", index=False)
        ext_plot = ext_summary[ext_summary["check"].isin(["combined_instrument_score", "dpt_screening_score", "vs_screening_score", "demand_index", "shallow_groundwater_index"])]
        svg_bar(
            FIGURES / "fig05_external_case_history_sanity_auc.svg",
            "External case-history sanity check",
            ext_plot["check"].tolist(),
            ext_plot["rank_auc_liquefied_higher"].tolist(),
            "AUC: liquefied cases ranked higher",
        )
        png_bar(
            FIGURES / "fig05_external_case_history_sanity_auc.png",
            "External case-history sanity check",
            ext_plot["check"].tolist(),
            ext_plot["rank_auc_liquefied_higher"].tolist(),
            "AUC: liquefied cases ranked higher",
        )
    convergence_rows = []
    target_layer = LAYERS.iloc[0]
    target_sc = next(s for s in SCENARIOS if s["scenario"] == "extreme")
    target_grad = next(g for g in GRADATIONS if g["gradation"] == "fines_washout")
    global N_MC, RNG
    original_n = N_MC
    for n in [1000, 3000, 6000, 12000]:
        N_MC = n
        vals = []
        for rep in range(5):
            RNG = np.random.default_rng(1182026 + n + rep)
            vals.append(pf_from_samples(target_layer, 50, target_sc, target_grad, "nonstationary")["pf"])
        convergence_rows.append(
            {
                "sample_size": n,
                "replicates": 5,
                "mean_pf": float(np.mean(vals)),
                "std_pf": float(np.std(vals, ddof=1)),
                "min_pf": float(np.min(vals)),
                "max_pf": float(np.max(vals)),
            }
        )
    N_MC = original_n
    RNG = np.random.default_rng(1182026)
    pd.DataFrame(convergence_rows).to_csv(DATA / "monte_carlo_convergence_check.csv", index=False)
    print("Wrote benchmark outputs to", ROOT)
    print(summary.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
