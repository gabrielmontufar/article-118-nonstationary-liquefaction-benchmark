"""Canterbury PRJ-2937 multi-site temporal transfer validation.

This expands the earlier single-CPT Canterbury stress test to all usable CPT
records in CANTERBURYDATASET.mat. The fitted probability models remain trained
only on the Nisqually PRJ-3758 site-calibrated table, so Canterbury is treated
as an external temporal/site transfer domain rather than as a recalibrated
training source.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation_metrics import binary_metrics, fit_logistic, predict_logistic


DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
LOCAL_SOURCE = ROOT / "raw_designsafe" / "canterbury_designsafe_PRJ-2937" / "CANTERBURYDATASET.mat"

MODEL_FEATURES = {
    "M0_static_stationary": ["pga_g"],
    "M1_nonstationary_groundwater_only": ["pga_g", "wtd_event_site_adjusted_m"],
    "M2_nonstationary_groundwater_gradation": ["pga_g", "qc10_mpa", "wtd_event_site_adjusted_m", "ic_median"],
    "M3_full_nonstationary_random_field": [
        "pga_g",
        "qc10_mpa",
        "wtd_event_site_adjusted_m",
        "ic_median",
        "theta_z_m",
    ],
}

EVENTS = [
    ("Yr2010", "2010 Darfield earthquake", "2010-09-04"),
    ("Yr2011", "2011 Christchurch earthquake", "2011-02-22"),
    ("Yr2016", "2016 Valentine earthquake", "2016-02-14"),
]


def _as_records() -> np.ndarray:
    if not LOCAL_SOURCE.exists():
        raise FileNotFoundError(LOCAL_SOURCE)
    mat = loadmat(LOCAL_SOURCE, squeeze_me=True, struct_as_record=False)
    return np.ravel(mat["CANTERBURYDATASET"])


def _scalar(value: object) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _text(value: object) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def _ic_proxy(qc_mpa: np.ndarray, fs_mpa: np.ndarray) -> np.ndarray:
    rf = 100.0 * np.clip(fs_mpa, 1e-6, None) / np.clip(qc_mpa, 1e-6, None)
    return (
        (3.47 - np.log10(np.clip(qc_mpa, 1e-3, None) * 1000.0)) ** 2
        + (np.log10(np.clip(rf, 0.01, None)) + 1.22) ** 2
    ) ** 0.5


def _theta_z(depth: np.ndarray, qc_mpa: np.ndarray) -> float:
    valid = np.isfinite(depth) & np.isfinite(qc_mpa) & (qc_mpa > 0)
    if int(valid.sum()) < 8:
        return float("nan")
    d = depth[valid]
    q = np.log(np.clip(qc_mpa[valid], 1e-3, None))
    order = np.argsort(d)
    d = d[order]
    q = q[order]
    q = q - np.nanmean(q)
    step = float(np.nanmedian(np.diff(d))) if len(d) > 1 else float("nan")
    if not np.isfinite(step) or step <= 0:
        return float("nan")
    max_lag = min(60, max(2, len(q) // 3))
    variance = float(np.nanvar(q))
    if variance <= 1e-12:
        return float("nan")
    for lag in range(1, max_lag + 1):
        corr = float(np.nanmean(q[:-lag] * q[lag:]) / variance)
        if corr <= np.exp(-1.0):
            return float(lag * step)
    return float(max_lag * step)


def _feature_row(record: object, record_index: int) -> dict[str, object] | None:
    try:
        depth = np.asarray(record.depth, dtype=float)
        qc_mpa = np.asarray(record.qc, dtype=float) / 1000.0
        fs_mpa = np.asarray(record.fs, dtype=float) / 1000.0
    except Exception:
        return None
    valid = np.isfinite(depth) & np.isfinite(qc_mpa) & np.isfinite(fs_mpa) & (qc_mpa > 0)
    if int(valid.sum()) < 8:
        return None
    shallow = valid & (depth <= 12.0)
    if int(shallow.sum()) < 4:
        shallow = valid
    ic = _ic_proxy(qc_mpa[valid], fs_mpa[valid])
    site_name = _text(getattr(record, "FILEname", getattr(record, "CPTname", "")))
    cpt_name = _text(getattr(record, "CPTname", ""))
    cpt_id = _text(getattr(record, "ID", ""))
    return {
        "site_id": f"Canterbury_{record_index:05d}",
        "site_name": site_name,
        "cpt_name": cpt_name,
        "borehole_id": cpt_id,
        "latitude": _scalar(getattr(record, "NorthingWGS84", np.nan)),
        "longitude": _scalar(getattr(record, "EastingWGS84", np.nan)),
        "qc10_mpa": float(np.nanquantile(qc_mpa[shallow], 0.10)),
        "qc50_mpa": float(np.nanmedian(qc_mpa[shallow])),
        "ic_median": float(np.nanmedian(ic)),
        "theta_z_m": _theta_z(depth, qc_mpa),
        "n_profile_points": int(valid.sum()),
        "data_source": "DesignSafe PRJ-2937 CANTERBURYDATASET.mat",
    }


def build_canterbury_event_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = _as_records()
    site_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    excluded_code10 = 0
    for idx, record in enumerate(records):
        site = _feature_row(record, idx)
        if site is None:
            continue
        site_rows.append(site)
        for event_key, event_name, event_date in EVENTS:
            manifestation = int(_scalar(getattr(record.Manifestation, event_key)))
            if manifestation == 10:
                excluded_code10 += 1
                continue
            pga_g = _scalar(getattr(record.PGA, event_key))
            gw_depth = _scalar(getattr(record.GWT, event_key))
            mw = _scalar(getattr(record.Magnitude, event_key))
            pga_sigma = _scalar(getattr(record.PGAsigma, event_key))
            if not np.isfinite(pga_g) or not np.isfinite(gw_depth):
                continue
            row = dict(site)
            row.update(
                {
                    "event_key": event_key,
                    "event_name": event_name,
                    "event_date": event_date,
                    "year": int(event_key.replace("Yr", "")),
                    "pga_g": float(pga_g),
                    "mw": float(mw),
                    "pga_lnsd": float(pga_sigma),
                    "wtd_event_site_adjusted_m": float(gw_depth),
                    "manifestation_code": manifestation,
                    "observed_liquefaction": int(manifestation > 0),
                    "label_rule": "manifestation_code 1-5 = liquefaction; 0 = no liquefaction; 10 excluded as indeterminate",
                }
            )
            event_rows.append(row)
    features = pd.DataFrame(event_rows)
    sites = pd.DataFrame(site_rows)
    if not features.empty:
        theta_median = float(features["theta_z_m"].median(skipna=True))
        features["theta_z_m"] = features["theta_z_m"].fillna(theta_median)
    features.attrs["excluded_code10"] = excluded_code10
    return sites, features


def _prob_metrics(prob: np.ndarray, y: np.ndarray) -> dict[str, object]:
    m = binary_metrics(prob, y)
    p = np.asarray(prob, dtype=float)
    yy = np.asarray(y, dtype=int)
    m["n_test"] = int(len(yy))
    m["prevalence"] = float(yy.mean())
    return m


def _cluster_bootstrap_delta(
    cases: pd.DataFrame,
    candidate_col: str,
    reference_col: str = "M0_static_stationary",
    reps: int = 2000,
    seed: int = 11820260607,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    cluster_to_idx = {
        site: idx.to_numpy()
        for site, idx in cases.reset_index().groupby("site_id")["index"]
    }
    clusters = np.array(sorted(cluster_to_idx))
    y_all = cases["observed_liquefaction"].to_numpy(dtype=float)
    cand_all = cases[candidate_col].to_numpy(dtype=float)
    ref_all = cases[reference_col].to_numpy(dtype=float)
    diffs = []
    for _ in range(reps):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([cluster_to_idx[site] for site in sampled])
        y = y_all[idx]
        cand = cand_all[idx]
        ref = ref_all[idx]
        diffs.append(float(np.mean((cand - y) ** 2 - (ref - y) ** 2)))
    arr = np.asarray(diffs)
    point = float(np.mean((cand_all - y_all) ** 2 - (ref_all - y_all) ** 2))
    return {
        "candidate": candidate_col,
        "reference": reference_col,
        "paired_brier_delta_vs_reference": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "bootstrap_reps": reps,
        "cluster_unit": "site_id",
    }


def evaluate_transfer(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    nisqually = pd.read_csv(OUTPUTS / "site_calibrated_results.csv")
    pred_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    wide = features[
        [
            "site_id",
            "site_name",
            "borehole_id",
            "event_key",
            "event_name",
            "event_date",
            "year",
            "observed_liquefaction",
            "manifestation_code",
            "pga_g",
            "wtd_event_site_adjusted_m",
            "qc10_mpa",
            "ic_median",
            "theta_z_m",
        ]
    ].copy()

    y = features["observed_liquefaction"].to_numpy(dtype=int)
    for model_name, cols in MODEL_FEATURES.items():
        params = fit_logistic(nisqually[cols].to_numpy(), nisqually["observed_liquefaction"].to_numpy(), l2=0.1)
        prob = predict_logistic(params, features[cols].to_numpy())
        wide[model_name] = prob
        metrics = _prob_metrics(prob, y)
        metrics.update(
            {
                "model_name": model_name,
                "n_train": int(len(nisqually)),
                "n_test": int(len(features)),
                "n_sites": int(features["site_id"].nunique()),
                "validation_protocol": "Nisqually-trained external transfer to all usable Canterbury CPT event states",
            }
        )
        metric_rows.append(metrics)
        for row, p in zip(features.itertuples(index=False), prob):
            pred_rows.append(
                {
                    "site_id": row.site_id,
                    "site_name": row.site_name,
                    "borehole_id": row.borehole_id,
                    "event_key": row.event_key,
                    "event_name": row.event_name,
                    "event_date": row.event_date,
                    "year": row.year,
                    "model_name": model_name,
                    "predicted_pf": float(p),
                    "observed_liquefaction": int(row.observed_liquefaction),
                    "manifestation_code": int(row.manifestation_code),
                    "pga_g": float(row.pga_g),
                    "gw_depth_m": float(row.wtd_event_site_adjusted_m),
                    "qc10_mpa": float(row.qc10_mpa),
                    "ic_median": float(row.ic_median),
                    "theta_z_m": float(row.theta_z_m),
                    "validation_protocol": "Nisqually-trained external transfer to all usable Canterbury CPT event states",
                }
            )

    policy = wide.copy()
    policy["conservative_max_M0_M3"] = policy[list(MODEL_FEATURES)].max(axis=1)
    policy["domain_adaptive_policy_pf"] = np.maximum(policy["M0_static_stationary"], policy["M3_full_nonstationary_random_field"])
    for policy_name in ["conservative_max_M0_M3", "domain_adaptive_policy_pf"]:
        metrics = _prob_metrics(policy[policy_name].to_numpy(), y)
        metrics.update(
            {
                "model_name": policy_name,
                "n_train": int(len(nisqually)),
                "n_test": int(len(features)),
                "n_sites": int(features["site_id"].nunique()),
                "validation_protocol": "Nisqually-trained external transfer to all usable Canterbury CPT event states",
            }
        )
        metric_rows.append(metrics)

    boot = {
        "M1_vs_M0": _cluster_bootstrap_delta(policy, "M1_nonstationary_groundwater_only"),
        "M2_vs_M0": _cluster_bootstrap_delta(policy, "M2_nonstationary_groundwater_gradation"),
        "M3_vs_M0": _cluster_bootstrap_delta(policy, "M3_full_nonstationary_random_field"),
        "conservative_max_vs_M0": _cluster_bootstrap_delta(policy, "conservative_max_M0_M3"),
        "domain_adaptive_policy_vs_M0": _cluster_bootstrap_delta(policy, "domain_adaptive_policy_pf"),
    }
    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows), policy, boot


def plot_event_metrics(metrics: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1200, 760), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 24)
    small = ImageFont.truetype("arial.ttf", 18)
    d.text((40, 28), "Canterbury multi-site external transfer: Brier and false negatives", font=font, fill=(0, 0, 0))
    rows = metrics.sort_values("brier_score").reset_index(drop=True)
    left, top = 340, 110
    max_brier = max(float(rows["brier_score"].max()), 1e-6)
    colors = [(80, 80, 80), (45, 120, 170), (30, 140, 90), (150, 70, 140), (180, 110, 40), (120, 70, 170)]
    for i, row in rows.iterrows():
        y = top + i * 88
        name = str(row["model_name"]).replace("_", " ")
        d.text((40, y + 6), name[:35], font=small, fill=(0, 0, 0))
        width = int(float(row["brier_score"]) / max_brier * 650)
        d.rectangle((left, y, left + width, y + 30), fill=colors[i % len(colors)])
        d.text((left + width + 12, y + 4), f"Brier {row['brier_score']:.3f}", font=small, fill=(0, 0, 0))
        d.text((left, y + 38), f"FN {int(row['fn'])}; FP {int(row['fp'])}; AUC {row['auc']:.3f}", font=small, fill=(0, 0, 0))
    n = int(rows.iloc[0]["n_test"])
    sites = int(rows.iloc[0]["n_sites"])
    d.text((40, 690), f"External domain: {n} scored event states across {sites} CPT records; manifestation code 10 excluded.", font=small, fill=(0, 0, 0))
    img.save(path)


def main() -> None:
    sites, features = build_canterbury_event_features()
    pred, metrics, policy_cases, boot = evaluate_transfer(features)

    DATA.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    sites.to_csv(DATA / "canterbury_multi_site_cpt_summary.csv", index=False)
    features.to_csv(DATA / "canterbury_multi_site_event_features.csv", index=False)
    pred.to_csv(OUTPUTS / "canterbury_multi_site_temporal_prediction_probabilities.csv", index=False)
    metrics.to_csv(OUTPUTS / "canterbury_multi_site_temporal_validation_metrics.csv", index=False)
    policy_cases.to_csv(OUTPUTS / "canterbury_multi_site_temporal_policy_cases.csv", index=False)
    plot_event_metrics(metrics, FIGURES / "fig10_canterbury_multi_site_validation.png")

    best = metrics.sort_values(["brier_score", "model_name"]).iloc[0]
    summary = {
        "status": "PASS_EXTERNAL_CANTERBURY_MULTI_SITE_VALIDATION" if int(len(features)) > 100 else "FAIL_TOO_FEW_EXTERNAL_CASES",
        "source": str(LOCAL_SOURCE.relative_to(ROOT)),
        "records_in_mat": int(len(_as_records())),
        "usable_cpt_records": int(sites["site_id"].nunique()),
        "scored_event_states": int(len(features)),
        "excluded_manifestation_code10_states": int(features.attrs.get("excluded_code10", 0)),
        "label_rule": "manifestation_code 1-5 = liquefaction; 0 = no liquefaction; 10 excluded as indeterminate",
        "training_source": "outputs/site_calibrated_results.csv",
        "training_cases": int(pd.read_csv(OUTPUTS / "site_calibrated_results.csv").shape[0]),
        "validation_protocol": "Nisqually-trained external transfer to all usable Canterbury CPT event states",
        "best_brier_model": str(best["model_name"]),
        "best_brier": float(best["brier_score"]),
        "best_false_negatives": int(best["fn"]),
        "reference_m0_brier": float(metrics[metrics.model_name.eq("M0_static_stationary")].iloc[0]["brier_score"]),
        "bootstrap": boot,
        "claim_enabled": (
            "Canterbury is now a large external temporal/site transfer check rather than a three-event descriptive "
            "example; claims remain transfer-validation claims because model coefficients are trained on Nisqually."
        ),
        "claim_blocked": (
            "Do not call this Canterbury site-calibrated training or universal non-stationary superiority; "
            "the domain is externally transferred and retains calibration/false-negative tradeoffs."
        ),
    }
    (OUTPUTS / "canterbury_multi_site_temporal_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics[["model_name", "n_test", "n_sites", "auc", "brier_score", "log_loss", "sensitivity", "specificity", "fn", "fp"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
