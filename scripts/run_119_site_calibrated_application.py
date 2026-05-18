"""Site-calibrated extension using the Nisqually CPT case-history dataset.

This script keeps the synthetic benchmark separate from a documented-site
application. It downloads are not performed here; it consumes the DesignSafe
PRJ-3758 XLSX files saved under the Google Drive validation-data folder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.groundwater_calibration import fit_groundwater_model, predict_groundwater
from src.random_fields import empirical_variogram, fit_exponential_variogram
from src.validation_metrics import binary_metrics, fit_logistic, predict_logistic

DATA = ROOT / "data"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

SOURCE = Path(r"G:\Mi unidad\Articulo 118 liquefaction validation datasets\nisqually_designsafe")
EVENT_DATE = "2001-02-28"
EVENT_NAME = "2001 Nisqually earthquake"
MW = 6.8
GAMMA_TOTAL = 18.0
GAMMA_W = 9.81


def _read_summary() -> pd.DataFrame:
    summary = pd.read_excel(SOURCE / "Summary Table S1.xlsx", sheet_name=0)
    summary = summary.rename(
        columns={
            "Site Name": "site_name",
            "CPT Longitude (WGS84)": "longitude",
            "CPT Latitude (WGS84)": "latitude",
            "CPT Test Date": "test_date",
            "WTD (m)": "wtd_m",
            "Pre-drill (m)": "predrill_m",
            "Manifestation": "manifestation",
            "Conditional Median PGA (g)": "pga_g",
            "Conditional Lognormal Standard Deviation of\nPGA": "pga_lnsd",
            "Geotechnical Reference": "geotechnical_reference",
            "Liquefaction Reference": "liquefaction_reference",
            "VS30 (m/s)": "vs30_m_s",
            "Cone Tip Resistance Type": "cone_tip_resistance_type",
        }
    )
    summary["site_id"] = summary["site_name"].str.replace(r"[^A-Za-z0-9]+", "_", regex=True).str.strip("_")
    summary["observed_liquefaction"] = summary["manifestation"].astype(str).str.lower().eq("yes").astype(int)
    return summary.dropna(subset=["site_name", "test_date", "wtd_m", "pga_g", "observed_liquefaction"])


def _metadata_from_sheet(path: Path) -> dict:
    meta = {}
    raw = pd.read_excel(path, sheet_name=0, header=None, usecols="G:H", nrows=12)
    for _, row in raw.iterrows():
        if pd.notna(row.iloc[0]):
            meta[str(row.iloc[0]).strip()] = row.iloc[1]
    return meta


def _read_cpt(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, usecols="A:D")
    df = df.rename(columns={"depth_m": "depth_m", "q_MPa": "qc_mpa", "fs_MPa": "fs_mpa", "u2_MPa": "u2_mpa"})
    return df.dropna(subset=["depth_m", "qc_mpa", "fs_mpa"]).query("depth_m >= 0").copy()


def _ic_proxy(qc_mpa: pd.Series, fs_mpa: pd.Series) -> pd.Series:
    rf = 100.0 * fs_mpa.clip(lower=1e-5) / qc_mpa.clip(lower=1e-5)
    # Robertson-style compact proxy; not a formal Ic normalization.
    return ((3.47 - np.log10(qc_mpa.clip(lower=1e-3) * 1000.0)) ** 2 + (np.log10(rf.clip(lower=0.01)) + 1.22) ** 2) ** 0.5


def build_site_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = _read_summary()
    profile_rows = []
    cpt_rows = []
    for _, row in summary.iterrows():
        xlsx = SOURCE / f"{row.site_name}.xlsx"
        if not xlsx.exists():
            continue
        cpt = _read_cpt(xlsx)
        cpt["site_id"] = row.site_id
        cpt["site_name"] = row.site_name
        cpt["ic_proxy"] = _ic_proxy(cpt["qc_mpa"], cpt["fs_mpa"])
        cpt["friction_ratio_pct"] = 100.0 * cpt["fs_mpa"].clip(lower=1e-6) / cpt["qc_mpa"].clip(lower=1e-6)
        cpt_rows.append(cpt)
        for k, (z0, z1) in enumerate([(0, 2), (2, 5), (5, 10), (10, 20)], 1):
            sub = cpt[(cpt.depth_m >= z0) & (cpt.depth_m < z1)]
            if sub.empty:
                continue
            profile_rows.append(
                {
                    "site_id": row.site_id,
                    "borehole_id": row.site_id,
                    "layer_id": f"L{k}",
                    "z_top_m": z0,
                    "z_bot_m": z1,
                    "z_mid_m": (z0 + z1) / 2,
                    "soil_type": "CPT-derived mixed soil",
                    "gamma_total_kN_m3": GAMMA_TOTAL,
                    "gamma_sat_kN_m3": 19.0,
                    "n60": np.nan,
                    "n160": np.nan,
                    "qc": float(sub.qc_mpa.median()),
                    "fs_cpt": float(sub.fs_mpa.median()),
                    "ic": float(sub.ic_proxy.median()),
                    "vs1": np.nan,
                    "fc_pct": np.nan,
                    "d50_mm": np.nan,
                    "pi": np.nan,
                    "gravel_pct": np.nan,
                    "measurement_date": row.test_date,
                    "data_source": "DesignSafe PRJ-3758 Nisqually CPT case-history XLSX",
                }
            )
    cpt_all = pd.concat(cpt_rows, ignore_index=True)
    profile = pd.DataFrame(profile_rows)
    gw = summary[
        ["site_id", "test_date", "wtd_m", "geotechnical_reference"]
    ].rename(columns={"test_date": "date", "wtd_m": "gw_depth_m", "geotechnical_reference": "data_source"})
    gw["measurement_type"] = "CPT report water-table depth"
    gw["screen_depth_m"] = np.nan
    gw["datum"] = "ground surface"
    gw["uncertainty_sd_m"] = 0.35
    grad = profile[
        ["site_id", "borehole_id", "layer_id", "measurement_date", "fc_pct", "d50_mm", "pi", "gravel_pct", "data_source"]
    ].rename(columns={"measurement_date": "date"})
    grad["test_method"] = "not reported; CPT Ic retained as gradation proxy"
    grad["uncertainty_sd"] = np.nan
    events = summary[
        ["site_id", "site_id", "pga_g", "observed_liquefaction", "manifestation", "liquefaction_reference"]
    ].copy()
    events.columns = ["site_id", "borehole_id", "pga_g", "observed_liquefaction", "manifestation_type", "data_source"]
    events["event_name"] = EVENT_NAME
    events["event_date"] = EVENT_DATE
    events["mw"] = MW
    events["csr_7p5"] = np.nan
    events["evidence_quality"] = "published case-history manifestation"
    events["notes"] = "Nisqually CPT case-history observation"
    return summary, cpt_all, profile, gw, grad, events


def _site_features(summary: pd.DataFrame, cpt_all: pd.DataFrame, gw_params: dict) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        cpt = cpt_all[cpt_all.site_id.eq(row.site_id)]
        if cpt.empty:
            continue
        shallow = cpt[(cpt.depth_m >= row.predrill_m) & (cpt.depth_m <= min(12.0, cpt.depth_m.max()))].copy()
        if shallow.empty:
            shallow = cpt.copy()
        shallow["ic_proxy"] = _ic_proxy(shallow.qc_mpa, shallow.fs_mpa)
        q10 = float(shallow.qc_mpa.quantile(0.10))
        q25 = float(shallow.qc_mpa.quantile(0.25))
        ic_med = float(shallow.ic_proxy.median())
        rf_med = float(shallow.friction_ratio_pct.median())
        z = shallow.depth_m.to_numpy()
        y = np.log(np.clip(shallow.qc_mpa.to_numpy(), 1e-3, None))
        variogram = fit_exponential_variogram(z[:: max(len(z) // 200, 1)], y[:: max(len(y) // 200, 1)])
        pred_event = predict_groundwater(pd.Series([EVENT_DATE]), gw_params).iloc[0]
        rows.append(
            {
                "site_id": row.site_id,
                "site_name": row.site_name,
                "longitude": row.longitude,
                "latitude": row.latitude,
                "test_date": row.test_date,
                "event_date": EVENT_DATE,
                "observed_liquefaction": int(row.observed_liquefaction),
                "pga_g": float(row.pga_g),
                "wtd_observed_m": float(row.wtd_m),
                "wtd_event_model_mean_m": float(pred_event.gw_depth_mean_m),
                "wtd_event_site_adjusted_m": float(row.wtd_m + pred_event.gw_depth_mean_m - gw_params["beta0"]),
                "qc10_mpa": q10,
                "qc25_mpa": q25,
                "ic_median": ic_med,
                "friction_ratio_median_pct": rf_med,
                "theta_z_m": variogram["theta_z_m"],
                "variogram_rmse": variogram["rmse"],
                "source_dataset": "DesignSafe PRJ-3758",
            }
        )
    return pd.DataFrame(rows)


def _crossfit_models(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = features.copy().reset_index(drop=True)
    df["site_family"] = df["site_name"].str.replace(r"-\d+$", "", regex=True)
    df["validation_split"] = np.where(df["site_family"].isin(["SODO", "UW"]), "heldout_SODO_UW", "train_development")
    models = {
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
    pred_rows = []
    metric_rows = []
    y = df["observed_liquefaction"].to_numpy()
    for name, cols in models.items():
        train = df.validation_split.eq("train_development")
        test = df.validation_split.eq("heldout_SODO_UW")
        params = fit_logistic(df.loc[train, cols].to_numpy(), df.loc[train, "observed_liquefaction"].to_numpy())
        prob = np.full(len(df), np.nan, dtype=float)
        prob[test.to_numpy()] = predict_logistic(params, df.loc[test, cols].to_numpy())
        m = binary_metrics(prob[test.to_numpy()], y[test.to_numpy()])
        m.update(
            {
                "model_name": name,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "validation_protocol": "out-of-sample family holdout: train on 16 non-SODO/UW cases, test on SODO+UW siblings",
            }
        )
        metric_rows.append(m)
        for i, p in enumerate(prob):
            pred_rows.append(
                {
                    "site_id": df.loc[i, "site_id"],
                    "site_family": df.loc[i, "site_family"],
                    "model_name": name,
                    "predicted_pf": float(p) if not np.isnan(p) else np.nan,
                    "observed_liquefaction": int(y[i]),
                    "validation_split": df.loc[i, "validation_split"],
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows)


def _plot_groundwater(obs: pd.DataFrame, curve: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1200, 720), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 22)
    small = ImageFont.truetype("arial.ttf", 18)
    left, right, top, bottom = 90, 1140, 70, 620
    dates = pd.to_datetime(pd.concat([obs["date"], curve["date"]]))
    xmin, xmax = dates.min(), dates.max()
    ymin = min(obs.gw_depth_m.min(), curve.gw_depth_ci_low_m.min()) - 0.5
    ymax = max(obs.gw_depth_m.max(), curve.gw_depth_ci_high_m.max()) + 0.5

    def xmap(dt):
        return left + (pd.to_datetime(dt) - xmin).days / max((xmax - xmin).days, 1) * (right - left)

    def ymap(y):
        return bottom - (y - ymin) / max(ymax - ymin, 1e-6) * (bottom - top)

    d.rectangle((left, top, right, bottom), outline=(40, 40, 40), width=2)
    pts_low = [(xmap(r.date), ymap(r.gw_depth_ci_low_m)) for r in curve.itertuples()]
    pts_high = [(xmap(r.date), ymap(r.gw_depth_ci_high_m)) for r in curve.itertuples()]
    band = pts_low + list(reversed(pts_high))
    d.polygon(band, fill=(210, 225, 245))
    pts = [(xmap(r.date), ymap(r.gw_depth_mean_m)) for r in curve.itertuples()]
    d.line(pts, fill=(33, 90, 150), width=4)
    for r in obs.itertuples():
        d.ellipse((xmap(r.date) - 4, ymap(r.gw_depth_m) - 4, xmap(r.date) + 4, ymap(r.gw_depth_m) + 4), fill=(20, 20, 20))
    xe = xmap(EVENT_DATE)
    d.line((xe, top, xe, bottom), fill=(180, 40, 40), width=3)
    d.text((xe + 8, top + 8), "Nisqually event", fill=(180, 40, 40), font=small)
    d.text((left, 25), "Observed and calibrated groundwater trajectory, Nisqually CPT dataset", fill=(0, 0, 0), font=font)
    d.text((left, bottom + 35), "Observed WTD points, fitted mean curve, 95% interval and earthquake date", fill=(0, 0, 0), font=small)
    img.save(path)


def _plot_validation(metrics: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1000, 620), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 22)
    small = ImageFont.truetype("arial.ttf", 18)
    d.text((50, 30), "Out-of-sample model comparison, Nisqually site-calibrated extension", font=font, fill=(0, 0, 0))
    x0, y0 = 260, 520
    d.line((x0, 90, x0, y0), fill=(40, 40, 40), width=2)
    d.line((x0, y0, 930, y0), fill=(40, 40, 40), width=2)
    rows = metrics.sort_values("model_name")
    vmax = max(rows.brier_score.max(), 0.3)
    gap = 90
    for i, r in enumerate(rows.itertuples()):
        y = 110 + i * gap
        d.text((40, y + 12), r.model_name.replace("_", " "), font=small, fill=(0, 0, 0))
        w = int((r.brier_score / vmax) * 600)
        d.rectangle((x0, y, x0 + w, y + 34), fill=(90, 130, 170))
        d.text((x0 + w + 10, y + 5), f"Brier {r.brier_score:.3f}; AUC {r.auc:.3f}", font=small, fill=(0, 0, 0))
    img.save(path)


def main() -> None:
    summary, cpt_all, profile, gw, grad, events = build_site_tables()
    gw_params, gw_fit = fit_groundwater_model(gw, EVENT_DATE)
    features = _site_features(summary, cpt_all, gw_params)
    pred, metrics = _crossfit_models(features)
    # Required CSV outputs.
    profile.to_csv(DATA / "site_profile_calibrated.csv", index=False)
    gw.to_csv(DATA / "site_groundwater_timeseries.csv", index=False)
    grad.to_csv(DATA / "site_gradation_timeseries.csv", index=False)
    events.to_csv(DATA / "site_event_observations.csv", index=False)
    pd.DataFrame([{**{"site_id": "Nisqually_PRJ3758", "calibration_period": "CPT report dates 1987-2003", "validation_period": "2001-02-28 event via spatial CV"}, **gw_params}]).to_csv(
        DATA / "site_groundwater_model_parameters.csv", index=False
    )
    grad_params = profile[["site_id", "layer_id", "ic"]].copy()
    grad_params["fc0_mean"] = np.nan
    grad_params["fc0_sd"] = np.nan
    grad_params["fc_trend_pct_per_year"] = np.nan
    grad_params["trend_sd"] = np.nan
    grad_params["d50_mean"] = np.nan
    grad_params["d50_sd"] = np.nan
    grad_params["model_status"] = "baseline_uncertainty_only"
    grad_params.to_csv(DATA / "site_gradation_model_parameters.csv", index=False)
    features.to_csv(OUTPUTS / "site_calibrated_results.csv", index=False)
    pred.to_csv(OUTPUTS / "site_prediction_probabilities.csv", index=False)
    metrics.to_csv(OUTPUTS / "site_validation_metrics.csv", index=False)
    model_comp = pred.merge(features[["site_id", "wtd_event_site_adjusted_m", "qc10_mpa", "ic_median", "theta_z_m"]], on="site_id", how="left")
    model_comp.rename(columns={"predicted_pf": "pf"}, inplace=True)
    model_comp["year"] = 2001
    model_comp["notes"] = "spatial cross-fitted site-calibrated probability"
    model_comp.to_csv(OUTPUTS / "site_model_form_comparison.csv", index=False)
    variogram_rows = []
    for site_id, g in cpt_all.groupby("site_id"):
        sub = g.iloc[:: max(len(g) // 200, 1)]
        pars = fit_exponential_variogram(sub.depth_m.to_numpy(), np.log(np.clip(sub.qc_mpa.to_numpy(), 1e-3, None)))
        pars.update({"site_id": site_id, "variable": "log_qc_mpa", "transform": "log", "aic": np.nan, "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan})
        variogram_rows.append(pars)
    pd.DataFrame(variogram_rows).to_csv(DATA / "site_vertical_variogram_parameters.csv", index=False)
    rf = features[["site_id", "theta_z_m"]].copy()
    for name in ["M0_static_stationary", "M1_nonstationary_groundwater_only", "M2_nonstationary_groundwater_gradation", "M3_full_nonstationary_random_field"]:
        tmp = pred[pred.model_name.eq(name)].set_index("site_id")["predicted_pf"]
        rf[f"pf_{name}"] = rf.site_id.map(tmp)
    rf["year"] = 2001
    rf["psys_any"] = rf["pf_M3_full_nonstationary_random_field"]
    rf["psys_two_or_more"] = np.nan
    rf["expected_failed_layers"] = np.nan
    rf["median_critical_depth_m"] = np.nan
    rf["random_field_model"] = "site-specific exponential vertical variogram diagnostic"
    rf.to_csv(OUTPUTS / "site_random_field_system_probability.csv", index=False)
    curve = predict_groundwater(pd.date_range(summary.test_date.min(), summary.test_date.max(), periods=220), gw_params)
    _plot_groundwater(gw, curve, FIGURES / "fig06_site_groundwater_calibration.png")
    _plot_validation(metrics, FIGURES / "fig07_model_form_comparison_site.png")
    emp = empirical_variogram(cpt_all.depth_m.iloc[::100].to_numpy(), np.log(np.clip(cpt_all.qc_mpa.iloc[::100].to_numpy(), 1e-3, None)))
    emp.to_csv(DATA / "site_vertical_variogram_empirical.csv", index=False)
    _plot_validation(metrics[metrics.model_name.str.contains("M")], FIGURES / "fig11_site_validation_brier.png")
    print(metrics[["model_name", "auc", "brier_score", "log_loss", "calibration_intercept", "calibration_slope", "sensitivity", "specificity"]].to_string(index=False))


if __name__ == "__main__":
    main()
