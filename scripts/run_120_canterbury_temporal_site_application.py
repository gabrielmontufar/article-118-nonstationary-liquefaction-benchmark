"""Temporal site application using Canterbury PRJ-2937.

This script adds a minimal documented-site time application to the benchmark
package. It uses one CPT location, 100 Osbourne St CPT01, with three earthquake
states from the Canterbury dataset. The transfer model is trained on the
Nisqually PRJ-3758 case histories produced by run_119 and then evaluated on
the Canterbury event sequence.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.random_fields import fit_exponential_variogram
from src.validation_metrics import binary_metrics, fit_logistic, predict_logistic

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
SOURCE = Path(r"G:\Mi unidad\Articulo 118 liquefaction validation datasets\canterbury_designsafe_PRJ-2937\CANTERBURYDATASET.mat")


def _ic_proxy(qc_mpa: np.ndarray, fs_mpa: np.ndarray) -> np.ndarray:
    rf = 100.0 * np.clip(fs_mpa, 1e-6, None) / np.clip(qc_mpa, 1e-6, None)
    return ((3.47 - np.log10(np.clip(qc_mpa, 1e-3, None) * 1000.0)) ** 2 + (np.log10(np.clip(rf, 0.01, None)) + 1.22) ** 2) ** 0.5


def load_canterbury_site() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mat = loadmat(SOURCE, squeeze_me=True, struct_as_record=False)
    record = next(r for r in mat["CANTERBURYDATASET"] if r.CPTname == "100 Osbourne St - CPT01 TabulatedData")
    depth = np.asarray(record.depth, dtype=float)
    qc_mpa = np.asarray(record.qc, dtype=float) / 1000.0
    fs_mpa = np.asarray(record.fs, dtype=float) / 1000.0
    u2_mpa = np.asarray(record.u2, dtype=float) / 1000.0
    ic = _ic_proxy(qc_mpa, fs_mpa)
    cpt = pd.DataFrame(
        {
            "site_id": "Canterbury_100_Osbourne_CPT01",
            "borehole_id": "CPT01",
            "depth_m": depth,
            "qc_mpa": qc_mpa,
            "fs_mpa": fs_mpa,
            "u2_mpa": u2_mpa,
            "ic_proxy": ic,
            "latitude": float(record.NorthingWGS84),
            "longitude": float(record.EastingWGS84),
            "data_source": "DesignSafe PRJ-2937 CANTERBURYDATASET.mat",
        }
    )
    profile_rows = []
    for k, (z0, z1) in enumerate([(0, 2), (2, 5), (5, 10), (10, 20)], 1):
        sub = cpt[(cpt.depth_m >= z0) & (cpt.depth_m < z1)]
        if sub.empty:
            continue
        profile_rows.append(
            {
                "site_id": "Canterbury_100_Osbourne_CPT01",
                "borehole_id": "CPT01",
                "layer_id": f"L{k}",
                "z_top_m": z0,
                "z_bot_m": z1,
                "z_mid_m": (z0 + z1) / 2,
                "soil_type": "CPT-derived Canterbury alluvial soil",
                "gamma_total_kN_m3": 18.0,
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
                "measurement_date": "site CPT in PRJ-2937",
                "data_source": "DesignSafe PRJ-2937 CANTERBURYDATASET.mat",
            }
        )
    profile = pd.DataFrame(profile_rows)
    events = []
    dates = {"Yr2010": "2010-09-04", "Yr2011": "2011-02-22", "Yr2016": "2016-02-14"}
    for key, label in [("Yr2010", "2010 Darfield earthquake"), ("Yr2011", "2011 Christchurch earthquake"), ("Yr2016", "2016 Valentine earthquake")]:
        manifestation = int(getattr(record.Manifestation, key))
        events.append(
            {
                "site_id": "Canterbury_100_Osbourne_CPT01",
                "borehole_id": "CPT01",
                "event_name": label,
                "event_date": dates[key],
                "year": int(key.replace("Yr", "")),
                "mw": float(getattr(record.Magnitude, key)),
                "pga_g": float(getattr(record.PGA, key)),
                "pga_lnsd": float(getattr(record.PGAsigma, key)),
                "gw_depth_m": float(getattr(record.GWT, key)),
                "observed_liquefaction": int(manifestation > 0 and manifestation != 10),
                "manifestation_code": manifestation,
                "manifestation_type": "published Canterbury manifestation code",
                "evidence_quality": "curated DesignSafe case-history event state",
                "data_source": "DesignSafe PRJ-2937 CANTERBURYDATASET.mat",
            }
        )
    return cpt, profile, pd.DataFrame(events)


def evaluate_temporal_transfer(cpt: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nisqually = pd.read_csv(OUTPUTS / "site_calibrated_results.csv")
    shallow = cpt[cpt.depth_m <= 12.0]
    q10 = float(shallow.qc_mpa.quantile(0.10))
    ic_med = float(shallow.ic_proxy.median())
    theta = fit_exponential_variogram(cpt.depth_m.iloc[::5].to_numpy(), np.log(np.clip(cpt.qc_mpa.iloc[::5].to_numpy(), 1e-3, None)))["theta_z_m"]
    rows = []
    for r in events.itertuples():
        rows.append(
            {
                "site_id": r.site_id,
                "year": r.year,
                "event_name": r.event_name,
                "event_date": r.event_date,
                "pga_g": r.pga_g,
                "mw": r.mw,
                "wtd_event_site_adjusted_m": r.gw_depth_m,
                "qc10_mpa": q10,
                "ic_median": ic_med,
                "theta_z_m": theta,
                "observed_liquefaction": r.observed_liquefaction,
            }
        )
    canterbury = pd.DataFrame(rows)
    models = {
        "M0_static_stationary": ["pga_g"],
        "M1_nonstationary_groundwater_only": ["pga_g", "wtd_event_site_adjusted_m"],
        "M2_nonstationary_groundwater_gradation": ["pga_g", "qc10_mpa", "wtd_event_site_adjusted_m", "ic_median"],
        "M3_full_nonstationary_random_field": ["pga_g", "qc10_mpa", "wtd_event_site_adjusted_m", "ic_median", "theta_z_m"],
    }
    pred_rows = []
    metrics = []
    for name, cols in models.items():
        params = fit_logistic(nisqually[cols].to_numpy(), nisqually.observed_liquefaction.to_numpy(), l2=0.1)
        prob = predict_logistic(params, canterbury[cols].to_numpy())
        m = binary_metrics(prob, canterbury.observed_liquefaction.to_numpy())
        m.update({"model_name": name, "n_train": int(len(nisqually)), "n_test": int(len(canterbury)), "validation_protocol": "Nisqually-trained temporal transfer to Canterbury 100 Osbourne CPT01"})
        metrics.append(m)
        for row, p in zip(canterbury.itertuples(), prob):
            pred_rows.append(
                {
                    "site_id": row.site_id,
                    "year": row.year,
                    "event_name": row.event_name,
                    "event_date": row.event_date,
                    "model_name": name,
                    "predicted_pf": float(p),
                    "observed_liquefaction": int(row.observed_liquefaction),
                    "pga_g": row.pga_g,
                    "mw": row.mw,
                    "gw_depth_m": row.wtd_event_site_adjusted_m,
                    "qc10_mpa": row.qc10_mpa,
                    "ic_median": row.ic_median,
                    "theta_z_m": row.theta_z_m,
                    "validation_protocol": "Nisqually-trained temporal transfer to Canterbury 100 Osbourne CPT01",
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(metrics)


def plot_temporal(pred: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1100, 660), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 22)
    small = ImageFont.truetype("arial.ttf", 18)
    left, right, top, bottom = 90, 1000, 90, 540
    d.text((50, 32), "Canterbury temporal site application: Pf(t) over documented events", font=font, fill=(0, 0, 0))
    d.rectangle((left, top, right, bottom), outline=(40, 40, 40), width=2)
    years = sorted(pred.year.unique())
    xmap = {y: left + i * (right - left) / (len(years) - 1) for i, y in enumerate(years)}
    colors = {
        "M0_static_stationary": (80, 80, 80),
        "M1_nonstationary_groundwater_only": (45, 120, 170),
        "M2_nonstationary_groundwater_gradation": (30, 140, 90),
        "M3_full_nonstationary_random_field": (150, 70, 140),
    }

    def ymap(p):
        return bottom - float(p) * (bottom - top)

    for name, group in pred.groupby("model_name"):
        pts = [(xmap[int(r.year)], ymap(r.predicted_pf)) for r in group.sort_values("year").itertuples()]
        d.line(pts, fill=colors[name], width=4)
        for x, y in pts:
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[name])
    obs = pred.drop_duplicates("year").sort_values("year")
    for r in obs.itertuples():
        x = xmap[int(r.year)]
        d.text((x - 28, bottom + 18), str(int(r.year)), font=small, fill=(0, 0, 0))
        if r.observed_liquefaction:
            d.rectangle((x - 8, top - 32, x + 8, top - 16), fill=(180, 40, 40))
        else:
            d.ellipse((x - 8, top - 32, x + 8, top - 16), outline=(40, 40, 40), width=3)
    yleg = 115
    for name, color in colors.items():
        d.line((720, yleg, 760, yleg), fill=color, width=4)
        d.text((770, yleg - 10), name.replace("_", " "), font=small, fill=(0, 0, 0))
        yleg += 34
    d.text((left, bottom + 55), "Red square marks the liquefaction event; open circles mark non-manifestation events.", font=small, fill=(0, 0, 0))
    img.save(path)


def main() -> None:
    cpt, profile, events = load_canterbury_site()
    pred, metrics = evaluate_temporal_transfer(cpt, events)
    cpt.to_csv(DATA / "canterbury_100_osbourne_cpt01_profile_points.csv", index=False)
    profile.to_csv(DATA / "canterbury_100_osbourne_site_profile_calibrated.csv", index=False)
    events[["site_id", "event_date", "gw_depth_m", "data_source"]].rename(columns={"event_date": "date"}).to_csv(DATA / "canterbury_100_osbourne_groundwater_timeseries.csv", index=False)
    events.to_csv(DATA / "canterbury_100_osbourne_event_observations.csv", index=False)
    pred.to_csv(OUTPUTS / "canterbury_temporal_prediction_probabilities.csv", index=False)
    metrics.to_csv(OUTPUTS / "canterbury_temporal_validation_metrics.csv", index=False)
    plot_temporal(pred, FIGURES / "fig09_canterbury_temporal_pf.png")
    print(metrics[["model_name", "auc", "brier_score", "log_loss", "calibration_intercept", "calibration_slope", "sensitivity", "specificity"]].to_string(index=False))


if __name__ == "__main__":
    main()
