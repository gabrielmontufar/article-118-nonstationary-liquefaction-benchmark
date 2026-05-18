"""Explore honest out-of-sample protocols for the Nisqually extension.

The point of this script is diagnostic. It keeps each split rule explicit and
reports negative results instead of searching for a favorable split.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_119_site_calibrated_application import EVENT_DATE, _site_features, build_site_tables
from src.groundwater_calibration import fit_groundwater_model
from src.validation_metrics import auc_rank, fit_logistic, predict_logistic

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

MODELS = {
    "M0_static_stationary": ["pga_g"],
    "M1_groundwater": ["pga_g", "wtd_event_site_adjusted_m"],
    "M2_groundwater_gradation": ["pga_g", "qc10_mpa", "wtd_event_site_adjusted_m", "ic_median"],
    "M3_random_field": ["pga_g", "qc10_mpa", "wtd_event_site_adjusted_m", "ic_median", "theta_z_m"],
}


def _region(row: pd.Series) -> str:
    name = str(row["site_family"])
    if name in {"Capitol Lake", "Green River", "Emerald Downs", "Blair Waterway"}:
        return "south_sound_or_green_river"
    if name in {"Discovery Park", "Elliott Ave", "Smith Cove", "UW"}:
        return "north_seattle"
    if name in {"Harbor Island", "SODO", "Duwamish", "Boeing Field"}:
        return "industrial_waterway"
    return "eastside"


def _safe_metrics(prob: np.ndarray, y: np.ndarray) -> dict:
    p = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    valid = np.isfinite(p)
    p = np.clip(p[valid], 1e-6, 1 - 1e-6)
    y = y[valid]
    out = {
        "n_test": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else math.nan,
        "auc": auc_rank(p, y) if len(np.unique(y)) == 2 else math.nan,
        "brier_score": float(np.mean((p - y) ** 2)) if len(y) else math.nan,
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else math.nan,
        "mean_predicted_probability": float(p.mean()) if len(y) else math.nan,
    }
    if len(y) >= 6 and len(np.unique(y)) == 2 and len(np.unique(np.round(p, 10))) > 1:
        logit = np.log(p / (1 - p))[:, None]
        pars = fit_logistic(logit, y)
        out["calibration_intercept"] = float(pars[0])
        out["calibration_slope"] = float(pars[1])
    else:
        out["calibration_intercept"] = math.nan
        out["calibration_slope"] = math.nan
    return out


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    if len(train) <= len(columns) + 1 or train["observed_liquefaction"].nunique() < 2:
        return None
    try:
        params = fit_logistic(train[columns].to_numpy(), train["observed_liquefaction"].to_numpy())
        return predict_logistic(params, test[columns].to_numpy())
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return None


def _evaluate_splits(features: pd.DataFrame, protocol: str, split_col: str, eligible_groups: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    if eligible_groups is None:
        eligible_groups = list(features[split_col].dropna().unique())
    for heldout in eligible_groups:
        test = features[features[split_col].eq(heldout)].copy()
        train = features[~features[split_col].eq(heldout)].copy()
        if test.empty:
            continue
        for model_name, cols in MODELS.items():
            prob = _fit_predict(train, test, cols)
            status = "ok" if prob is not None else "skipped_insufficient_training_data"
            if prob is None:
                prob = np.full(len(test), np.nan)
            for row, p in zip(test.itertuples(index=False), prob):
                pred_rows.append(
                    {
                        "protocol": protocol,
                        "heldout_group": heldout,
                        "model_name": model_name,
                        "site_id": row.site_id,
                        "site_name": row.site_name,
                        "site_family": row.site_family,
                        "region": row.region,
                        "date_period": row.date_period,
                        "test_date": row.test_date,
                        "observed_liquefaction": int(row.observed_liquefaction),
                        "predicted_pf": float(p) if np.isfinite(p) else np.nan,
                        "status": status,
                    }
                )
    pred = pd.DataFrame(pred_rows)
    metric_rows = []
    for (protocol_name, model_name), g in pred[pred.status.eq("ok")].groupby(["protocol", "model_name"]):
        m = _safe_metrics(g["predicted_pf"].to_numpy(), g["observed_liquefaction"].to_numpy())
        m.update({"protocol": protocol_name, "model_name": model_name, "n_groups": int(g["heldout_group"].nunique())})
        metric_rows.append(m)
    metrics = pd.DataFrame(metric_rows)
    return pred, metrics


def _evaluate_temporal(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = features[features["date_period"].eq("pre_event")]
    post = features[features["date_period"].eq("post_event")]
    rows = []
    split_specs = [
        ("temporal_pre_train_post_test", pre, post, "post_event"),
        ("temporal_post_train_pre_test", post, pre, "pre_event"),
    ]
    for protocol, train, test, heldout in split_specs:
        for model_name, cols in MODELS.items():
            prob = _fit_predict(train, test, cols)
            status = "ok" if prob is not None else "skipped_insufficient_training_data"
            if protocol == "temporal_post_train_pre_test":
                status = "skipped_not_defensible_only_two_post_event_training_cases"
                prob = None
            if prob is None:
                prob = np.full(len(test), np.nan)
            for row, p in zip(test.itertuples(index=False), prob):
                rows.append(
                    {
                        "protocol": protocol,
                        "heldout_group": heldout,
                        "model_name": model_name,
                        "site_id": row.site_id,
                        "site_name": row.site_name,
                        "site_family": row.site_family,
                        "region": row.region,
                        "date_period": row.date_period,
                        "test_date": row.test_date,
                        "observed_liquefaction": int(row.observed_liquefaction),
                        "predicted_pf": float(p) if np.isfinite(p) else np.nan,
                        "status": status,
                    }
                )
    pred = pd.DataFrame(rows)
    metric_rows = []
    ok = pred[pred.status.eq("ok")]
    for (protocol, model_name), g in ok.groupby(["protocol", "model_name"]):
        m = _safe_metrics(g["predicted_pf"].to_numpy(), g["observed_liquefaction"].to_numpy())
        m.update({"protocol": protocol, "model_name": model_name, "n_groups": int(g["heldout_group"].nunique())})
        metric_rows.append(m)
    return pred, pd.DataFrame(metric_rows)


def _nested_family_cv(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    families = sorted(features["site_family"].unique())
    for outer in families:
        outer_train = features[~features.site_family.eq(outer)].copy()
        outer_test = features[features.site_family.eq(outer)].copy()
        inner_pred, inner_metrics = _evaluate_splits(outer_train, "inner_leave_one_family_out", "site_family")
        if inner_metrics.empty:
            continue
        ranked = inner_metrics.sort_values(["brier_score", "log_loss", "model_name"], na_position="last")
        selected = str(ranked.iloc[0]["model_name"])
        prob = _fit_predict(outer_train, outer_test, MODELS[selected])
        status = "ok" if prob is not None else "skipped_insufficient_training_data"
        if prob is None:
            prob = np.full(len(outer_test), np.nan)
        for row, p in zip(outer_test.itertuples(index=False), prob):
            rows.append(
                {
                    "protocol": "nested_family_cv_select_by_inner_brier",
                    "heldout_group": outer,
                    "model_name": selected,
                    "site_id": row.site_id,
                    "site_name": row.site_name,
                    "site_family": row.site_family,
                    "region": row.region,
                    "date_period": row.date_period,
                    "test_date": row.test_date,
                    "observed_liquefaction": int(row.observed_liquefaction),
                    "predicted_pf": float(p) if np.isfinite(p) else np.nan,
                    "status": status,
                    "selected_model": selected,
                }
            )
    pred = pd.DataFrame(rows)
    metrics = pd.DataFrame()
    ok = pred[pred.status.eq("ok")]
    if not ok.empty:
        m = _safe_metrics(ok["predicted_pf"].to_numpy(), ok["observed_liquefaction"].to_numpy())
        m.update(
            {
                "protocol": "nested_family_cv_select_by_inner_brier",
                "model_name": "selected_per_outer_fold",
                "n_groups": int(ok["heldout_group"].nunique()),
            }
        )
        metrics = pd.DataFrame([m])
    return pred, metrics


def _fixed_sodo_uw(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = features.copy()
    df["fixed_sodo_uw"] = np.where(df["site_family"].isin(["SODO", "UW"]), "heldout_SODO_UW", "train_development")
    pred_rows = []
    train = df[df.fixed_sodo_uw.eq("train_development")]
    test = df[df.fixed_sodo_uw.eq("heldout_SODO_UW")]
    for model_name, cols in MODELS.items():
        prob = _fit_predict(train, test, cols)
        status = "ok" if prob is not None else "skipped_insufficient_training_data"
        if prob is None:
            prob = np.full(len(test), np.nan)
        for row, p in zip(test.itertuples(index=False), prob):
            pred_rows.append(
                {
                    "protocol": "fixed_family_holdout_SODO_UW",
                    "heldout_group": "SODO_UW",
                    "model_name": model_name,
                    "site_id": row.site_id,
                    "site_name": row.site_name,
                    "site_family": row.site_family,
                    "region": row.region,
                    "date_period": row.date_period,
                    "test_date": row.test_date,
                    "observed_liquefaction": int(row.observed_liquefaction),
                    "predicted_pf": float(p) if np.isfinite(p) else np.nan,
                    "status": status,
                }
            )
    pred = pd.DataFrame(pred_rows)
    metrics = []
    for (protocol, model_name), g in pred[pred.status.eq("ok")].groupby(["protocol", "model_name"]):
        m = _safe_metrics(g["predicted_pf"].to_numpy(), g["observed_liquefaction"].to_numpy())
        m.update({"protocol": protocol, "model_name": model_name, "n_groups": 2})
        metrics.append(m)
    return pred, pd.DataFrame(metrics)


def _best_worst(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, g in metrics.groupby("protocol"):
        ordered = g.sort_values("brier_score", na_position="last")
        for rank, row in zip(["best", "worst"], [ordered.iloc[0], ordered.iloc[-1]]):
            out = row.to_dict()
            out["rank_by_brier"] = rank
            rows.append(out)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in work.to_numpy(dtype=str)]
    return "\n".join([header, sep, *body])


def _write_report(metrics: pd.DataFrame, best_worst: pd.DataFrame, path: Path) -> None:
    pivot = metrics.pivot_table(index="protocol", columns="model_name", values="brier_score", aggfunc="first")
    improvements = []
    for protocol, row in pivot.iterrows():
        m0 = row.get("M0_static_stationary", np.nan)
        for model in ["M1_groundwater", "M2_groundwater_gradation"]:
            value = row.get(model, np.nan)
            if np.isfinite(m0) and np.isfinite(value):
                improvements.append(
                    {
                        "protocol": protocol,
                        "model_name": model,
                        "brier_score": value,
                        "m0_brier_score": m0,
                        "delta_brier_vs_m0": value - m0,
                        "improves_m0": bool(value < m0),
                    }
                )
    imp = pd.DataFrame(improvements)
    defensible = imp[imp.improves_m0] if not imp.empty else pd.DataFrame()
    with path.open("w", encoding="utf-8") as f:
        f.write("# Nisqually out-of-sample protocol exploration\n\n")
        f.write("This file is generated by `scripts/explore_nisqually_out_of_sample_protocols.py`.\n\n")
        f.write("## Best and worst by protocol\n\n")
        f.write(_markdown_table(best_worst[["protocol", "rank_by_brier", "model_name", "n_test", "auc", "brier_score", "log_loss", "calibration_intercept", "calibration_slope"]]))
        f.write("\n\n## M1/M2 Brier comparison against M0\n\n")
        if imp.empty:
            f.write("No protocols had comparable M0, M1 and M2 Brier scores.\n")
        else:
            f.write(_markdown_table(imp))
            f.write("\n\n")
            if defensible.empty:
                f.write("Conclusion: no tested out-of-sample protocol gives M1 or M2 a lower Brier score than M0.\n")
            else:
                f.write("Protocols where M1 or M2 improves Brier relative to M0:\n\n")
                f.write(_markdown_table(defensible))
        f.write("\n\n## Protocol notes\n\n")
        f.write("- Leave-one-family-out and leave-one-region-out are spatially honest, but some held-out families/regions are single-class, so AUC is interpreted only after pooling cross-fitted predictions.\n")
        f.write("- The temporal pre/post split has only two post-event CPT cases; it is reported as a stress test and not a strong validation claim.\n")
        f.write("- The reverse temporal split is skipped because training on only two post-event cases is not defensible.\n")
        f.write("- Nested family CV selects model form inside the training families only, then evaluates the selected model on the outer held-out family.\n")


def main() -> None:
    summary, cpt_all, _profile, gw, _grad, _events = build_site_tables()
    gw_params, _gw_fit = fit_groundwater_model(gw, EVENT_DATE)
    features = _site_features(summary, cpt_all, gw_params)
    features["site_family"] = features["site_name"].str.replace(r"-\d+$", "", regex=True)
    features["region"] = features.apply(_region, axis=1)
    features["test_date"] = pd.to_datetime(features["test_date"])
    features["date_period"] = np.where(features["test_date"] <= pd.Timestamp(EVENT_DATE), "pre_event", "post_event")

    pred_parts = []
    metric_parts = []
    for pred, metrics in [
        _fixed_sodo_uw(features),
        _evaluate_splits(features, "leave_one_family_out", "site_family"),
        _evaluate_splits(features, "leave_one_region_out", "region"),
        _evaluate_temporal(features),
        _nested_family_cv(features),
    ]:
        pred_parts.append(pred)
        metric_parts.append(metrics)

    predictions = pd.concat(pred_parts, ignore_index=True)
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics = metrics.sort_values(["protocol", "brier_score", "model_name"], na_position="last")
    best_worst = _best_worst(metrics)

    predictions.to_csv(OUTPUTS / "nisqually_protocol_exploration_predictions.csv", index=False)
    metrics.to_csv(OUTPUTS / "nisqually_protocol_exploration_metrics.csv", index=False)
    best_worst.to_csv(OUTPUTS / "nisqually_protocol_exploration_best_worst.csv", index=False)
    _write_report(metrics, best_worst, OUTPUTS / "nisqually_protocol_exploration_report.md")

    print(best_worst[["protocol", "rank_by_brier", "model_name", "n_test", "auc", "brier_score", "log_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
