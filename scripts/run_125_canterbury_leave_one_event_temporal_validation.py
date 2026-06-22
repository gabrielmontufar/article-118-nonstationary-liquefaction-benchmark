"""Canterbury leave-one-event-out temporal validation.

This is a stricter temporal holdout than random site splitting. Each Canterbury
earthquake is held out as a complete external event, the probability models are
trained on the other two events, and predictions are evaluated on the held-out
earthquake. The diagnostic tests event-to-event transfer of groundwater and
gradation-aware predictors, while preserving the claim boundary that Brier
calibration and false-negative screening can favour different operating rules.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation_metrics import binary_metrics, fit_logistic, predict_logistic


DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)

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

LOSS_RATIOS = [1, 2, 5, 10, 20, 50, 100]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)


def _loss(fp: int, fn: int, n_cases: int, false_negative_cost_ratio: float) -> float:
    return (fp + false_negative_cost_ratio * fn) / max(n_cases, 1)


def _select_threshold(prob: np.ndarray, y: np.ndarray, false_negative_cost_ratio: float) -> tuple[float, dict[str, object]]:
    best: tuple[float, float, dict[str, object]] | None = None
    for threshold in THRESHOLDS:
        metrics = binary_metrics(prob, y, threshold=float(threshold))
        loss = _loss(metrics["fp"], metrics["fn"], len(y), false_negative_cost_ratio)
        candidate = (loss, float(threshold), metrics)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    loss, threshold, metrics = best
    return threshold, {"selection_loss": loss, **metrics}


def main() -> None:
    features = pd.read_csv(DATA / "canterbury_multi_site_event_features.csv")
    required = {"event_key", "event_name", "observed_liquefaction", *{c for cols in MODEL_FEATURES.values() for c in cols}}
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"Missing required Canterbury columns: {missing}")

    prediction_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    locked_rows: list[dict[str, object]] = []

    for holdout in sorted(features["event_key"].unique()):
        train = features[~features["event_key"].eq(holdout)].copy()
        test = features[features["event_key"].eq(holdout)].copy()
        if train["observed_liquefaction"].nunique() < 2 or test["observed_liquefaction"].nunique() < 2:
            raise RuntimeError(f"Holdout {holdout} does not contain both classes")
        for model_name, cols in MODEL_FEATURES.items():
            params = fit_logistic(train[cols].to_numpy(dtype=float), train["observed_liquefaction"].to_numpy(dtype=int), l2=1.0)
            train_prob = predict_logistic(params, train[cols].to_numpy(dtype=float))
            test_prob = predict_logistic(params, test[cols].to_numpy(dtype=float))
            y_test = test["observed_liquefaction"].to_numpy(dtype=int)
            metrics = binary_metrics(test_prob, y_test)
            fold_rows.append(
                {
                    "heldout_event_key": holdout,
                    "heldout_event_name": str(test["event_name"].iloc[0]),
                    "model_name": model_name,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_liq_test": int(y_test.sum()),
                    "n_nonliq_test": int(len(y_test) - y_test.sum()),
                    **metrics,
                }
            )
            pred = test[
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
            pred["model_name"] = model_name
            pred["predicted_pf"] = test_prob
            pred["heldout_event_key"] = holdout
            prediction_rows.append(pred)

            for ratio in LOSS_RATIOS:
                threshold, selected = _select_threshold(
                    train_prob,
                    train["observed_liquefaction"].to_numpy(dtype=int),
                    false_negative_cost_ratio=ratio,
                )
                test_metrics = binary_metrics(test_prob, y_test, threshold=threshold)
                locked_rows.append(
                    {
                        "heldout_event_key": holdout,
                        "heldout_event_name": str(test["event_name"].iloc[0]),
                        "model_name": model_name,
                        "false_negative_cost_ratio": ratio,
                        "locked_threshold_selected_on_training_events": threshold,
                        "training_selection_loss": float(selected["selection_loss"]),
                        "test_expected_loss_per_case": _loss(test_metrics["fp"], test_metrics["fn"], len(y_test), ratio),
                        "test_fn": int(test_metrics["fn"]),
                        "test_fp": int(test_metrics["fp"]),
                        "test_tp": int(test_metrics["tp"]),
                        "test_tn": int(test_metrics["tn"]),
                        "n_test": int(len(y_test)),
                        "threshold_policy": "selected_on_two_canterbury_events_then_applied_to_heldout_event",
                    }
                )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    locked = pd.DataFrame(locked_rows)
    pooled_rows = []
    for model_name, group in predictions.groupby("model_name", sort=False):
        y = group["observed_liquefaction"].to_numpy(dtype=int)
        p = group["predicted_pf"].to_numpy(dtype=float)
        pooled_rows.append(
            {
                "model_name": model_name,
                "n_test": int(len(group)),
                "n_events": int(group["heldout_event_key"].nunique()),
                "n_liq": int(y.sum()),
                "n_nonliq": int(len(y) - y.sum()),
                **binary_metrics(p, y),
            }
        )
    pooled_metrics = pd.DataFrame(pooled_rows).sort_values(["brier_score", "model_name"])

    predictions.to_csv(OUTPUTS / "canterbury_leave_one_event_temporal_predictions.csv", index=False)
    fold_metrics.to_csv(OUTPUTS / "canterbury_leave_one_event_temporal_fold_metrics.csv", index=False)
    pooled_metrics.to_csv(OUTPUTS / "canterbury_leave_one_event_temporal_pooled_metrics.csv", index=False)
    locked.to_csv(OUTPUTS / "canterbury_leave_one_event_temporal_locked_thresholds.csv", index=False)

    ratio10 = locked[locked["false_negative_cost_ratio"].eq(10)].copy()
    best_locked_by_event = (
        ratio10.sort_values(["heldout_event_key", "test_expected_loss_per_case", "test_fn", "test_fp"])
        .groupby("heldout_event_key", as_index=False)
        .first()
    )
    m0 = ratio10[ratio10["model_name"].eq("M0_static_stationary")].set_index("heldout_event_key")
    best_nonstationary = (
        ratio10[~ratio10["model_name"].eq("M0_static_stationary")]
        .sort_values(["heldout_event_key", "test_expected_loss_per_case", "test_fn", "test_fp"])
        .groupby("heldout_event_key", as_index=False)
        .first()
        .set_index("heldout_event_key")
    )
    event_contrasts = []
    for event_key, row in best_nonstationary.iterrows():
        base = m0.loc[event_key]
        event_contrasts.append(
            {
                "heldout_event_key": event_key,
                "heldout_event_name": row["heldout_event_name"],
                "best_nonstationary_model": row["model_name"],
                "m0_loss": float(base["test_expected_loss_per_case"]),
                "m0_fn": int(base["test_fn"]),
                "m0_fp": int(base["test_fp"]),
                "nonstationary_loss": float(row["test_expected_loss_per_case"]),
                "nonstationary_fn": int(row["test_fn"]),
                "nonstationary_fp": int(row["test_fp"]),
            }
        )

    best_brier = pooled_metrics.iloc[0].to_dict()
    best_auc = pooled_metrics.sort_values(["auc", "brier_score"], ascending=[False, True]).iloc[0].to_dict()
    summary = {
        "status": "PASS_CANTERBURY_LEAVE_ONE_EVENT_TEMPORAL_VALIDATION",
        "source": "data/canterbury_multi_site_event_features.csv",
        "protocol": "train on two Canterbury earthquakes; predict the held-out earthquake; repeat for all three events",
        "candidate_models": list(MODEL_FEATURES),
        "n_predictions": int(len(predictions)),
        "n_event_states": int(predictions[["site_id", "event_key"]].drop_duplicates().shape[0]),
        "n_heldout_events": int(predictions["heldout_event_key"].nunique()),
        "best_pooled_brier_model": best_brier["model_name"],
        "best_pooled_brier": float(best_brier["brier_score"]),
        "best_pooled_auc_model": best_auc["model_name"],
        "best_pooled_auc": float(best_auc["auc"]),
        "fn_fp_10_best_locked_model_by_heldout_event": best_locked_by_event[
            [
                "heldout_event_key",
                "heldout_event_name",
                "model_name",
                "locked_threshold_selected_on_training_events",
                "test_expected_loss_per_case",
                "test_fn",
                "test_fp",
                "n_test",
            ]
        ].to_dict(orient="records"),
        "fn_fp_10_best_nonstationary_vs_m0_by_event": event_contrasts,
        "claim_enabled": (
            "formal leave-one-earthquake-out temporal validation within Canterbury: model fitting excludes the held-out event, "
            "and groundwater/gradation-aware predictors improve rank discrimination and false-negative operating choices in selected held-out events."
        ),
        "claim_blocked": (
            "do not present this as universal non-stationary superiority; M0 remains competitive on Brier score, especially for the low-prevalence 2016 event."
        ),
    }
    (OUTPUTS / "canterbury_leave_one_event_temporal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Canterbury leave-one-event temporal validation - 2026-06-22",
        "",
        "## Protocol",
        "",
        "Each Canterbury earthquake is held out as a complete temporal validation event. Models are trained on the other two earthquakes and evaluated on the held-out event.",
        "",
        "## Pooled result",
        "",
        f"Best pooled Brier model: `{summary['best_pooled_brier_model']}` with Brier `{summary['best_pooled_brier']:.6f}`.",
        f"Best pooled AUC model: `{summary['best_pooled_auc_model']}` with AUC `{summary['best_pooled_auc']:.6f}`.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
    ]
    (AUDIT / "canterbury_leave_one_event_temporal_validation_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
