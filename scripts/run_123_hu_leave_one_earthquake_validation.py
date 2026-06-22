"""Leave-one-earthquake-out validation on Hu et al. case histories.

This script strengthens the external evidence gate without changing the
manuscript's claim boundary. It calibrates simple screening scores on all but
one earthquake and predicts the held-out earthquake. The result is an
event-level external validation diagnostic, not site-specific design
validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation_metrics import auc_rank, binary_metrics, fit_logistic, predict_logistic


DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)


CANDIDATES = [
    {
        "model": "combined_instrument_score",
        "score_column": "combined_instrument_score",
        "source": "Hu full set screening score",
        "orientation": "higher_more_likely_liquefaction",
    },
    {
        "model": "dpt_screening_score",
        "score_column": "dpt_screening_score",
        "source": "Hu full set DPT/N120 score",
        "orientation": "higher_more_likely_liquefaction",
    },
    {
        "model": "vs_screening_score",
        "score_column": "vs_screening_score",
        "source": "Hu full set Vs score",
        "orientation": "higher_more_likely_liquefaction",
    },
    {
        "model": "static_limit_state_pf_proxy",
        "score_column": "pf_static_proxy",
        "source": "static benchmark proxy subset",
        "orientation": "higher_more_likely_liquefaction",
    },
]


def _fit_predict_score(train: pd.DataFrame, test: pd.DataFrame, score_col: str) -> np.ndarray:
    x_train = train[[score_col]].to_numpy(dtype=float)
    y_train = train["observed_liquefaction"].to_numpy(dtype=float)
    params = fit_logistic(x_train, y_train, l2=1.0)
    return predict_logistic(params, test[[score_col]].to_numpy(dtype=float))


def main() -> None:
    cases = pd.read_csv(DATA / "external_case_history_compatibility_cases.csv")
    cases = cases.rename(columns={"observed_liquefaction": "observed_liquefaction"})

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    pooled_rows: list[dict] = []

    for spec in CANDIDATES:
        score_col = spec["score_column"]
        df = cases.dropna(subset=[score_col, "observed_liquefaction", "earthquake"]).copy()
        df["observed_liquefaction"] = df["observed_liquefaction"].astype(int)

        pred_parts: list[pd.DataFrame] = []
        for event in sorted(df["earthquake"].dropna().unique()):
            test = df[df["earthquake"].eq(event)].copy()
            train = df[~df["earthquake"].eq(event)].copy()
            if len(test) < 2 or train["observed_liquefaction"].nunique() < 2:
                continue
            pred = _fit_predict_score(train, test, score_col)
            part = test[
                [
                    "case_id",
                    "earthquake",
                    "site",
                    "mw",
                    "pga",
                    "groundwater_table_m",
                    "critical_layer_depth_m",
                    "observed_liquefaction",
                    score_col,
                ]
            ].copy()
            part["model"] = spec["model"]
            part["predicted_probability"] = pred
            part["heldout_earthquake"] = event
            pred_parts.append(part)

            y = part["observed_liquefaction"].to_numpy(dtype=int)
            p = part["predicted_probability"].to_numpy(dtype=float)
            raw = part[score_col].to_numpy(dtype=float)
            metrics = binary_metrics(p, y)
            fold_rows.append(
                {
                    "model": spec["model"],
                    "heldout_earthquake": event,
                    "n_test": int(len(part)),
                    "n_liq": int(y.sum()),
                    "n_nonliq": int(len(y) - y.sum()),
                    "raw_score_auc": auc_rank(raw, y),
                    **metrics,
                }
            )

        if not pred_parts:
            continue

        preds = pd.concat(pred_parts, ignore_index=True)
        prediction_frames.append(preds)
        y = preds["observed_liquefaction"].to_numpy(dtype=int)
        p = preds["predicted_probability"].to_numpy(dtype=float)
        raw = preds[score_col].to_numpy(dtype=float)
        metrics = binary_metrics(p, y)
        pooled_rows.append(
            {
                "model": spec["model"],
                "source": spec["source"],
                "n_test": int(len(preds)),
                "n_earthquakes": int(preds["heldout_earthquake"].nunique()),
                "n_liq": int(y.sum()),
                "n_nonliq": int(len(y) - y.sum()),
                "raw_score_auc": auc_rank(raw, y),
                **metrics,
            }
        )

    if not prediction_frames:
        raise RuntimeError("No leave-one-earthquake predictions were generated")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    pooled_metrics = pd.DataFrame(pooled_rows).sort_values("brier_score")

    predictions.to_csv(OUTPUTS / "hu_leave_one_earthquake_predictions.csv", index=False)
    fold_metrics.to_csv(OUTPUTS / "hu_leave_one_earthquake_fold_metrics.csv", index=False)
    pooled_metrics.to_csv(OUTPUTS / "hu_leave_one_earthquake_metrics.csv", index=False)

    best = pooled_metrics.iloc[0].to_dict()
    summary = {
        "status": "PASS_HU_LEAVE_ONE_EARTHQUAKE_VALIDATION",
        "source": "data/external_case_history_compatibility_cases.csv",
        "protocol": "leave one earthquake out; calibrate on all other earthquakes; predict held-out earthquake",
        "candidate_models": [s["model"] for s in CANDIDATES],
        "best_brier_model": best["model"],
        "best_brier": float(best["brier_score"]),
        "best_auc": float(best["auc"]),
        "best_raw_score_auc": float(best["raw_score_auc"]),
        "n_predictions_best_model": int(best["n_test"]),
        "n_earthquakes_best_model": int(best["n_earthquakes"]),
        "claim_enabled": "event-level external validation support for bounded screening scores across held-out earthquakes",
        "claim_blocked": "do not describe this as site-specific design validation or validation of the non-stationary time-update operator",
    }
    (OUTPUTS / "hu_leave_one_earthquake_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Hu leave-one-earthquake validation - 2026-06-22",
        "",
        "## Protocol",
        "",
        "Each earthquake in the Hu et al. external case-history table is held out in turn. Screening scores are calibrated on all remaining earthquakes and then predicted on the held-out event.",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        f"Best Brier model: `{summary['best_brier_model']}`.",
        f"Best Brier: `{summary['best_brier']:.6f}`.",
        f"Best AUC: `{summary['best_auc']:.6f}`.",
        f"Predictions in best model: `{summary['n_predictions_best_model']}` across `{summary['n_earthquakes_best_model']}` held-out earthquakes.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}.",
        "",
        f"Blocked: {summary['claim_blocked']}.",
    ]
    (AUDIT / "hu_leave_one_earthquake_validation_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
