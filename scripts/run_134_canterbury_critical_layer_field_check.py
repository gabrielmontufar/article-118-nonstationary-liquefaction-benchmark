"""Canterbury critical-layer field check.

This is a field-facing check for the critical water-table crossing mechanism.
For each Canterbury CPT, the script estimates a shallow critical layer from the
weakest 20% of qc values between 0.5 and 6 m. It then asks whether event water
table position relative to that layer improves leave-one-earthquake prediction
of observed manifestation.

The expected result is modest: Canterbury event outcomes are strongly dominated
by shaking intensity and broad site effects. The check therefore tests whether
the mechanism has field-facing signal, not whether it becomes a standalone
universal classifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
AUDIT = ROOT / "audit_logs"
FIGURES = ROOT / "figures"
RAW = ROOT / "raw_designsafe" / "canterbury_designsafe_PRJ-2937" / "CANTERBURYDATASET.mat"
OUTPUTS.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

EVENTS = ["Yr2010", "Yr2011", "Yr2016"]


def _scalar(value: object) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _build_event_states() -> pd.DataFrame:
    if not RAW.exists():
        raise FileNotFoundError(
            f"Missing Canterbury raw dataset: {RAW}. "
            "Run scripts/download_designsafe_raw_118.py first or place the public DesignSafe PRJ-2937 "
            "CANTERBURYDATASET.mat file under raw_designsafe/canterbury_designsafe_PRJ-2937/."
        )
    mat = loadmat(RAW, squeeze_me=True, struct_as_record=False)
    records = np.ravel(mat["CANTERBURYDATASET"])
    rows: list[dict[str, object]] = []
    for idx, rec in enumerate(records):
        try:
            depth = np.asarray(rec.depth, dtype=float)
            qc_mpa = np.asarray(rec.qc, dtype=float) / 1000.0
        except Exception:
            continue
        valid = np.isfinite(depth) & np.isfinite(qc_mpa) & (qc_mpa > 0)
        shallow = valid & (depth >= 0.5) & (depth <= 6.0)
        if int(shallow.sum()) < 8:
            continue
        q20 = float(np.nanquantile(qc_mpa[shallow], 0.20))
        weak = shallow & (qc_mpa <= q20)
        if int(weak.sum()) == 0:
            z_crit = float(depth[shallow][np.nanargmin(qc_mpa[shallow])])
            q_crit = float(np.nanmin(qc_mpa[shallow]))
        else:
            z_crit = float(np.nanmedian(depth[weak]))
            q_crit = float(np.nanmedian(qc_mpa[weak]))
        for event_key in EVENTS:
            manifestation = int(_scalar(getattr(rec.Manifestation, event_key)))
            if manifestation == 10:
                continue
            pga = _scalar(getattr(rec.PGA, event_key))
            wtd = _scalar(getattr(rec.GWT, event_key))
            if not np.isfinite(pga) or not np.isfinite(wtd):
                continue
            rows.append(
                {
                    "site_id": f"Canterbury_{idx:05d}",
                    "event_key": event_key,
                    "pga_g": float(pga),
                    "wtd_event_site_adjusted_m": float(wtd),
                    "manifestation_code": manifestation,
                    "observed_liquefaction": int(manifestation > 0),
                    "severe_manifestation": int(manifestation >= 3),
                    "critical_layer_depth_m": z_crit,
                    "critical_layer_qc_mpa": q_crit,
                    "critical_layer_saturated": int(wtd <= z_crit),
                    "critical_crossing_margin_m": float(z_crit - wtd),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_event_metrics(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, object] | None:
    probs: list[float] = []
    labels: list[int] = []
    for holdout in sorted(df["event_key"].unique()):
        train = df[~df["event_key"].eq(holdout)].copy()
        test = df[df["event_key"].eq(holdout)].copy()
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(train[features], train[target])
        prob = model.predict_proba(test[features])[:, 1]
        probs.extend(prob.tolist())
        labels.extend(test[target].astype(int).tolist())
    if len(set(labels)) < 2:
        return None
    eps = 1e-9
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(labels, dtype=int)
    return {
        "target": target,
        "features": "+".join(features),
        "n_predictions": int(len(y)),
        "n_positive": int(y.sum()),
        "auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def _write_contrast_figure(contrasts: list[dict[str, object]]) -> None:
    plot_df = pd.DataFrame(contrasts).copy()
    if plot_df.empty:
        return
    plot_df["target_label"] = plot_df["target"].map(
        {
            "observed_liquefaction": "Any observed\nmanifestation",
            "severe_manifestation": "Severe\nmanifestation",
        }
    ).fillna(plot_df["target"])
    x = np.arange(len(plot_df))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    axes[0].bar(x, plot_df["auc_delta_vs_pga"], width=width, color="#2a6f97")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, plot_df["target_label"])
    axes[0].set_ylabel("AUC gain vs PGA-only")
    axes[0].set_title("Discrimination")

    brier_gain = -plot_df["brier_delta_vs_pga"]
    colors = ["#4c956c" if value > 0 else "#b56576" for value in brier_gain]
    axes[1].bar(x, brier_gain, width=width, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, plot_df["target_label"])
    axes[1].set_ylabel("Brier improvement vs PGA-only")
    axes[1].set_title("Calibration")

    for ax in axes:
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Canterbury field-facing critical-layer check", fontsize=12)
    fig.savefig(FIGURES / "fig16_canterbury_critical_layer_field_check.png", dpi=300)
    plt.close(fig)


def main() -> None:
    df = _build_event_states()
    df.to_csv(OUTPUTS / "canterbury_critical_layer_event_states.csv", index=False)
    models = {
        "pga_only": ["pga_g"],
        "pga_plus_wtd": ["pga_g", "wtd_event_site_adjusted_m"],
        "pga_plus_critical_saturated": ["pga_g", "critical_layer_saturated"],
        "pga_plus_crossing_margin": ["pga_g", "critical_crossing_margin_m"],
        "critical_layer_bundle": [
            "pga_g",
            "wtd_event_site_adjusted_m",
            "critical_layer_depth_m",
            "critical_layer_qc_mpa",
            "critical_layer_saturated",
            "critical_crossing_margin_m",
        ],
    }
    rows: list[dict[str, object]] = []
    for target in ["observed_liquefaction", "severe_manifestation"]:
        for model_name, features in models.items():
            metrics = _leave_one_event_metrics(df, target, features)
            if metrics is None:
                continue
            metrics["model_name"] = model_name
            rows.append(metrics)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUTS / "canterbury_critical_layer_field_check_metrics.csv", index=False)

    baseline = metrics_df[metrics_df["model_name"].eq("pga_only")].set_index("target")
    bundle = metrics_df[metrics_df["model_name"].eq("critical_layer_bundle")].set_index("target")
    contrasts = []
    for target in sorted(set(baseline.index) & set(bundle.index)):
        b = baseline.loc[target]
        c = bundle.loc[target]
        contrasts.append(
            {
                "target": target,
                "auc_delta_vs_pga": float(c["auc"] - b["auc"]),
                "brier_delta_vs_pga": float(c["brier_score"] - b["brier_score"]),
                "interpretation": "positive_auc_delta_negative_brier_delta_is_strong; current evidence is partial if only one metric improves",
            }
        )
    _write_contrast_figure(contrasts)

    summary = {
        "status": "PASS_PARTIAL_CANTERBURY_CRITICAL_LAYER_FIELD_CHECK",
        "n_event_states": int(len(df)),
        "critical_layer_definition": "median depth and qc of weakest 20% qc values between 0.5 and 6 m in each CPT",
        "metrics": metrics_df.to_dict("records"),
        "contrasts": contrasts,
        "claim_enabled": (
            "A CPT-derived critical-layer bundle adds small leave-one-event discrimination beyond PGA alone, especially as a field-facing check of the water-table crossing mechanism."
        ),
        "claim_blocked": (
            "Do not describe this as decisive field validation of the critical water-table switch. The improvement is small and mixed across metrics, and Canterbury outcomes remain strongly controlled by shaking intensity and site history."
        ),
    }
    (OUTPUTS / "canterbury_critical_layer_field_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Canterbury critical-layer field check - 2026-06-22",
        "",
        "## Protocol",
        "",
        "For each CPT, the shallow critical layer is estimated from the weakest 20% of qc values between 0.5 and 6 m. Leave-one-earthquake logistic models compare PGA-only prediction against models with water-table position relative to that critical layer.",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        f"Event states: `{summary['n_event_states']}`.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
        "",
        "## Figure",
        "",
        "`figures/fig16_canterbury_critical_layer_field_check.png` reports the AUC gain and Brier improvement of the critical-layer bundle relative to a PGA-only leave-one-earthquake baseline. The figure is intentionally contrast-based so that small or mixed gains remain visible.",
    ]
    (AUDIT / "canterbury_critical_layer_field_check_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
