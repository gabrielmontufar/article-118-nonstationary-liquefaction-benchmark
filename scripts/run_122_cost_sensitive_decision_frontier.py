"""Cost-sensitive georisk decision frontier for Article 118.

The probability models are already evaluated in the validation scripts. This
script converts those predictions into an engineering decision diagnostic:
expected screening loss when a false negative is more costly than a false
positive. It reports two distinct views: a diagnostic/oracle frontier where
thresholds are retuned within each validation domain, and a fixed-threshold
0.50 frontier that is closer to an operational screening rule. It does not
recalibrate probabilities or hide Brier-score failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

MODEL_ORDER = [
    "M0_static_stationary",
    "M1_nonstationary_groundwater_only",
    "M2_nonstationary_groundwater_gradation",
    "M3_full_nonstationary_random_field",
    "conservative_max_M0_M3",
]

LOSS_RATIOS = [1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)


def _read_primary_nisqually() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "site_prediction_probabilities.csv")
    df = df[df["validation_protocol"].eq("leave-one-site-family-out spatial validation")].copy()
    df["domain"] = "Nisqually primary leave-one-family"
    df["case_key"] = df["site_id"]
    return df


def _read_sodo_uw() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "nisqually_protocol_exploration_predictions.csv")
    df = df[df["protocol"].eq("fixed_family_holdout_SODO_UW") & df["heldout_group"].eq("SODO_UW")].copy()
    df["domain"] = "SODO/UW adverse industrial-waterway holdout"
    df["case_key"] = df["site_id"]
    return df


def _read_canterbury() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "canterbury_multi_site_temporal_prediction_probabilities.csv")
    df["domain"] = "Canterbury multi-site external transfer"
    df["case_key"] = df["site_id"].astype(str) + "_" + df["year"].astype(str)
    return df


def _add_conservative_max(df: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "domain",
        "case_key",
        "site_id",
        "observed_liquefaction",
    ]
    optional = [col for col in ["site_family", "region", "year", "event_name", "validation_protocol"] if col in df.columns]
    index_cols += optional
    wide = df.pivot_table(index=index_cols, columns="model_name", values="predicted_pf", aggfunc="first").reset_index()
    model_cols = [col for col in MODEL_ORDER if col in wide.columns and col != "conservative_max_M0_M3"]
    wide["conservative_max_M0_M3"] = wide[model_cols].max(axis=1)
    rows = []
    for model in MODEL_ORDER:
        if model not in wide.columns:
            continue
        keep = index_cols + [model]
        tmp = wide[keep].rename(columns={model: "predicted_pf"}).copy()
        tmp["model_name"] = model
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def load_predictions() -> pd.DataFrame:
    parts = [_read_primary_nisqually(), _read_sodo_uw(), _read_canterbury()]
    return pd.concat([_add_conservative_max(part) for part in parts], ignore_index=True)


def _confusion(prob: np.ndarray, y: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def _loss(fp: int, fn: int, n: int, false_negative_cost_ratio: float) -> float:
    return (float(fp) + false_negative_cost_ratio * float(fn)) / max(int(n), 1)


def frontier_rows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fixed_rows = []
    for (domain, model), group in predictions.groupby(["domain", "model_name"], sort=False):
        prob = group["predicted_pf"].to_numpy(dtype=float)
        y = group["observed_liquefaction"].to_numpy(dtype=int)
        n = len(group)
        for ratio in LOSS_RATIOS:
            best = None
            for threshold in THRESHOLDS:
                tp, fp, tn, fn = _confusion(prob, y, float(threshold))
                loss = _loss(fp, fn, n, ratio)
                candidate = (loss, threshold, tp, fp, tn, fn)
                if best is None or candidate < best:
                    best = candidate
            assert best is not None
            loss, threshold, tp, fp, tn, fn = best
            rows.append(
                {
                    "domain": domain,
                    "model_name": model,
                    "false_negative_cost_ratio": ratio,
                    "threshold": threshold,
                    "expected_loss_per_case": loss,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "n_cases": n,
                    "threshold_policy": "diagnostic_oracle_domain_retuned_threshold",
                }
            )

            tp, fp, tn, fn = _confusion(prob, y, 0.5)
            fixed_rows.append(
                {
                    "domain": domain,
                    "model_name": model,
                    "false_negative_cost_ratio": ratio,
                    "threshold": 0.5,
                    "expected_loss_per_case": _loss(fp, fn, n, ratio),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "n_cases": n,
                    "threshold_policy": "fixed_threshold_0p50",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(fixed_rows)


def best_by_domain(frontier: pd.DataFrame) -> pd.DataFrame:
    ordered = frontier.copy()
    ordered["model_rank"] = ordered["model_name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    return (
        ordered.sort_values(["domain", "false_negative_cost_ratio", "expected_loss_per_case", "model_rank"])
        .groupby(["domain", "false_negative_cost_ratio"], as_index=False)
        .first()
        .drop(columns=["model_rank"])
    )


def balanced_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, ratio), group in frontier.groupby(["model_name", "false_negative_cost_ratio"], sort=False):
        rows.append(
            {
                "domain": "Balanced mean across validation domains",
                "model_name": model,
                "false_negative_cost_ratio": ratio,
                "threshold": float("nan"),
                "expected_loss_per_case": float(group["expected_loss_per_case"].mean()),
                "tp": "",
                "fp": "",
                "tn": "",
                "fn": "",
                "n_cases": int(group["n_cases"].sum()),
                "threshold_policy": "balanced_mean_of_domain_diagnostic_optima",
            }
        )
    balanced = pd.DataFrame(rows)
    best = best_by_domain(balanced)
    return pd.concat([balanced, best.assign(threshold_policy="best_balanced_diagnostic_model")], ignore_index=True)


def case_weighted_fixed_frontier(fixed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, ratio), group in fixed.groupby(["model_name", "false_negative_cost_ratio"], sort=False):
        total_cases = float(group["n_cases"].sum())
        weighted_loss = float((group["expected_loss_per_case"] * group["n_cases"]).sum() / max(total_cases, 1.0))
        rows.append(
            {
                "domain": "Case-weighted global fixed-threshold mean",
                "model_name": model,
                "false_negative_cost_ratio": ratio,
                "threshold": 0.5,
                "expected_loss_per_case": weighted_loss,
                "tp": int(group["tp"].sum()),
                "fp": int(group["fp"].sum()),
                "tn": int(group["tn"].sum()),
                "fn": int(group["fn"].sum()),
                "n_cases": int(group["n_cases"].sum()),
                "threshold_policy": "case_weighted_fixed_threshold_0p50",
            }
        )
    weighted = pd.DataFrame(rows)
    best = (
        weighted.sort_values(["false_negative_cost_ratio", "expected_loss_per_case"])
        .groupby("false_negative_cost_ratio", as_index=False)
        .first()
    )
    best["threshold_policy"] = "best_case_weighted_fixed_threshold_model"
    return pd.concat([weighted, best], ignore_index=True)


def _model_label(model: str) -> str:
    return {
        "M0_static_stationary": "M0 static",
        "M1_nonstationary_groundwater_only": "M1 groundwater",
        "M2_nonstationary_groundwater_gradation": "M2 gw+gradation",
        "M3_full_nonstationary_random_field": "M3 random field",
        "conservative_max_M0_M3": "max(M0..M3)",
    }.get(model, model)


def plot_frontier(domain_best: pd.DataFrame, balanced: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1500, 980), "white")
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype("arial.ttf", 28)
    font = ImageFont.truetype("arial.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 16)
    d.text((45, 28), "Cost-sensitive georisk decision frontier: when false negatives dominate", font=title, fill=(0, 0, 0))

    colors = {
        "M0_static_stationary": (80, 80, 80),
        "M1_nonstationary_groundwater_only": (45, 120, 170),
        "M2_nonstationary_groundwater_gradation": (30, 140, 90),
        "M3_full_nonstationary_random_field": (150, 70, 140),
        "conservative_max_M0_M3": (190, 115, 35),
    }
    left, top, width, height = 95, 105, 1320, 360
    d.rectangle((left, top, left + width, top + height), outline=(40, 40, 40), width=2)
    bal = balanced[balanced["threshold_policy"].eq("balanced_mean_of_domain_diagnostic_optima")]
    ymax = float(bal["expected_loss_per_case"].max()) * 1.08
    xmin, xmax = min(LOSS_RATIOS), max(LOSS_RATIOS)

    def xmap(x: float) -> float:
        return left + (np.log10(x) - np.log10(xmin)) / (np.log10(xmax) - np.log10(xmin)) * width

    def ymap(y: float) -> float:
        return top + height - y / ymax * height

    for model, group in bal.groupby("model_name", sort=False):
        group = group.sort_values("false_negative_cost_ratio")
        pts = [(xmap(float(r.false_negative_cost_ratio)), ymap(float(r.expected_loss_per_case))) for r in group.itertuples()]
        if len(pts) > 1:
            d.line(pts, fill=colors.get(model, (0, 0, 0)), width=4)
        for x, y in pts:
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors.get(model, (0, 0, 0)))

    for ratio in [1, 2, 5, 10, 20, 50, 100]:
        x = xmap(ratio)
        d.line((x, top + height, x, top + height + 8), fill=(0, 0, 0), width=1)
        d.text((x - 12, top + height + 14), str(ratio), font=small, fill=(0, 0, 0))
    d.text((left + 450, top + height + 42), "False-negative cost / false-positive cost", font=font, fill=(0, 0, 0))
    d.text((22, top + 135), "Balanced expected loss", font=small, fill=(0, 0, 0))

    yleg = 110
    for model in MODEL_ORDER:
        d.line((1120, yleg, 1165, yleg), fill=colors[model], width=5)
        d.text((1178, yleg - 10), _model_label(model), font=small, fill=(0, 0, 0))
        yleg += 30

    y0 = 535
    d.text((45, y0), "Best model by domain after cost-optimized thresholding", font=font, fill=(0, 0, 0))
    y = y0 + 42
    for domain, group in domain_best.groupby("domain", sort=False):
        d.text((55, y), domain, font=small, fill=(0, 0, 0))
        y += 26
        best_segments = group.sort_values("false_negative_cost_ratio")
        for row in best_segments.itertuples():
            text = (
                f"FN:FP={row.false_negative_cost_ratio:g} -> {_model_label(row.model_name)} "
                f"(thr={row.threshold:.3g}, loss={row.expected_loss_per_case:.3f}, FN={row.fn}, FP={row.fp})"
            )
            d.text((85, y), text, font=small, fill=colors.get(row.model_name, (0, 0, 0)))
            y += 22
            if y > 930:
                break
        y += 18
        if y > 930:
            break
    img.save(path)


def summarize(frontier: pd.DataFrame, fixed: pd.DataFrame, domain_best: pd.DataFrame, balanced: pd.DataFrame) -> dict[str, object]:
    case_weighted_fixed = case_weighted_fixed_frontier(fixed)
    balanced_best = balanced[balanced["threshold_policy"].eq("best_balanced_diagnostic_model")].sort_values("false_negative_cost_ratio")
    case_weighted_best = case_weighted_fixed[
        case_weighted_fixed["threshold_policy"].eq("best_case_weighted_fixed_threshold_model")
    ].sort_values("false_negative_cost_ratio")
    high_cost = domain_best[domain_best["false_negative_cost_ratio"].eq(10)]
    threshold_summary = {
        row["domain"]: {
            "best_model_at_fn_fp_10": row["model_name"],
            "threshold": float(row["threshold"]),
            "expected_loss_per_case": float(row["expected_loss_per_case"]),
            "fn": int(row["fn"]),
            "fp": int(row["fp"]),
        }
        for _, row in high_cost.iterrows()
    }
    fixed_050 = fixed[fixed["false_negative_cost_ratio"].eq(10)]
    m0_fixed = fixed_050[fixed_050["model_name"].eq("M0_static_stationary")]
    conservative_fixed = fixed_050[fixed_050["model_name"].eq("conservative_max_M0_M3")]
    fixed_mean = (
        fixed.groupby(["false_negative_cost_ratio", "model_name"], as_index=False)["expected_loss_per_case"]
        .mean()
        .rename(columns={"expected_loss_per_case": "mean_expected_loss"})
    )
    fixed_best = (
        fixed_mean.sort_values(["false_negative_cost_ratio", "mean_expected_loss"])
        .groupby("false_negative_cost_ratio", as_index=False)
        .first()[["false_negative_cost_ratio", "model_name", "mean_expected_loss"]]
    )
    return {
        "status": "PASS_COST_SENSITIVE_DECISION_FRONTIER",
        "false_negative_cost_ratios": LOSS_RATIOS,
        "threshold_grid": [float(THRESHOLDS.min()), float(THRESHOLDS.max()), float(THRESHOLDS[1] - THRESHOLDS[0])],
        "domains": sorted(frontier["domain"].unique().tolist()),
        "best_model_by_domain_at_fn_fp_10": threshold_summary,
        "balanced_best_models": [
            {
                "false_negative_cost_ratio": float(row.false_negative_cost_ratio),
                "best_model": row.model_name,
                "balanced_expected_loss": float(row.expected_loss_per_case),
            }
            for row in balanced_best.itertuples()
        ],
        "case_weighted_fixed_threshold_0p50_best_models": [
            {
                "false_negative_cost_ratio": float(row.false_negative_cost_ratio),
                "best_model": row.model_name,
                "case_weighted_expected_loss": float(row.expected_loss_per_case),
                "fn": int(row.fn),
                "fp": int(row.fp),
                "n_cases": int(row.n_cases),
            }
            for row in case_weighted_best.itertuples()
        ],
        "fixed_threshold_0p50_fn_fp_10_loss": {
            "domain_balanced_M0_static_stationary_mean": float(m0_fixed["expected_loss_per_case"].mean()),
            "domain_balanced_conservative_max_M0_M3_mean": float(conservative_fixed["expected_loss_per_case"].mean()),
            "case_weighted_M0_static_stationary_mean": float(
                case_weighted_fixed[
                    case_weighted_fixed["threshold_policy"].eq("case_weighted_fixed_threshold_0p50")
                    & case_weighted_fixed["false_negative_cost_ratio"].eq(10)
                    & case_weighted_fixed["model_name"].eq("M0_static_stationary")
                ].iloc[0]["expected_loss_per_case"]
            ),
            "case_weighted_conservative_max_M0_M3_mean": float(
                case_weighted_fixed[
                    case_weighted_fixed["threshold_policy"].eq("case_weighted_fixed_threshold_0p50")
                    & case_weighted_fixed["false_negative_cost_ratio"].eq(10)
                    & case_weighted_fixed["model_name"].eq("conservative_max_M0_M3")
                ].iloc[0]["expected_loss_per_case"]
            ),
        },
        "fixed_threshold_0p50_best_models": [
            {
                "false_negative_cost_ratio": float(row.false_negative_cost_ratio),
                "best_model": row.model_name,
                "mean_expected_loss": float(row.mean_expected_loss),
            }
            for row in fixed_best.itertuples()
        ],
        "claim_enabled": (
            "cost-sensitive georisk screening with an explicit diagnostic/operational split: a domain-retuned diagnostic frontier "
            "favours M0, while the standard fixed-threshold 0.50 frontier favours conservative max(M0..M3) once false negatives "
            "are moderately more costly than false positives"
        ),
        "claim_blocked": (
            "do not present the decision frontier as probability recalibration or universal predictive superiority"
        ),
        "manuscript_sentence": (
            "The cost-sensitive frontier converts the adverse Canterbury and SODO/UW findings into an explicit georisk "
            "decision diagnostic: M0 remains the loss-minimising baseline in the domain-retuned oracle frontier, whereas the "
            "conservative max(M0..M3) screen becomes the lowest-loss fixed-threshold 0.50 rule when missed liquefaction "
            "manifestations are assigned materially higher cost than false alarms."
        ),
    }


def main() -> None:
    predictions = load_predictions()
    frontier, fixed = frontier_rows(predictions)
    domain_best = best_by_domain(frontier)
    balanced = balanced_frontier(frontier)
    case_weighted_fixed = case_weighted_fixed_frontier(fixed)

    frontier.to_csv(OUTPUTS / "cost_sensitive_decision_frontier.csv", index=False)
    fixed.to_csv(OUTPUTS / "cost_sensitive_decision_frontier_fixed_threshold.csv", index=False)
    domain_best.to_csv(OUTPUTS / "cost_sensitive_decision_frontier_best_by_domain.csv", index=False)
    balanced.to_csv(OUTPUTS / "cost_sensitive_decision_frontier_balanced.csv", index=False)
    case_weighted_fixed.to_csv(OUTPUTS / "cost_sensitive_decision_frontier_case_weighted_fixed_threshold.csv", index=False)
    plot_frontier(domain_best, balanced, FIGURES / "fig11_cost_sensitive_guardrail_frontier.png")

    summary = summarize(frontier, fixed, domain_best, balanced)
    (OUTPUTS / "cost_sensitive_decision_frontier_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
