"""Reader-oriented figures for the Article 118 manuscript.

The figures are not decorative. They turn the densest validation results into
simple visual objects: what improved, what it cost, and where the claim stops.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 24, fill=(20, 20, 20), bold: bool = False) -> None:
    draw.text(xy, text, font=_font(size, bold=bold), fill=fill)


def _wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, size: int = 22, fill=(40, 40, 40), bold: bool = False, line_spacing: int = 8) -> int:
    y = xy[1]
    for line in textwrap.wrap(text, width=width):
        _label(draw, (xy[0], y), line, size=size, fill=fill, bold=bold)
        y += size + line_spacing
    return y


def temporal_holdout_metrics() -> None:
    metrics = pd.read_csv(OUTPUTS / "canterbury_leave_one_event_temporal_pooled_metrics.csv")
    metrics = metrics.sort_values("brier_score").reset_index(drop=True)
    img = Image.new("RGB", (1500, 920), "white")
    d = ImageDraw.Draw(img)
    _label(d, (55, 35), "Canterbury temporal holdout: simpler question, harder test", 32, bold=True)
    _label(d, (55, 78), "Each earthquake is removed, models learn from the other two, then predict the held-out event.", 21, fill=(70, 70, 70))

    colors = {
        "M0_static_stationary": (90, 90, 90),
        "M1_nonstationary_groundwater_only": (38, 118, 170),
        "M2_nonstationary_groundwater_gradation": (30, 145, 95),
        "M3_full_nonstationary_random_field": (145, 75, 155),
    }
    labels = {
        "M0_static_stationary": "M0 static",
        "M1_nonstationary_groundwater_only": "M1 groundwater",
        "M2_nonstationary_groundwater_gradation": "M2 groundwater + gradation",
        "M3_full_nonstationary_random_field": "M3 full random field",
    }
    left, top = 470, 170
    max_brier = float(metrics["brier_score"].max())
    min_brier = float(metrics["brier_score"].min())
    for i, row in metrics.iterrows():
        y = top + i * 125
        model = row["model_name"]
        _label(d, (70, y + 10), labels.get(model, model), 23, bold=(i == 0))
        width = int(float(row["brier_score"]) / max_brier * 660)
        d.rounded_rectangle((left, y, left + width, y + 34), radius=6, fill=colors.get(model, (80, 80, 80)))
        _label(d, (left + width + 18, y - 1), f"Brier {row['brier_score']:.3f}", 22)
        auc_width = int(float(row["auc"]) * 660)
        d.rounded_rectangle((left, y + 52, left + auc_width, y + 86), radius=6, fill=(210, 225, 235))
        d.rectangle((left, y + 52, left + auc_width, y + 86), outline=colors.get(model, (80, 80, 80)), width=2)
        _label(d, (left + auc_width + 18, y + 50), f"AUC {row['auc']:.3f}", 22)
    _label(d, (70, 705), f"Best pooled score: M2 has the lowest Brier ({min_brier:.3f}) and the highest AUC ({metrics['auc'].max():.3f}).", 24, bold=True)
    _label(d, (70, 745), "Read this as temporal transfer evidence, not as universal model dominance: event-wise results remain mixed.", 22, fill=(70, 70, 70))
    _label(d, (70, 815), "n = 15,890 held-out event states; 63,560 model predictions across three held-out earthquakes.", 22, fill=(70, 70, 70))
    img.save(FIGURES / "fig12_canterbury_leave_one_event_temporal_validation.png")


def locked_threshold_tradeoff() -> None:
    summary = pd.read_csv(OUTPUTS / "locked_threshold_external_case_weighted_summary.csv")
    row10 = summary[summary["false_negative_cost_ratio"].eq(10)].copy()
    cant = pd.read_csv(OUTPUTS / "locked_threshold_external_decision_validation.csv")
    cant = cant[(cant["false_negative_cost_ratio"].eq(10)) & cant["domain"].eq("Canterbury multi-site external transfer")]
    m0 = cant[cant["model_name"].eq("M0_static_stationary")].iloc[0]
    m1 = cant[cant["model_name"].eq("M1_nonstationary_groundwater_only")].iloc[0]

    img = Image.new("RGB", (1500, 850), "white")
    d = ImageDraw.Draw(img)
    _label(d, (55, 35), "Locked threshold: fewer missed manifestations, more false alarms", 32, bold=True)
    _label(d, (55, 78), "Thresholds are chosen before Canterbury is seen. The figure shows the price of reducing false negatives.", 21, fill=(70, 70, 70))

    left, top = 170, 185
    max_fn = max(int(m0["fn"]), int(m1["fn"]), 1)
    max_fp = max(int(m0["fp"]), int(m1["fp"]), 1)
    rows = [("M0 static", int(m0["fn"]), int(m0["fp"]), (90, 90, 90)), ("M1 groundwater", int(m1["fn"]), int(m1["fp"]), (38, 118, 170))]
    for i, (name, fn, fp, color) in enumerate(rows):
        y = top + i * 210
        _label(d, (left, y - 28), name, 25, bold=True)
        fn_w = int(fn / max_fn * 430)
        fp_w = int(fp / max_fp * 430)
        d.rounded_rectangle((left + 270, y, left + 270 + fn_w, y + 45), radius=8, fill=(185, 65, 65))
        d.rounded_rectangle((left + 270, y + 70, left + 270 + fp_w, y + 115), radius=8, fill=color)
        _label(d, (left + 720, y + 3), f"False negatives: {fn}", 24, fill=(120, 30, 30), bold=True)
        _label(d, (left + 720, y + 73), f"False positives: {fp}", 24, fill=(40, 40, 40))
    _label(d, (170, 620), "Interpretation", 26, bold=True)
    _wrapped(
        d,
        (170, 665),
        "M1 cuts Canterbury false negatives from 99 to 9 at FN:FP = 10, but it buys this by adding 770 false alarms.",
        width=92,
        size=24,
    )
    _wrapped(
        d,
        (170, 735),
        "That is a risk-management trade-off, not a proof that one probability model is always better.",
        width=92,
        size=23,
        fill=(70, 70, 70),
    )
    img.save(FIGURES / "fig13_locked_threshold_fn_fp_tradeoff.png")


def evidence_boundary() -> None:
    img = Image.new("RGB", (1500, 880), "white")
    d = ImageDraw.Draw(img)
    _label(d, (55, 35), "What the evidence proves, and where it stops", 32, bold=True)
    _label(d, (55, 78), "The manuscript is strongest when every claim is kept inside its evidence box.", 21, fill=(70, 70, 70))

    boxes = [
        ("Verified", "Equations, code, and reproducible outputs", (60, 130, 190)),
        ("Validated", "Hu event holdout and Canterbury event holdout", (35, 145, 95)),
        ("Decision-ready", "Locked thresholds and false-negative costs", (190, 120, 40)),
        ("Blocked", "Universal superiority or design-level site prediction", (170, 55, 55)),
    ]
    x0, y0 = 95, 190
    for i, (title, body, color) in enumerate(boxes):
        x = x0 + i * 345
        d.rounded_rectangle((x, y0, x + 290, y0 + 220), radius=14, outline=color, width=4, fill=(248, 250, 250))
        _label(d, (x + 24, y0 + 26), title, 28, fill=color, bold=True)
        _wrapped(d, (x + 24, y0 + 88), body, width=22, size=22, fill=(40, 40, 40))
        if i < len(boxes) - 1:
            d.line((x + 300, y0 + 110, x + 335, y0 + 110), fill=(100, 100, 100), width=4)
            d.polygon([(x + 335, y0 + 110), (x + 318, y0 + 100), (x + 318, y0 + 120)], fill=(100, 100, 100))

    _label(d, (95, 500), "Plain-language claim:", 28, bold=True)
    _label(d, (95, 545), "Changing groundwater and gradation can change the screening decision.", 28, fill=(20, 20, 20))
    _label(d, (95, 590), "The paper shows when that matters, what it costs, and when the safer answer is still M0.", 26, fill=(70, 70, 70))
    _label(d, (95, 690), "Plain-language limit:", 28, bold=True)
    _label(d, (95, 735), "It is not a magic predictor for every site. A design decision still needs site calibration.", 26, fill=(70, 70, 70))
    img.save(FIGURES / "fig14_evidence_claim_boundary.png")


def main() -> None:
    temporal_holdout_metrics()
    locked_threshold_tradeoff()
    evidence_boundary()
    print("Wrote reader-oriented figures fig12, fig13 and fig14.")


if __name__ == "__main__":
    main()
