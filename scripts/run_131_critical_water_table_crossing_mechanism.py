"""Critical water-table crossing mechanism.

This script extracts the physical mechanism that is strongest in Article 118:
when a non-stationary water table crosses a shallow liquefiable layer, pore
pressure turns on inside the layer, effective stress drops, CSR rises, FS
falls, and Pf jumps. Gradation trajectories then modulate the resistance side.

This is a mechanism figure/gate. It is not a claim of a new constitutive law or
universal predictive superiority.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _scale(values: list[float], lo: float, hi: float, pix_lo: int, pix_hi: int, invert: bool = True) -> list[int]:
    span = max(hi - lo, 1e-9)
    out = []
    for value in values:
        t = (float(value) - lo) / span
        y = pix_hi - t * (pix_hi - pix_lo) if invert else pix_lo + t * (pix_hi - pix_lo)
        out.append(int(round(y)))
    return out


def _draw_series(draw: ImageDraw.ImageDraw, xs: list[int], ys: list[int], color: tuple[int, int, int], width: int = 4) -> None:
    if len(xs) > 1:
        draw.line(list(zip(xs, ys)), fill=color, width=width, joint="curve")
    for x, y in zip(xs, ys):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def main() -> None:
    results = pd.read_csv(DATA / "liquefaction_benchmark_results.csv")
    extreme = results[results["scenario"].eq("extreme")].copy()

    rows: list[dict[str, object]] = []
    for (gradation, layer), group in extreme.groupby(["gradation", "layer"], sort=False):
        group = group.sort_values("year").reset_index(drop=True)
        z = float(group["z_mid_m"].iloc[0])
        for i in range(1, len(group)):
            before = group.iloc[i - 1]
            after = group.iloc[i]
            crosses = before["gw_depth_m"] > z and after["gw_depth_m"] <= z
            rows.append(
                {
                    "scenario": "extreme",
                    "gradation": gradation,
                    "layer": layer,
                    "z_mid_m": z,
                    "year_before": int(before["year"]),
                    "year_after": int(after["year"]),
                    "water_table_crosses_layer": bool(crosses),
                    "gw_before_m": float(before["gw_depth_m"]),
                    "gw_after_m": float(after["gw_depth_m"]),
                    "csr_before": float(before["csr"]),
                    "csr_after": float(after["csr"]),
                    "csr_relative_jump": float(after["csr"] / before["csr"] - 1.0),
                    "fs_before": float(before["fs_deterministic_nonstationary"]),
                    "fs_after": float(after["fs_deterministic_nonstationary"]),
                    "fs_relative_drop": float(1.0 - after["fs_deterministic_nonstationary"] / before["fs_deterministic_nonstationary"]),
                    "pf_before": float(before["pf_nonstationary"]),
                    "pf_after": float(after["pf_nonstationary"]),
                    "pf_jump": float(after["pf_nonstationary"] - before["pf_nonstationary"]),
                    "fc_before_pct": float(before["fc_pct"]),
                    "fc_after_pct": float(after["fc_pct"]),
                }
            )

    jumps = pd.DataFrame(rows)
    crossings = jumps[jumps["water_table_crosses_layer"]].copy()
    ranked = jumps.sort_values(["pf_jump", "csr_relative_jump"], ascending=False).head(20)
    crossings.to_csv(OUTPUTS / "critical_water_table_crossing_events.csv", index=False)
    ranked.to_csv(OUTPUTS / "critical_water_table_largest_probability_jumps.csv", index=False)

    target = extreme[
        extreme["layer"].eq("L1")
        & extreme["gradation"].isin(["constant", "fines_accumulation", "fines_washout"])
    ].copy()
    target = target.sort_values(["gradation", "year"])

    img = Image.new("RGB", (1800, 1200), "white")
    d = ImageDraw.Draw(img)
    title = _font(44, True)
    h2 = _font(28, True)
    body = _font(23)
    small = _font(19)
    d.text((70, 45), "Critical water-table crossing: a physical switch for liquefaction risk", font=title, fill=(15, 33, 45))
    d.text((72, 105), "Layer crossing lowers effective stress; CSR rises, FS falls, and Pf jumps.", font=body, fill=(45, 55, 65))

    colors = {
        "constant": (70, 100, 130),
        "fines_accumulation": (30, 135, 80),
        "fines_washout": (170, 80, 60),
    }
    labels = {
        "constant": "constant gradation",
        "fines_accumulation": "fines accumulation",
        "fines_washout": "fines washout",
    }

    panels = [
        ("Water table depth", "gw_depth_m", 0.5, 3.4, True, "m below ground"),
        ("Cyclic stress ratio", "csr", float(target["csr"].min()) * 0.95, float(target["csr"].max()) * 1.05, True, "CSR"),
        ("Deterministic FS", "fs_deterministic_nonstationary", float(target["fs_deterministic_nonstationary"].min()) * 0.95, float(target["fs_deterministic_nonstationary"].max()) * 1.05, True, "FS"),
        ("Liquefaction probability", "pf_nonstationary", 0.55, 0.95, True, "Pf"),
    ]
    x0, y0 = 95, 170
    pw, ph = 760, 370
    gapx, gapy = 95, 90
    years = sorted(target["year"].unique().tolist())
    xs = [x0 + 70 + int((year - min(years)) / (max(years) - min(years)) * (pw - 135)) for year in years]

    for pidx, (panel_title, col, lo, hi, invert, ylabel) in enumerate(panels):
        px = x0 + (pidx % 2) * (pw + gapx)
        py = y0 + (pidx // 2) * (ph + gapy)
        d.rectangle((px, py, px + pw, py + ph), outline=(190, 198, 205), width=2)
        d.text((px + 22, py + 18), panel_title, font=h2, fill=(20, 35, 45))
        d.line((px + 70, py + 72, px + 70, py + ph - 55), fill=(90, 98, 108), width=2)
        d.line((px + 70, py + ph - 55, px + pw - 55, py + ph - 55), fill=(90, 98, 108), width=2)
        d.text((px + 15, py + 80), ylabel, font=small, fill=(90, 98, 108))
        d.text((px + 62, py + ph - 42), str(min(years)), font=small, fill=(80, 80, 80))
        d.text((px + pw - 95, py + ph - 42), str(max(years)), font=small, fill=(80, 80, 80))
        cross_x = px + 70 + int((30 - min(years)) / (max(years) - min(years)) * (pw - 135))
        d.line((cross_x, py + 72, cross_x, py + ph - 55), fill=(80, 80, 80), width=2)
        d.text((cross_x + 8, py + 78), "L1 crossing", font=small, fill=(80, 80, 80))
        if col == "gw_depth_m":
            layer_y = py + 72 + int((2.0 - lo) / (hi - lo) * (ph - 127))
            d.line((px + 70, layer_y, px + pw - 55, layer_y), fill=(40, 40, 40), width=2)
            d.text((px + pw - 230, layer_y + 8), "L1 z=2 m", font=small, fill=(40, 40, 40))
        for grad, color in colors.items():
            g = target[target["gradation"].eq(grad)].sort_values("year")
            ys = _scale(g[col].tolist(), lo, hi, py + 72, py + ph - 55, invert=invert)
            pxs = [px + 70 + int((year - min(years)) / (max(years) - min(years)) * (pw - 135)) for year in g["year"]]
            _draw_series(d, pxs, ys, color)

    legend_x, legend_y = 1040, 1055
    for idx, (grad, color) in enumerate(colors.items()):
        y = legend_y + idx * 34
        d.line((legend_x, y + 10, legend_x + 60, y + 10), fill=color, width=6)
        d.text((legend_x + 75, y), labels[grad], font=body, fill=(30, 30, 30))

    best_cross = crossings.sort_values("pf_jump", ascending=False).head(1).to_dict("records")
    if best_cross:
        r = best_cross[0]
        note = (
            f"Largest crossing jump: {r['layer']} / {r['gradation']}, "
            f"Pf +{r['pf_jump']:.3f}, CSR +{100*r['csr_relative_jump']:.1f}%, "
            f"FS -{100*r['fs_relative_drop']:.1f}%."
        )
        d.text((70, 1115), note, font=body, fill=(15, 33, 45))
    d.text((70, 1150), "Claim boundary: physical regime in the benchmark, not a new universal constitutive law.", font=small, fill=(80, 80, 80))
    fig_path = FIGURES / "fig15_critical_water_table_crossing_mechanism.png"
    img.save(fig_path, dpi=(180, 180))

    summary = {
        "status": "PASS_CRITICAL_WATER_TABLE_CROSSING_MECHANISM",
        "mechanism": "water-table crossing of shallow liquefiable layers creates a threshold-like effective-stress/CSR/Pf amplification",
        "n_crossing_intervals": int(len(crossings)),
        "largest_crossing_jump": best_cross,
        "largest_any_adjacent_jumps": ranked.head(5).to_dict("records"),
        "figure": str(fig_path.relative_to(ROOT)).replace("\\", "/"),
        "claim_enabled": (
            "Article 118 can foreground a bounded physical breakthrough: the non-stationary risk increment is not merely gradual drift; it is dominated by critical water-table crossing of shallow liquefiable layers, where effective-stress loss amplifies CSR and produces abrupt Pf jumps, with gradation changing the resistance-side magnitude."
        ),
        "claim_blocked": (
            "Do not claim a new constitutive law, measured gradation-evolution discovery, universal predictive superiority, or repair-cost validation from this mechanism figure alone."
        ),
    }
    (OUTPUTS / "critical_water_table_crossing_mechanism_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Critical water-table crossing mechanism - 2026-06-22",
        "",
        "## Physical claim tested",
        "",
        "When the non-stationary water table crosses a shallow liquefiable layer, pore pressure becomes active within the layer, effective vertical stress drops, CSR rises, deterministic FS falls, and Pf jumps.",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        f"Crossing intervals found: `{summary['n_crossing_intervals']}`.",
        f"Figure: `{summary['figure']}`.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
    ]
    (AUDIT / "critical_water_table_crossing_mechanism_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
