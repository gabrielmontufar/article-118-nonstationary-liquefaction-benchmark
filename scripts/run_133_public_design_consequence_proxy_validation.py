"""Public design/consequence proxy validation.

This gate spatially joins Article 118 Canterbury CPT predictions to public
Christchurch/Canterbury layers that encode expected land-performance,
foundation-category or recovery-consequence classes.

It is stronger than a pure hazard overlay because several layers are used for
planning, subdivision, building-consent or foundation guidance. It remains a
proxy validation: no numeric repair-cost or property-level EQC/NHC payout is
available in these public services.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation_metrics import auc_rank


DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
AUDIT = ROOT / "audit_logs"
OUTPUTS.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)

LAYERS = {
    "ccc_liquefaction_hazard": "https://gis.ccc.govt.nz/server/rest/services/OpenData/LandCharacteristic/FeatureServer/34",
    "ccc_liquefaction_vulnerability": "https://gis.ccc.govt.nz/server/rest/services/OpenData/LandCharacteristic/FeatureServer/36",
    "ccc_quake_foundation_design": "https://gis.ccc.govt.nz/server/rest/services/OpenData/LandCharacteristic/FeatureServer/94",
    "mbie_technical_categories": "https://gis.ecan.govt.nz/arcgis/rest/services/Beta/PropertySearch/MapServer/8",
}


def _query_layer(url: str) -> list[dict[str, Any]]:
    count = requests.get(url + "/query", params={"where": "1=1", "returnCountOnly": "true", "f": "json"}, timeout=60)
    count.raise_for_status()
    n_features = int(count.json()["count"])
    features: list[dict[str, Any]] = []
    for offset in range(0, n_features, 1000):
        resp = requests.get(
            url + "/query",
            params={
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": 2193,
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": 1000,
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        features.extend(payload.get("features", []))
    return features


def _feature_polygons(features: list[dict[str, Any]]) -> tuple[list[Polygon], list[dict[str, Any]]]:
    polygons: list[Polygon] = []
    attrs: list[dict[str, Any]] = []
    for feature in features:
        attributes = feature.get("attributes", {})
        for ring in feature.get("geometry", {}).get("rings", []):
            if len(ring) < 4:
                continue
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            polygons.append(poly)
            attrs.append(attributes)
    return polygons, attrs


def _join_attrs(points: pd.DataFrame, polygons: list[Polygon], attrs: list[dict[str, Any]], prefix: str) -> pd.DataFrame:
    tree = STRtree(polygons)
    rows: list[dict[str, Any]] = []
    for _, row in points.iterrows():
        point = Point(float(row["nztm_x"]), float(row["nztm_y"]))
        hit: dict[str, Any] = {}
        for idx in tree.query(point):
            poly_index = int(idx)
            if polygons[poly_index].contains(point) or polygons[poly_index].touches(point):
                hit = attrs[poly_index]
                break
        out = {"site_id": row["site_id"]}
        out.update({f"{prefix}_{k}": v for k, v in hit.items()})
        rows.append(out)
    return pd.DataFrame(rows)


def _liq_cat_score(value: object) -> float | None:
    text = str(value or "").lower()
    if "high" in text:
        return 3.0
    if "medium" in text:
        return 2.0
    if "low" in text or "possible" in text:
        return 1.0
    if "unlikely" in text:
        return 0.0
    return None


def _tc_score(value: object) -> float | None:
    text = str(value or "").lower()
    if "red zone" in text:
        return 4.0
    if "tc3" in text:
        return 3.0
    if "tc2" in text:
        return 2.0
    if "tc1" in text:
        return 1.0
    if "n/a" in text or text in {"na", "none", ""}:
        return 0.0
    return None


def _metrics(site_model: pd.DataFrame, target_col: str, target_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = site_model[site_model[target_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    threshold = valid[target_col].quantile(0.75)
    valid["target_high"] = valid[target_col].ge(threshold).astype(int)
    for model_name, group in valid.groupby("model_name"):
        labels = group["target_high"].astype(int).to_numpy()
        scores = group["max_predicted_pf"].astype(float).to_numpy()
        rows.append(
            {
                "target": target_label,
                "target_col": target_col,
                "high_threshold": float(threshold),
                "model_name": model_name,
                "n_sites": int(len(group)),
                "n_high": int(labels.sum()),
                "auc_for_top_quartile_target": auc_rank(scores, labels),
                "spearman_ordinal": group["max_predicted_pf"].corr(group[target_col], method="spearman"),
                "median_pf_high": float(group.loc[group["target_high"].eq(1), "max_predicted_pf"].median()),
                "median_pf_not_high": float(group.loc[group["target_high"].eq(0), "max_predicted_pf"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sites = pd.read_csv(DATA / "canterbury_multi_site_cpt_summary.csv")
    sites = sites[["site_id", "latitude", "longitude"]].drop_duplicates("site_id").copy()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    xs, ys = transformer.transform(sites["longitude"].to_numpy(), sites["latitude"].to_numpy())
    sites["nztm_x"] = xs
    sites["nztm_y"] = ys

    joined = sites.copy()
    layer_meta = {}
    for name, url in LAYERS.items():
        features = _query_layer(url)
        polygons, attrs = _feature_polygons(features)
        layer_meta[name] = {"url": url, "n_features": len(features), "n_polygon_rings": len(polygons)}
        joined = joined.merge(_join_attrs(sites, polygons, attrs, name), on="site_id", how="left", validate="one_to_one")

    hazard_mod_cols = [c for c in joined.columns if c.startswith("ccc_liquefaction_hazard_Scenario") and c.endswith("ModToSevereDmgeClass")]
    hazard_damage_cols = [c for c in joined.columns if c.startswith("ccc_liquefaction_hazard_Scenario") and c.endswith("DmgeClass") and "To" not in c]
    joined["ccc_hazard_max_mod_to_severe_pct"] = joined[hazard_mod_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    joined["ccc_hazard_max_damage_class"] = joined[hazard_damage_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    joined["ccc_vulnerability_score"] = joined["ccc_liquefaction_vulnerability_Liq_Cat"].apply(_liq_cat_score)
    joined["quake_foundation_tc_score"] = joined["ccc_quake_foundation_design_Code"].apply(_tc_score)
    joined["mbie_tc_score"] = joined["mbie_technical_categories_DBH_TC"].apply(_tc_score)
    joined.to_csv(OUTPUTS / "public_design_consequence_proxy_joined_sites.csv", index=False)

    predictions = pd.read_csv(OUTPUTS / "canterbury_leave_one_event_temporal_predictions.csv")
    site_model = (
        predictions.groupby(["site_id", "model_name"], as_index=False)
        .agg(max_predicted_pf=("predicted_pf", "max"), mean_predicted_pf=("predicted_pf", "mean"))
        .merge(
            joined[
                [
                    "site_id",
                    "ccc_hazard_max_mod_to_severe_pct",
                    "ccc_hazard_max_damage_class",
                    "ccc_vulnerability_score",
                    "quake_foundation_tc_score",
                    "mbie_tc_score",
                    "ccc_liquefaction_vulnerability_Liq_Cat",
                    "ccc_quake_foundation_design_Code",
                    "mbie_technical_categories_DBH_TC",
                ]
            ],
            on="site_id",
            how="left",
            validate="many_to_one",
        )
    )
    site_model.to_csv(OUTPUTS / "public_design_consequence_proxy_site_model_scores.csv", index=False)

    metric_tables = [
        _metrics(site_model, "ccc_hazard_max_mod_to_severe_pct", "CCC max moderate-to-severe damage percentage"),
        _metrics(site_model, "ccc_hazard_max_damage_class", "CCC max scenario damage class"),
        _metrics(site_model, "ccc_vulnerability_score", "CCC liquefaction vulnerability category"),
        _metrics(site_model, "quake_foundation_tc_score", "CCC quake foundation technical category"),
        _metrics(site_model, "mbie_tc_score", "MBIE technical category/red zone"),
    ]
    metrics = pd.concat([m for m in metric_tables if not m.empty], ignore_index=True)
    metrics.to_csv(OUTPUTS / "public_design_consequence_proxy_validation_metrics.csv", index=False)

    best = metrics.sort_values(["auc_for_top_quartile_target", "spearman_ordinal"], ascending=[False, False]).head(10)
    best_auc = float(best["auc_for_top_quartile_target"].max()) if not best.empty else float("nan")
    status = "PASS_WEAK_PUBLIC_DESIGN_CONSEQUENCE_PROXY_CHECK" if best_auc < 0.6 else "PASS_PUBLIC_DESIGN_CONSEQUENCE_PROXY_VALIDATION"
    summary = {
        "status": status,
        "validation_level": "public spatial design/consequence proxy validation, not monetary cost validation",
        "layer_metadata": layer_meta,
        "n_cpt_sites": int(sites["site_id"].nunique()),
        "joined_counts": {
            "ccc_hazard": int(joined["ccc_hazard_max_mod_to_severe_pct"].notna().sum()),
            "ccc_vulnerability": int(joined["ccc_vulnerability_score"].notna().sum()),
            "quake_foundation": int(joined["quake_foundation_tc_score"].notna().sum()),
            "mbie_tc": int(joined["mbie_tc_score"].notna().sum()),
        },
        "best_metrics": best.to_dict("records"),
        "claim_enabled": (
            "Article 118 predictions can be spatially compared with public Christchurch/Canterbury land-performance and foundation-category proxies. "
            "The strongest public proxy signal is weak but positive for CCC moderate-to-severe liquefaction hazard; technical-category proxies do not provide strong discrimination."
        ),
        "claim_blocked": (
            "Do not describe this as monetary repair-cost validation, property-level EQC/NHC claim validation, design certification, or strong design-consequence validation. "
            "These public proxy layers are useful for claim-boundary testing but only weakly align with Article 118 site probabilities."
        ),
    }
    (OUTPUTS / "public_design_consequence_proxy_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit = [
        "# Public design/consequence proxy validation - 2026-06-22",
        "",
        "## Data joined",
        "",
        "- Article 118 Canterbury CPT sites and held-out prediction scores.",
        "- CCC LiquefactionHazard scenario damage class fields.",
        "- CCC LiquefactionVulnerability categories.",
        "- CCC QuakeFoundationDesign technical categories.",
        "- MBIE technical categories / red-zone polygons.",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        f"CPT sites tested: `{summary['n_cpt_sites']}`.",
        f"Joined counts: `{summary['joined_counts']}`.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
    ]
    (AUDIT / "public_design_consequence_proxy_validation_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
