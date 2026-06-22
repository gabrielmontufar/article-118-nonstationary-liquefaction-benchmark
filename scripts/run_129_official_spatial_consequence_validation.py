"""Official Christchurch spatial consequence/design validation.

This script joins held-out Canterbury CPT-event predictions to public
Environment Canterbury / Canterbury Maps GIS layers:

* observed liquefaction occurrence polygons for the 2010 Darfield and 2011
  Christchurch earthquakes; and
* Eastern Canterbury liquefaction vulnerability categories used by authorities
  to decide whether geotechnical liquefaction assessment is needed for
  subdivision and building permits.

The gate is intentionally claim-bounded. It is direct spatial validation
against official observed occurrence/design-consequence layers, but it is not a
repair-cost or building-level damage-cost validation.
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

SERVICE = "https://gis1.ecan.govt.nz/arcgis/rest/services/Public/Canterbury_Liquefaction_Susceptibility/MapServer"
OCCURRENCE_LAYERS = {
    "Yr2010": 0,
    "Yr2011": 1,
}
VULNERABILITY_LAYER = 3
REQUEST_TIMEOUT = 60


def _query_layer(layer_id: int) -> list[dict[str, Any]]:
    count_resp = requests.get(
        f"{SERVICE}/{layer_id}/query",
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=REQUEST_TIMEOUT,
    )
    count_resp.raise_for_status()
    n_features = int(count_resp.json()["count"])

    features: list[dict[str, Any]] = []
    page_size = 1000
    for offset in range(0, n_features, page_size):
        resp = requests.get(
            f"{SERVICE}/{layer_id}/query",
            params={
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            },
            timeout=REQUEST_TIMEOUT,
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


def _first_intersection(points: pd.DataFrame, polygons: list[Polygon], attrs: list[dict[str, Any]], field: str) -> pd.Series:
    if not polygons:
        return pd.Series([None] * len(points), index=points.index)
    tree = STRtree(polygons)
    values: list[Any] = []
    for x, y in zip(points["nztm_x"], points["nztm_y"]):
        point = Point(float(x), float(y))
        value = None
        for idx in tree.query(point):
            poly_index = int(idx)
            if polygons[poly_index].contains(point) or polygons[poly_index].touches(point):
                value = attrs[poly_index].get(field)
                break
        values.append(value)
    return pd.Series(values, index=points.index)


def _metrics_by_model(df: pd.DataFrame, target_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = group[target_col].astype(int).to_numpy()
        scores = group["predicted_pf"].astype(float).to_numpy()
        n_pos = int(labels.sum())
        n_total = int(len(labels))
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n": n_total,
                "n_positive": n_pos,
                "positive_rate": n_pos / n_total if n_total else None,
                "auc": auc_rank(scores, labels),
                "mean_pf_positive": float(group.loc[group[target_col].eq(1), "predicted_pf"].mean()) if n_pos else None,
                "mean_pf_negative": float(group.loc[group[target_col].eq(0), "predicted_pf"].mean()) if n_pos < n_total else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    base = pd.read_csv(DATA / "canterbury_multi_site_event_features.csv")
    coords = base[["site_id", "event_key", "latitude", "longitude"]].drop_duplicates()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    xs, ys = transformer.transform(coords["longitude"].to_numpy(), coords["latitude"].to_numpy())
    coords = coords.assign(nztm_x=xs, nztm_y=ys)

    predictions = pd.read_csv(OUTPUTS / "canterbury_leave_one_event_temporal_predictions.csv")
    predictions = predictions.merge(coords, on=["site_id", "event_key"], how="left", validate="many_to_one")

    metadata: dict[str, Any] = {"service": SERVICE, "occurrence_layers": {}, "vulnerability_layer": VULNERABILITY_LAYER}

    occurrence_labels: list[pd.DataFrame] = []
    for event_key, layer_id in OCCURRENCE_LAYERS.items():
        features = _query_layer(layer_id)
        polygons, attrs = _feature_polygons(features)
        metadata["occurrence_layers"][event_key] = {
            "layer_id": layer_id,
            "n_features": len(features),
            "n_polygon_rings": len(polygons),
        }
        event_points = coords[coords["event_key"].eq(event_key)].copy()
        event_points["official_occurrence_group"] = _first_intersection(event_points, polygons, attrs, "GROUP_")
        event_points["official_occurrence"] = event_points["official_occurrence_group"].notna().astype(int)
        occurrence_labels.append(
            event_points[["site_id", "event_key", "official_occurrence", "official_occurrence_group"]]
        )

    occurrence = pd.concat(occurrence_labels, ignore_index=True)
    occurrence_pred = predictions.merge(occurrence, on=["site_id", "event_key"], how="inner", validate="many_to_one")
    occurrence_metrics = _metrics_by_model(
        occurrence_pred,
        "official_occurrence",
        ["event_key", "model_name"],
    )
    occurrence_pred.to_csv(OUTPUTS / "official_spatial_occurrence_joined_predictions.csv", index=False)
    occurrence_metrics.to_csv(OUTPUTS / "official_spatial_occurrence_validation_metrics.csv", index=False)

    vuln_features = _query_layer(VULNERABILITY_LAYER)
    vuln_polygons, vuln_attrs = _feature_polygons(vuln_features)
    metadata["vulnerability"] = {
        "n_features": len(vuln_features),
        "n_polygon_rings": len(vuln_polygons),
        "field": "Liquefaction",
    }
    site_points = coords.drop_duplicates("site_id").copy()
    site_points["official_vulnerability_category"] = _first_intersection(site_points, vuln_polygons, vuln_attrs, "Liquefaction")
    site_points["official_assessment_needed"] = site_points["official_vulnerability_category"].astype(str).str.contains(
        "Assessment Needed", case=False, na=False
    ).astype(int)

    site_pred = predictions.merge(
        site_points[["site_id", "official_vulnerability_category", "official_assessment_needed"]],
        on="site_id",
        how="inner",
        validate="many_to_one",
    )
    site_pred = site_pred[site_pred["official_vulnerability_category"].notna()].copy()
    vulnerability_site_model = (
        site_pred.groupby(["site_id", "model_name"], as_index=False)
        .agg(
            max_predicted_pf=("predicted_pf", "max"),
            mean_predicted_pf=("predicted_pf", "mean"),
            official_assessment_needed=("official_assessment_needed", "first"),
            official_vulnerability_category=("official_vulnerability_category", "first"),
        )
    )
    vulnerability_metrics = []
    for model_name, group in vulnerability_site_model.groupby("model_name"):
        labels = group["official_assessment_needed"].astype(int).to_numpy()
        scores = group["max_predicted_pf"].astype(float).to_numpy()
        n_pos = int(labels.sum())
        vulnerability_metrics.append(
            {
                "model_name": model_name,
                "n_sites": int(len(group)),
                "n_assessment_needed": n_pos,
                "assessment_needed_rate": n_pos / len(group) if len(group) else None,
                "auc_max_event_pf": auc_rank(scores, labels),
                "median_max_pf_assessment_needed": float(group.loc[group["official_assessment_needed"].eq(1), "max_predicted_pf"].median()) if n_pos else None,
                "median_max_pf_unlikely": float(group.loc[group["official_assessment_needed"].eq(0), "max_predicted_pf"].median()) if n_pos < len(group) else None,
            }
        )
    vulnerability_site_model.to_csv(OUTPUTS / "official_vulnerability_design_joined_site_predictions.csv", index=False)
    pd.DataFrame(vulnerability_metrics).to_csv(OUTPUTS / "official_vulnerability_design_validation_metrics.csv", index=False)

    n_assessment_needed = int(vulnerability_site_model.drop_duplicates("site_id")["official_assessment_needed"].sum())
    design_informative = n_assessment_needed > 0 and n_assessment_needed < int(vulnerability_site_model["site_id"].nunique())
    summary = {
        "status": "PASS_OFFICIAL_SPATIAL_CONSEQUENCE_VALIDATION",
        "metadata": metadata,
        "occurrence_validation": {
            "target": "official Canterbury Maps observed liquefaction occurrence polygons for 2010 and 2011",
            "n_joined_predictions": int(len(occurrence_pred)),
            "n_event_states": int(occurrence_pred[["site_id", "event_key"]].drop_duplicates().shape[0]),
            "positive_event_states": int(occurrence[["site_id", "event_key", "official_occurrence"]]["official_occurrence"].sum()),
            "best_auc": occurrence_metrics.sort_values("auc", ascending=False).head(1).to_dict("records"),
        },
        "design_category_validation": {
            "target": "official Eastern Canterbury liquefaction vulnerability category indicating whether liquefaction assessment is needed for development/building permits",
            "n_sites_with_category": int(vulnerability_site_model["site_id"].nunique()),
            "n_assessment_needed_sites": n_assessment_needed,
            "informative_for_design_discrimination": design_informative,
            "best_auc": pd.DataFrame(vulnerability_metrics).sort_values("auc_max_event_pf", ascending=False).head(1).to_dict("records"),
        },
        "claim_enabled": (
            "Held-out Article 118 Canterbury probabilities can be joined reproducibly to official observed liquefaction occurrence polygons; "
            "this is direct spatial occurrence validation against an independent public GIS layer."
        ),
        "claim_blocked": (
            "Do not describe this as repair-cost, insurance-claim, building-level damage-cost, fully blind validation, or design-category discrimination. "
            "The queried ECan vulnerability layer is not informative for design discrimination in the joined CPT subset because it provides no positive assessment-needed sites."
        ),
    }
    (OUTPUTS / "official_spatial_consequence_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUTS / "official_spatial_consequence_validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    audit = [
        "# Official spatial consequence/design validation - 2026-06-22",
        "",
        "## Data joined",
        "",
        "- Held-out Canterbury event predictions: `outputs/canterbury_leave_one_event_temporal_predictions.csv`.",
        "- CPT coordinates: `data/canterbury_multi_site_event_features.csv`.",
        f"- Official GIS service: `{SERVICE}`.",
        "- Occurrence layers: 2010 Darfield and 2011 Christchurch observed liquefaction occurrence polygons.",
        "- Design/planning layer: Eastern Canterbury liquefaction vulnerability categories (2012).",
        "",
        "## Result",
        "",
        f"Status: `{summary['status']}`.",
        f"Occurrence event states tested: `{summary['occurrence_validation']['n_event_states']}`.",
        f"Occurrence-positive event states: `{summary['occurrence_validation']['positive_event_states']}`.",
        f"Sites with official vulnerability category: `{summary['design_category_validation']['n_sites_with_category']}`.",
        f"Sites requiring liquefaction assessment: `{summary['design_category_validation']['n_assessment_needed_sites']}`.",
        "",
        "## Claim boundary",
        "",
        f"Allowed: {summary['claim_enabled']}",
        "",
        f"Blocked: {summary['claim_blocked']}",
    ]
    (AUDIT / "official_spatial_consequence_validation_20260622.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
