# Official spatial consequence/design validation - 2026-06-22

## Data joined

- Held-out Canterbury event predictions: `outputs/canterbury_leave_one_event_temporal_predictions.csv`.
- CPT coordinates: `data/canterbury_multi_site_event_features.csv`.
- Official GIS service: `https://gis1.ecan.govt.nz/arcgis/rest/services/Public/Canterbury_Liquefaction_Susceptibility/MapServer`.
- Occurrence layers: 2010 Darfield and 2011 Christchurch observed liquefaction occurrence polygons.
- Design/planning layer: Eastern Canterbury liquefaction vulnerability categories (2012).

## Result

Status: `PASS_OFFICIAL_SPATIAL_CONSEQUENCE_VALIDATION`.
Occurrence event states tested: `11100`.
Occurrence-positive event states: `4545`.
Sites with official vulnerability category: `5665`.
Sites requiring liquefaction assessment: `0`.

## Claim boundary

Allowed: Held-out Article 118 Canterbury probabilities can be joined reproducibly to official observed liquefaction occurrence polygons; this is direct spatial occurrence validation against an independent public GIS layer.

Blocked: Do not describe this as repair-cost, insurance-claim, building-level damage-cost, fully blind validation, or design-category discrimination. The queried ECan vulnerability layer is not informative for design discrimination in the joined CPT subset because it provides no positive assessment-needed sites.
