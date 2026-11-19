# Prepared inputs and provenance

The input bundle uses the completed local project's simulation-ready files in EPSG:3005 (BC Albers). The runner loads vectors in that CRS and requires the DEM to already use it. Coordinates and distances are in metres. The HAZUS damage module converts simulated metre depths to feet for interpolation.

| Input | Purpose |
| --- | --- |
| `raw_segments_clipped_to_flood_zone_clean_gdf.geojson` | 729 cleaned source segments used in the final notebook branch |
| `serpentine_nicomekl_hazard.geojson` | Study boundary; runner adds the notebook's 5 km expansion buffer |
| `serpentine_nicomekl_merged_dem_epsg3005.tif` | Prepared elevation raster |
| `dike_gdf.geojson` | Structural protection context for scoring |
| `appurtenant_gdf.geojson` | Drainage/protection context for scoring |
| `coastal_storage_side_mask.gpkg` | Coastal storage-side footprint constraint |
| `base_exposure_with_hazus.geojson` | 2,000 exposure geometries with building TIV and HAZUS mapping |
| `hazus_full_depth_damage_curves.csv` | Wide-format building depth-damage functions |

The packaged exposure layer retains only `exposure_id`, `building_tiv`, `hazus_occupancy`, and `hazus_a_zone_building_dmg_fn_id`, together with original geometry and CRS. Unused address, parcel, and assessment attributes were omitted. Feature order, geometry, and all retained model inputs are preserved. This is a reduced input table, not a synthetic dataset or anonymized spatial layer.

The original saved `scored_segments_gdf.geojson` came from an earlier notebook stage. The new runner therefore recomputes scores from the cleaned raw segments, matching the final 10,000-event notebook branch instead of assuming that earlier scored file describes the final run.

## Sources recorded in the report

- Government of British Columbia: [Mapped Floodplains in BC (Historical)](https://www.app.geo.ca/en-ca/map-browser/record/cdf4900e-90c0-449f-beea-43b669bd76a8).
- GeoBC: [Digital Elevation Model for British Columbia - CDED](https://open.canada.ca/data/en/dataset/7b4fef7e-7cae-4379-97b8-62b03e9ac83d).
- Government of British Columbia: [Flood Protection Works - Structural Works](https://app.geo.ca/en-ca/map-browser/record/15d7bca1-a4ab-4188-92ad-f86de271d8fb) and [Appurtenant Structures](https://app.geo.ca/en-ca/map-browser/record/4639cc3d-93dd-4ff2-ae77-0aae4f3cd588).
- City of Surrey: [Property Assessments Data and Maps](https://opendatasurrey.ca/property-assessments).
- City of Langley: [Zoning Open Data](https://maps.langleycity.ca/server/rest/services/Maps/Open_Data/FeatureServer/90/query).
- Township of Langley: [Assessments 2026](https://services5.arcgis.com/frpHL0Fv8koQRVWY/ArcGIS/rest/services/Assessments_2026/FeatureServer/0/query).
- FEMA: [HAZUS flood documentation and depth-damage functions](https://www.fema.gov/flood-maps/products-tools/hazus).

References are transcribed from the original report; they do not constitute a new license grant for third-party material. This repository includes prepared inputs for the simulation, not the full historical GIS preparation notebooks or a tested automatic rebuild from raw services. Raw source revisions may produce different exposure, segment, or elevation layers.

The optional R script extracts curve tables from the archived CRAN `hazus` 0.1 source package. It is retained as a preparation reference and is not needed to run the Python model because the prepared curves are bundled. Run it from the repository root with `dplyr` and `readr` installed. Its packaged copy uses the current directory rather than the author's computer path and does not install dependencies automatically. This extraction step was not rerun during packaging.
