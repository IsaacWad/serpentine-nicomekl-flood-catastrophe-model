# ============================================================
# assign_segment_score.py
#
# Purpose:
# Assign a physical flood-risk score to each source segment.
#
# This file is run before the Monte Carlo simulation loop.
# It takes raw segment lines and returns scored segment lines.
# ============================================================

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString


def validate_segment_input(
    segments_gdf,
    segment_id_col="segment_id",
    event_source_col="event_source",
):
    """
    Validate that the input segment GeoDataFrame has the minimum
    structure needed for segment scoring.

    Required:
    - GeoDataFrame
    - valid CRS
    - geometry column
    - segment id column
    - event source column
    - non-empty geometries

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        Raw segment line data.

    segment_id_col : str
        Column containing unique segment IDs.

    event_source_col : str
        Column identifying event source type:
        coastal, riverine, interior_drainage, etc.

    Returns
    -------
    geopandas.GeoDataFrame
        A validated copy of the input GeoDataFrame.
    """

    if not isinstance(segments_gdf, gpd.GeoDataFrame):
        raise TypeError("segments_gdf must be a GeoDataFrame.")

    if segments_gdf.empty:
        raise ValueError("segments_gdf is empty.")

    if segments_gdf.crs is None:
        raise ValueError("segments_gdf must have a CRS.")

    required_cols = [segment_id_col, event_source_col, "geometry"]

    missing_cols = [
        col for col in required_cols
        if col not in segments_gdf.columns
    ]

    if missing_cols:
        raise ValueError(f"segments_gdf is missing columns: {missing_cols}")

    out = segments_gdf.copy()

    # Remove null or empty geometries
    out = out[
        out.geometry.notna()
        & (~out.geometry.is_empty)
    ].copy()

    if out.empty:
        raise ValueError("No valid segment geometries after removing null/empty geometries.")

    # Check for missing IDs or event sources
    if out[segment_id_col].isna().any():
        raise ValueError(f"{segment_id_col} contains missing values.")

    if out[event_source_col].isna().any():
        raise ValueError(f"{event_source_col} contains missing values.")

    # Check for duplicate segment IDs within event source
    duplicate_check = out.duplicated(
        subset=[event_source_col, segment_id_col],
        keep=False
    )

    if duplicate_check.any():
        duplicates = out.loc[
            duplicate_check,
            [event_source_col, segment_id_col]
        ].head(10)

        raise ValueError(
            "Duplicate segment IDs found within event source. "
            f"Examples:\n{duplicates}"
        )

    return out.reset_index(drop=True)



def classify_segment_context(
    row,
    event_source_col="event_source",
    low_elevation_col="low_elevation_share",
    near_dike_col="near_dike_500m_share",
    near_appurtenant_col="near_appurtenant_500m_share",
):
    """
    Classify one event-source segment using physical flood context.

    This function assumes the segment already has feature columns such as:
    - low_elevation_share
    - near_dike_500m_share
    - near_appurtenant_500m_share

    It returns:
    - protection_context
    - segment_score
    - segment_risk_class

    Important:
    This function does NOT use TIV or property exposure.
    Exposure affects loss later, not physical event severity.
    """

    event_source = row[event_source_col]

    low_elevation_share = row[low_elevation_col]
    near_dike_share = row[near_dike_col]
    near_appurtenant_share = row[near_appurtenant_col]

    # -----------------------------
    # 1. Protection context
    # -----------------------------

    if event_source in ["riverine", "coastal"]:
        if near_dike_share >= 0.50:
            protection_context = "protected"
        elif near_dike_share >= 0.20:
            protection_context = "partially_protected"
        else:
            protection_context = "limited_protection"

    elif event_source == "interior_drainage":
        if near_appurtenant_share >= 0.50:
            protection_context = "protected"
        elif near_appurtenant_share >= 0.20:
            protection_context = "partially_protected"
        else:
            protection_context = "limited_protection"

    else:
        protection_context = "unknown"

    # -----------------------------
    # 2. Physical segment score
    # -----------------------------

    segment_score = 0

    # Low elevation increases susceptibility
    if low_elevation_share >= 0.75:
        segment_score += 2
    elif low_elevation_share >= 0.50:
        segment_score += 1

    # Protection reduces severity potential for normal flood sources
    if event_source in ["riverine", "coastal", "interior_drainage"]:
        if protection_context == "protected":
            segment_score -= 1
        elif protection_context == "limited_protection":
            segment_score += 1

    # -----------------------------
    # 3. Floor score at 0
    # -----------------------------

    segment_score_raw = segment_score
    segment_score = max(segment_score, 0)

    # -----------------------------
    # 4. Convert score to risk class
    # -----------------------------

    if segment_score == 0:
        segment_risk_class = "low_risk"
    elif segment_score == 1:
        segment_risk_class = "standard"
    elif segment_score == 2:
        segment_risk_class = "elevated"
    else:
        segment_risk_class = "high_risk"


    return pd.Series({
    "protection_context": protection_context,
    "segment_score_raw": segment_score_raw,
    "segment_score": segment_score,
    "segment_risk_class": segment_risk_class,
    })


def calculate_near_protection_features(
    segments_gdf,
    dike_gdf=None,
    appurtenant_gdf=None,
    buffer_distance_m=500,
    segment_id_col="segment_id",
):
    """
    Calculate protection proximity features for each segment.

    For each segment, this computes:

    - near_dike_500m_share:
        share of segment length within buffer_distance_m of a dike

    - near_appurtenant_500m_share:
        share of segment length within buffer_distance_m of an appurtenant work

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        Segment line data.

    dike_gdf : geopandas.GeoDataFrame or None
        Dike line/polygon data. If None, near_dike share is set to 0.

    appurtenant_gdf : geopandas.GeoDataFrame or None
        Appurtenant works point/line/polygon data. If None, near_appurtenant share is set to 0.

    buffer_distance_m : float
        Distance in meters used to define "near" protection.

    segment_id_col : str
        Segment ID column.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of segments_gdf with protection proximity columns added.
    """

    out = segments_gdf.copy()

    if out.crs is None:
        raise ValueError("segments_gdf must have a CRS.")

    # Make sure we are working in projected CRS with meters
    out["segment_length_m"] = out.geometry.length

    if (out["segment_length_m"] <= 0).any():
        raise ValueError("Some segments have zero or negative length.")

    # --------------------------------------------------------
    # Helper: calculate share of segment inside protection buffer
    # --------------------------------------------------------

    def _calculate_share_near_layer(segment_geom, protection_buffer_geom):
        if protection_buffer_geom is None or protection_buffer_geom.is_empty:
            return 0.0

        intersection = segment_geom.intersection(protection_buffer_geom)

        if intersection.is_empty:
            return 0.0

        return intersection.length / segment_geom.length

    # --------------------------------------------------------
    # Dike buffer union
    # --------------------------------------------------------

    if dike_gdf is not None and not dike_gdf.empty:
        dike_local = dike_gdf.to_crs(out.crs).copy()
        dike_buffer_union = dike_local.geometry.buffer(buffer_distance_m).union_all()
    else:
        dike_buffer_union = None

    # --------------------------------------------------------
    # Appurtenant buffer union
    # --------------------------------------------------------

    if appurtenant_gdf is not None and not appurtenant_gdf.empty:
        appurtenant_local = appurtenant_gdf.to_crs(out.crs).copy()
        appurtenant_buffer_union = appurtenant_local.geometry.buffer(buffer_distance_m).union_all()
    else:
        appurtenant_buffer_union = None

    # --------------------------------------------------------
    # Calculate shares
    # --------------------------------------------------------

    out[f"near_dike_{buffer_distance_m:.0f}m_share"] = out.geometry.apply(
        lambda geom: _calculate_share_near_layer(
            segment_geom=geom,
            protection_buffer_geom=dike_buffer_union
        )
    )

    out[f"near_appurtenant_{buffer_distance_m:.0f}m_share"] = out.geometry.apply(
        lambda geom: _calculate_share_near_layer(
            segment_geom=geom,
            protection_buffer_geom=appurtenant_buffer_union
        )
    )

    return out


def calculate_low_elevation_share(
    segments_gdf,
    dem_path,
    low_elevation_threshold_m=2.0,
    sample_spacing_m=50,
):
    """
    Calculate the share of sampled points along each segment that are
    below or equal to a low-elevation threshold.

    Adds:
    - low_elevation_threshold_m
    - sample_spacing_m
    - sampled_point_count
    - valid_sampled_point_count
    - mean_segment_elevation_m
    - min_segment_elevation_m
    - low_elevation_share
    """

    if not isinstance(segments_gdf, gpd.GeoDataFrame):
        raise TypeError("segments_gdf must be a GeoDataFrame.")

    if segments_gdf.empty:
        raise ValueError("segments_gdf is empty.")

    if segments_gdf.crs is None:
        raise ValueError("segments_gdf must have a CRS.")

    out = segments_gdf.copy()

    def _sample_points_along_line(geom, spacing_m):
        """
        Sample points along a LineString or MultiLineString.
        """

        if geom is None or geom.is_empty:
            return []

        if geom.length <= 0:
            return []

        distances = np.arange(0, geom.length + spacing_m, spacing_m)

        points = []

        for distance in distances:
            try:
                points.append(geom.interpolate(min(distance, geom.length)))
            except Exception:
                continue

        return points

    sampled_point_counts = []
    valid_sampled_point_counts = []
    mean_elevations = []
    min_elevations = []
    low_elevation_shares = []

    with rasterio.open(dem_path) as dem_src:

        nodata = dem_src.nodata

        for geom in out.geometry:

            sample_points = _sample_points_along_line(
                geom=geom,
                spacing_m=sample_spacing_m,
            )

            sampled_point_counts.append(len(sample_points))

            if len(sample_points) == 0:
                valid_sampled_point_counts.append(0)
                mean_elevations.append(np.nan)
                min_elevations.append(np.nan)
                low_elevation_shares.append(0.0)
                continue

            coords = []

            for point in sample_points:
                if point is None or point.is_empty:
                    continue

                try:
                    coords.append((point.x, point.y))
                except Exception:
                    continue

            if len(coords) == 0:
                valid_sampled_point_counts.append(0)
                mean_elevations.append(np.nan)
                min_elevations.append(np.nan)
                low_elevation_shares.append(0.0)
                continue

            sampled_values = []

            for value in dem_src.sample(coords):
                z = float(value[0])

                if nodata is not None and np.isclose(z, nodata):
                    sampled_values.append(np.nan)
                else:
                    sampled_values.append(z)

            sampled_values = np.array(sampled_values, dtype=float)
            valid_values = sampled_values[np.isfinite(sampled_values)]

            valid_sampled_point_counts.append(len(valid_values))

            if len(valid_values) == 0:
                mean_elevations.append(np.nan)
                min_elevations.append(np.nan)
                low_elevation_shares.append(0.0)
                continue

            mean_elevations.append(float(np.mean(valid_values)))
            min_elevations.append(float(np.min(valid_values)))

            low_share = np.mean(valid_values <= low_elevation_threshold_m)
            low_elevation_shares.append(float(low_share))

    out["low_elevation_threshold_m"] = low_elevation_threshold_m
    out["sample_spacing_m"] = sample_spacing_m
    out["sampled_point_count"] = sampled_point_counts
    out["valid_sampled_point_count"] = valid_sampled_point_counts
    out["mean_segment_elevation_m"] = mean_elevations
    out["min_segment_elevation_m"] = min_elevations
    out["low_elevation_share"] = low_elevation_shares

    return out



def assign_segment_scores(
    segments_gdf,
    dem_path,
    dike_gdf=None,
    appurtenant_gdf=None,
    segment_id_col="segment_id",
    event_source_col="event_source",
    low_elevation_threshold_m=2.0,
    sample_spacing_m=50,
    protection_buffer_distance_m=500,
    crs="EPSG:3005",
):
    """
    Full wrapper for assigning physical segment scores.

    This function takes raw event-source segments and adds:
        - segment_length_m
        - near_dike_500m_share
        - near_appurtenant_500m_share
        - mean_segment_elevation_m
        - min_segment_elevation_m
        - low_elevation_share
        - protection_context
        - segment_score_raw
        - segment_score
        - segment_risk_class

    Important:
    This score is physical/event-source based.
    It does not use property TIV or losses.
    """

    # --------------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------------

    scored_segments_gdf = validate_segment_input(
        segments_gdf=segments_gdf,
        segment_id_col=segment_id_col,
        event_source_col=event_source_col,
    )

    scored_segments_gdf = scored_segments_gdf.to_crs(crs)

    # --------------------------------------------------------
    # 2. Calculate protection proximity features
    # --------------------------------------------------------

    scored_segments_gdf = calculate_near_protection_features(
        segments_gdf=scored_segments_gdf,
        dike_gdf=dike_gdf,
        appurtenant_gdf=appurtenant_gdf,
        buffer_distance_m=protection_buffer_distance_m,
        segment_id_col=segment_id_col,
    )

    # --------------------------------------------------------
    # 3. Calculate low-elevation share from DEM
    # --------------------------------------------------------

    scored_segments_gdf = calculate_low_elevation_share(
        segments_gdf=scored_segments_gdf,
        dem_path=dem_path,
        low_elevation_threshold_m=low_elevation_threshold_m,
        sample_spacing_m=sample_spacing_m,
    )

    # --------------------------------------------------------
    # 4. Classify / score each segment
    # --------------------------------------------------------

    near_dike_col = f"near_dike_{protection_buffer_distance_m:.0f}m_share"
    near_appurtenant_col = f"near_appurtenant_{protection_buffer_distance_m:.0f}m_share"

    segment_classification_df = scored_segments_gdf.apply(
        lambda row: classify_segment_context(
            row=row,
            event_source_col=event_source_col,
            low_elevation_col="low_elevation_share",
            near_dike_col=near_dike_col,
            near_appurtenant_col=near_appurtenant_col,
        ),
        axis=1,
    )

    scored_segments_gdf = scored_segments_gdf.join(segment_classification_df)

    # --------------------------------------------------------
    # 5. Create positive sampling weight for event selection
    # --------------------------------------------------------

    scored_segments_gdf["segment_selection_weight"] = (
        scored_segments_gdf["segment_score"] + 1
    )

    scored_segments_gdf["segment_selection_probability"] = (
        scored_segments_gdf["segment_selection_weight"]
        / scored_segments_gdf["segment_selection_weight"].sum()
    )

    return scored_segments_gdf