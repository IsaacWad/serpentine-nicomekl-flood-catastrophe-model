# ============================================================
# flood_footprint/coastV2.py
# Coastal flood footprint V2
# ============================================================

import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

try:
    from .core import (
        clean_geometry,
        make_geometry_gdf,
        create_initial_flood_setup,
        sample_dem_elevation_at_point,
        initialize_current_water_zones_from_stage_1,
        sample_candidate_points_from_current_footprint,
        attach_nearest_pressure_zone_to_candidate_points,
        score_stage_candidate_points,
        select_active_candidate_points,
        assign_stage_expansion_distances,
        expand_active_points_with_dem_scan,
        allocate_water_to_expanded_points,
        build_expansion_lobes_from_expanded_points,
        build_next_stage_footprint_from_lobes_radial_scan,
        classify_radial_scan_boundary_segments,
        create_next_stage_polygons_from_boundary_segment_runs,
        assign_depths_to_next_stage_polygons_from_lobes,
    )
except ImportError:
    from core import (
        clean_geometry,
        make_geometry_gdf,
        create_initial_flood_setup,
        sample_dem_elevation_at_point,
        initialize_current_water_zones_from_stage_1,
        sample_candidate_points_from_current_footprint,
        attach_nearest_pressure_zone_to_candidate_points,
        score_stage_candidate_points,
        select_active_candidate_points,
        assign_stage_expansion_distances,
        expand_active_points_with_dem_scan,
        allocate_water_to_expanded_points,
        build_expansion_lobes_from_expanded_points,
        build_next_stage_footprint_from_lobes_radial_scan,
        classify_radial_scan_boundary_segments,
        create_next_stage_polygons_from_boundary_segment_runs,
        assign_depths_to_next_stage_polygons_from_lobes,
    )


# ============================================================
# Geometry helpers
# ============================================================

def _extract_single_geometry(geometry_or_gdf, name):
    """
    Return one valid geometry from a shapely geometry, GeoSeries, or GeoDataFrame.
    """

    if geometry_or_gdf is None:
        raise ValueError(f"{name} is None.")

    if hasattr(geometry_or_gdf, "geom_type"):
        geometry = geometry_or_gdf

    elif isinstance(geometry_or_gdf, gpd.GeoDataFrame):
        if geometry_or_gdf.empty:
            raise ValueError(f"{name} GeoDataFrame is empty.")
        geometry = _union_gdf_geometry(geometry_or_gdf)

    elif isinstance(geometry_or_gdf, gpd.GeoSeries):
        if geometry_or_gdf.empty:
            raise ValueError(f"{name} GeoSeries is empty.")
        geometry = unary_union(list(geometry_or_gdf))

    else:
        raise TypeError(
            f"{name} must be a shapely geometry, GeoSeries, or GeoDataFrame."
        )

    geometry = clean_geometry(geometry)

    if geometry is None or geometry.is_empty:
        raise ValueError(f"{name} geometry is empty.")

    return geometry


def _union_gdf_geometry(gdf):
    """
    GeoPandas union helper that works across older and newer versions.
    """

    valid_gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    if valid_gdf.empty:
        raise ValueError("Cannot union an empty GeoDataFrame.")

    try:
        return valid_gdf.geometry.union_all()
    except AttributeError:
        return unary_union(list(valid_gdf.geometry))


def _as_crs_gdf(gdf, crs, name):
    """
    Reproject a GeoDataFrame when needed.
    """

    if gdf is None or gdf.empty:
        raise ValueError(f"{name} is empty.")

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)

    return gdf.to_crs(crs).copy()


# ============================================================
# Static coastal context
# ============================================================

def build_coastal_v2_context(
    expansion_boundary,
    coastal_storage_gdf=None,
    coastal_storage_side_gdf=None,
    crs="EPSG:3005",
):
    """
    Build static coastal masks once before the simulation.

    Coastal V2 rule:
        The storage/bay side is a source-side mask for initialization.
        Initial event water is allocated only to landward wedges.

    Prefer coastal_storage_side_gdf when available because it represents the
    full bay/open-water side. Fall back to coastal_storage_gdf for simpler tests.
    """

    expansion_boundary_geometry = _extract_single_geometry(
        expansion_boundary,
        "expansion_boundary",
    )

    if coastal_storage_side_gdf is not None:
        storage_gdf = _as_crs_gdf(
            coastal_storage_side_gdf,
            crs,
            "coastal_storage_side_gdf",
        )
        storage_mask_source = "coastal_storage_side_gdf"

    elif coastal_storage_gdf is not None:
        storage_gdf = _as_crs_gdf(
            coastal_storage_gdf,
            crs,
            "coastal_storage_gdf",
        )
        storage_mask_source = "coastal_storage_gdf"

    else:
        raise ValueError(
            "Provide either coastal_storage_side_gdf or coastal_storage_gdf."
        )

    coastal_storage_mask = _union_gdf_geometry(storage_gdf)
    coastal_storage_mask = coastal_storage_mask.intersection(
        expansion_boundary_geometry
    )
    coastal_storage_mask = clean_geometry(coastal_storage_mask)

    if coastal_storage_mask is None or coastal_storage_mask.is_empty:
        raise ValueError("coastal_storage_mask is empty.")

    landward_domain_mask = expansion_boundary_geometry.difference(
        coastal_storage_mask
    )
    landward_domain_mask = clean_geometry(landward_domain_mask)

    if landward_domain_mask is None or landward_domain_mask.is_empty:
        raise ValueError("landward_domain_mask is empty.")

    mask_rows = [
        {
            "mask_type": "coastal_storage",
            "mask_source": storage_mask_source,
            "area_sqm": coastal_storage_mask.area,
            "area_sqkm": coastal_storage_mask.area / 1_000_000,
            "geometry": coastal_storage_mask,
        },
        {
            "mask_type": "landward_domain",
            "mask_source": "expansion_boundary_minus_coastal_storage",
            "area_sqm": landward_domain_mask.area,
            "area_sqkm": landward_domain_mask.area / 1_000_000,
            "geometry": landward_domain_mask,
        },
    ]

    masks_gdf = make_geometry_gdf(
        rows=mask_rows,
        crs=crs,
    )

    return {
        "crs": crs,
        "expansion_boundary_geometry": expansion_boundary_geometry,
        "coastal_storage_mask": coastal_storage_mask,
        "landward_domain_mask": landward_domain_mask,
        "masks_gdf": masks_gdf,
        "storage_mask_source": storage_mask_source,
    }


# ============================================================
# Initial coastal wedges
# ============================================================

def create_initial_coastal_wedges(
    selected_segment,
    expansion_boundary,
    initial_buffer_radius_m=250,
    n_stage_1_reference_points=16,
    crs="EPSG:3005",
):
    """
    Create the initial circular wedge stencil around the coastal segment.

    No water is allocated here. These wedges are diagnostic geometry until they
    are classified as landward or coastal storage.
    """

    initial_setup = create_initial_flood_setup(
        selected_segment=selected_segment,
        buffer_radius_m=initial_buffer_radius_m,
        expansion_boundary=expansion_boundary,
        n_reference_points=n_stage_1_reference_points,
        crs=crs,
    )

    source_point = initial_setup["source_point"]
    base_buffer_geometry = initial_setup["base_buffer_geometry"]

    reference_points_gdf = (
        initial_setup["reference_points_gdf"]
        .sort_values("angle_degrees")
        .reset_index(drop=True)
        .copy()
    )

    if reference_points_gdf.empty:
        raise ValueError("No initial reference points were created.")

    n_points = len(reference_points_gdf)
    max_radius_m = (
        reference_points_gdf["distance_from_source_m"].max() * 2
    )

    wedge_rows = []

    for wedge_index, row in reference_points_gdf.iterrows():

        current_angle = float(row["angle_degrees"])

        previous_angle = float(
            reference_points_gdf.iloc[wedge_index - 1]["angle_degrees"]
        )
        next_angle = float(
            reference_points_gdf.iloc[
                (wedge_index + 1) % n_points
            ]["angle_degrees"]
        )

        if wedge_index == 0:
            previous_angle = previous_angle - 360

        if wedge_index == n_points - 1:
            next_angle = next_angle + 360

        left_boundary_angle = (previous_angle + current_angle) / 2
        right_boundary_angle = (current_angle + next_angle) / 2

        left_rad = np.deg2rad(left_boundary_angle)
        center_rad = np.deg2rad(current_angle)
        right_rad = np.deg2rad(right_boundary_angle)

        left_point = Point(
            source_point.x + max_radius_m * np.cos(left_rad),
            source_point.y + max_radius_m * np.sin(left_rad),
        )
        center_outer_point = Point(
            source_point.x + max_radius_m * np.cos(center_rad),
            source_point.y + max_radius_m * np.sin(center_rad),
        )
        right_point = Point(
            source_point.x + max_radius_m * np.cos(right_rad),
            source_point.y + max_radius_m * np.sin(right_rad),
        )

        raw_wedge_geometry = Polygon(
            [
                (source_point.x, source_point.y),
                (left_point.x, left_point.y),
                (center_outer_point.x, center_outer_point.y),
                (right_point.x, right_point.y),
                (source_point.x, source_point.y),
            ]
        )

        wedge_geometry = raw_wedge_geometry.intersection(
            base_buffer_geometry
        )
        wedge_geometry = clean_geometry(wedge_geometry)

        if wedge_geometry is None or wedge_geometry.is_empty:
            continue

        boundary_midpoint = row.geometry

        wedge_rows.append(
            {
                "stage_id": 1,
                "initial_wedge_id": f"initial_wedge_{wedge_index}",
                "reference_id": int(row["reference_id"]),
                "angle_degrees": current_angle,
                "left_boundary_angle": left_boundary_angle % 360,
                "right_boundary_angle": right_boundary_angle % 360,
                "initial_buffer_radius_m": initial_buffer_radius_m,
                "wedge_area_sqm": wedge_geometry.area,
                "boundary_midpoint_x": boundary_midpoint.x,
                "boundary_midpoint_y": boundary_midpoint.y,
                "boundary_midpoint_distance_m": source_point.distance(
                    boundary_midpoint
                ),
                "geometry": wedge_geometry,
            }
        )

    initial_wedges_gdf = make_geometry_gdf(
        rows=wedge_rows,
        crs=crs,
    )

    if initial_wedges_gdf.empty:
        raise ValueError("No initial coastal wedges were created.")

    return {
        **initial_setup,
        "initial_wedges_gdf": initial_wedges_gdf,
    }


def classify_initial_coastal_wedges(
    initial_wedges_gdf,
    coastal_context,
    storage_overlap_threshold=0.99,
    crs="EPSG:3005",
):
    """
    Label each initial wedge as landward or coastal storage.
    """

    if initial_wedges_gdf is None or initial_wedges_gdf.empty:
        raise ValueError("initial_wedges_gdf is empty.")

    if not 0.0 <= storage_overlap_threshold <= 1.0:
        raise ValueError("storage_overlap_threshold must be between 0 and 1.")

    storage_mask = coastal_context["coastal_storage_mask"]
    landward_mask = coastal_context["landward_domain_mask"]

    wedges_gdf = initial_wedges_gdf.to_crs(crs).copy()

    wedges_gdf["wedge_area_sqm"] = wedges_gdf.geometry.area
    wedges_gdf["storage_overlap_area_sqm"] = (
        wedges_gdf.geometry.intersection(storage_mask).area
    )
    wedges_gdf["landward_overlap_area_sqm"] = (
        wedges_gdf.geometry.intersection(landward_mask).area
    )

    wedges_gdf["storage_overlap_share"] = (
        wedges_gdf["storage_overlap_area_sqm"]
        / wedges_gdf["wedge_area_sqm"].replace(0, np.nan)
    ).fillna(0.0)

    wedges_gdf["landward_overlap_share"] = (
        wedges_gdf["landward_overlap_area_sqm"]
        / wedges_gdf["wedge_area_sqm"].replace(0, np.nan)
    ).fillna(0.0)

    representative_points = wedges_gdf.geometry.representative_point()

    wedges_gdf["representative_point_in_storage"] = [
        storage_mask.covers(point)
        for point in representative_points
    ]

    wedges_gdf["stage_1_zone_type"] = np.where(
        (
            wedges_gdf["representative_point_in_storage"]
            | (
                wedges_gdf["storage_overlap_share"]
                >= storage_overlap_threshold
            )
        ),
        "coastal_storage",
        "landward_expansion",
    )

    wedges_gdf["is_initial_landward_wedge"] = (
        wedges_gdf["stage_1_zone_type"] == "landward_expansion"
    )
    wedges_gdf["storage_overlap_threshold"] = storage_overlap_threshold

    return gpd.GeoDataFrame(
        wedges_gdf,
        geometry="geometry",
        crs=crs,
    )


# ============================================================
# Initial landward points, scoring, and allocation
# ============================================================

def create_landward_wedge_midpoint_points(
    classified_initial_wedges_gdf,
    source_point,
    crs="EPSG:3005",
):
    """
    Create one scoring point from each initial landward wedge.

    The point is placed on the outer boundary of the initial buffer at the
    center angle of the wedge, not inside the wedge body.
    """

    if classified_initial_wedges_gdf is None:
        raise ValueError("classified_initial_wedges_gdf is None.")

    landward_wedges_gdf = classified_initial_wedges_gdf[
        classified_initial_wedges_gdf["is_initial_landward_wedge"]
    ].copy()

    if landward_wedges_gdf.empty:
        raise ValueError("No initial wedges were classified as landward.")

    point_rows = []

    for point_index, row in landward_wedges_gdf.iterrows():

        midpoint = Point(
            row["boundary_midpoint_x"],
            row["boundary_midpoint_y"],
        )

        dx = midpoint.x - source_point.x
        dy = midpoint.y - source_point.y

        angle_degrees = np.degrees(np.arctan2(dy, dx)) % 360

        point_rows.append(
            {
                "stage_id": 1,
                "stage_1_point_id": (
                    "landward_midpoint_"
                    + str(row["initial_wedge_id"])
                ),
                "initial_wedge_id": row["initial_wedge_id"],
                "reference_id": int(row["reference_id"]),
                "angle_degrees": angle_degrees,
                "source_to_point_distance_m": source_point.distance(midpoint),
                "point_position": "outer_boundary_midpoint",
                "wedge_area_sqm": float(row["wedge_area_sqm"]),
                "storage_overlap_share": float(
                    row["storage_overlap_share"]
                ),
                "landward_overlap_share": float(
                    row["landward_overlap_share"]
                ),
                "geometry": midpoint,
            }
        )

    return make_geometry_gdf(
        rows=point_rows,
        crs=crs,
    )


def score_landward_wedge_midpoints(
    landward_midpoint_points_gdf,
    source_point,
    dem_path,
    expansion_boundary,
    elevation_scale_m=1.0,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    min_score=0.01,
    crs="EPSG:3005",
):
    """
    Score one midpoint point per landward wedge.
    """

    if landward_midpoint_points_gdf is None or landward_midpoint_points_gdf.empty:
        raise ValueError("landward_midpoint_points_gdf is empty.")

    if random_low <= 0 or random_high <= 0 or random_low > random_high:
        raise ValueError("Random score bounds are invalid.")

    source_elevation_m = sample_dem_elevation_at_point(
        point=source_point,
        dem_path=dem_path,
    )

    expansion_boundary_geometry = _extract_single_geometry(
        expansion_boundary,
        "expansion_boundary",
    )

    rng = np.random.default_rng(random_seed)

    scored_gdf = landward_midpoint_points_gdf.to_crs(crs).copy()

    stage_1_scores = []
    point_elevations = []
    elevation_differences = []
    elevation_scores = []
    random_factors = []
    inside_flags = []

    for _, row in scored_gdf.iterrows():

        point = row.geometry

        inside_expansion_boundary = (
            expansion_boundary_geometry.contains(point)
            or expansion_boundary_geometry.touches(point)
        )

        point_elevation_m = sample_dem_elevation_at_point(
            point=point,
            dem_path=dem_path,
        )

        elevation_difference_m = point_elevation_m - source_elevation_m

        if (
            np.isnan(source_elevation_m)
            or np.isnan(point_elevation_m)
            or not inside_expansion_boundary
        ):
            elevation_score = min_score

        else:
            uphill_m = max(elevation_difference_m, 0)

            elevation_score = np.exp(
                -uphill_m / max(elevation_scale_m, 1e-6)
            )

            if elevation_difference_m < 0:
                downhill_bonus = min(
                    abs(elevation_difference_m)
                    / max(elevation_scale_m, 1e-6),
                    1.0,
                )
                elevation_score = elevation_score * (
                    1 + 0.25 * downhill_bonus
                )

        random_factor = rng.uniform(random_low, random_high)
        stage_1_score = max(elevation_score * random_factor, min_score)

        point_elevations.append(point_elevation_m)
        elevation_differences.append(elevation_difference_m)
        elevation_scores.append(elevation_score)
        random_factors.append(random_factor)
        stage_1_scores.append(stage_1_score)
        inside_flags.append(inside_expansion_boundary)

    scored_gdf["source_elevation_m"] = source_elevation_m
    scored_gdf["stage_1_point_elevation_m"] = point_elevations
    scored_gdf["elevation_difference_from_source_m"] = elevation_differences
    scored_gdf["inside_expansion_boundary"] = inside_flags
    scored_gdf["elevation_score"] = elevation_scores
    scored_gdf["random_factor"] = random_factors
    scored_gdf["stage_1_score"] = stage_1_scores

    total_score = float(scored_gdf["stage_1_score"].sum())

    if total_score <= 0 or np.isnan(total_score):
        scored_gdf["stage_1_score_share"] = 1.0 / len(scored_gdf)
    else:
        scored_gdf["stage_1_score_share"] = (
            scored_gdf["stage_1_score"] / total_score
        )

    return gpd.GeoDataFrame(
        scored_gdf,
        geometry="geometry",
        crs=crs,
    )


def allocate_initial_water_to_landward_wedges(
    classified_initial_wedges_gdf,
    scored_landward_midpoints_gdf,
    event_water_budget,
    severity_max_depth_m,
    depth_scale_factor=50_000,
    crs="EPSG:3005",
):
    """
    Allocate the full coastal event water budget to landward wedges only.
    """

    if event_water_budget <= 0:
        raise ValueError("event_water_budget must be greater than 0.")

    if severity_max_depth_m < 0:
        raise ValueError("severity_max_depth_m must be non-negative.")

    if depth_scale_factor <= 0:
        raise ValueError("depth_scale_factor must be greater than 0.")

    landward_wedges_gdf = classified_initial_wedges_gdf[
        classified_initial_wedges_gdf["is_initial_landward_wedge"]
    ].copy()

    if landward_wedges_gdf.empty:
        raise ValueError("No landward wedges are available for allocation.")

    allocation_cols = [
        "initial_wedge_id",
        "stage_1_point_id",
        "stage_1_score",
        "stage_1_score_share",
        "stage_1_point_elevation_m",
        "elevation_difference_from_source_m",
        "random_factor",
    ]

    allocated_gdf = landward_wedges_gdf.merge(
        scored_landward_midpoints_gdf[allocation_cols],
        on="initial_wedge_id",
        how="left",
        validate="one_to_one",
    )

    if allocated_gdf["stage_1_score_share"].isna().any():
        raise ValueError("Some landward wedges did not receive score shares.")

    allocated_gdf["segment_water_budget"] = (
        allocated_gdf["stage_1_score_share"] * event_water_budget
    )

    allocated_gdf["segment_area_sqm"] = allocated_gdf.geometry.area
    allocated_gdf["raw_depth_proxy"] = (
        allocated_gdf["segment_water_budget"]
        / allocated_gdf["segment_area_sqm"].clip(lower=1e-6)
    )
    allocated_gdf["max_depth_m"] = np.minimum(
        allocated_gdf["raw_depth_proxy"] * depth_scale_factor,
        severity_max_depth_m,
    )
    allocated_gdf["is_active_for_expansion"] = True
    allocated_gdf["stage_1_segment_id"] = allocated_gdf[
        "initial_wedge_id"
    ].str.replace("initial_wedge", "stage_1_segment", regex=False)

    final_total_water = float(allocated_gdf["segment_water_budget"].sum())
    water_balance_error = final_total_water - event_water_budget
    tolerance = max(1e-9, event_water_budget * 1e-9)

    if abs(water_balance_error) > tolerance:
        raise ValueError(
            "Initial coastal allocation does not conserve water. "
            f"Balance error: {water_balance_error}"
        )

    return gpd.GeoDataFrame(
        allocated_gdf,
        geometry="geometry",
        crs=crs,
    )


def build_initial_coastal_landward_setup(
    selected_segment,
    dem_path,
    expansion_boundary,
    coastal_storage_gdf=None,
    coastal_storage_side_gdf=None,
    event_water_budget=7.0,
    severity_max_depth_m=3.0,
    initial_buffer_radius_m=250,
    n_stage_1_reference_points=16,
    storage_overlap_threshold=0.99,
    elevation_scale_m=1.0,
    depth_scale_factor=50_000,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    crs="EPSG:3005",
):
    """
    First coastal V2 test block.

    Steps:
        1. Build initial flood buffer and wedge stencil.
        2. Label each wedge as coastal storage or landward expansion.
        3. Take one midpoint point from each landward wedge and score it.
        4. Allocate the full event water budget across landward wedges.
    """

    coastal_context = build_coastal_v2_context(
        expansion_boundary=expansion_boundary,
        coastal_storage_gdf=coastal_storage_gdf,
        coastal_storage_side_gdf=coastal_storage_side_gdf,
        crs=crs,
    )

    initial_setup = create_initial_coastal_wedges(
        selected_segment=selected_segment,
        expansion_boundary=coastal_context["expansion_boundary_geometry"],
        initial_buffer_radius_m=initial_buffer_radius_m,
        n_stage_1_reference_points=n_stage_1_reference_points,
        crs=crs,
    )

    classified_initial_wedges_gdf = classify_initial_coastal_wedges(
        initial_wedges_gdf=initial_setup["initial_wedges_gdf"],
        coastal_context=coastal_context,
        storage_overlap_threshold=storage_overlap_threshold,
        crs=crs,
    )

    landward_midpoint_points_gdf = create_landward_wedge_midpoint_points(
        classified_initial_wedges_gdf=classified_initial_wedges_gdf,
        source_point=initial_setup["source_point"],
        crs=crs,
    )

    scored_landward_midpoints_gdf = score_landward_wedge_midpoints(
        landward_midpoint_points_gdf=landward_midpoint_points_gdf,
        source_point=initial_setup["source_point"],
        dem_path=dem_path,
        expansion_boundary=coastal_context["expansion_boundary_geometry"],
        elevation_scale_m=elevation_scale_m,
        random_seed=random_seed,
        random_low=random_low,
        random_high=random_high,
        crs=crs,
    )

    stage_1_landward_depth_segments_gdf = (
        allocate_initial_water_to_landward_wedges(
            classified_initial_wedges_gdf=classified_initial_wedges_gdf,
            scored_landward_midpoints_gdf=scored_landward_midpoints_gdf,
            event_water_budget=event_water_budget,
            severity_max_depth_m=severity_max_depth_m,
            depth_scale_factor=depth_scale_factor,
            crs=crs,
        )
    )

    wedge_counts = (
        classified_initial_wedges_gdf["stage_1_zone_type"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "event_water_budget": float(event_water_budget),
        "total_initial_wedges": int(len(classified_initial_wedges_gdf)),
        "landward_initial_wedges": int(
            wedge_counts.get("landward_expansion", 0)
        ),
        "coastal_storage_initial_wedges": int(
            wedge_counts.get("coastal_storage", 0)
        ),
        "landward_allocated_water_budget": float(
            stage_1_landward_depth_segments_gdf[
                "segment_water_budget"
            ].sum()
        ),
        "storage_allocated_water_budget": 0.0,
        "water_balance_error": float(
            stage_1_landward_depth_segments_gdf[
                "segment_water_budget"
            ].sum()
            - event_water_budget
        ),
        "max_depth_m": float(
            stage_1_landward_depth_segments_gdf["max_depth_m"].max()
        ),
        "mean_depth_m": float(
            stage_1_landward_depth_segments_gdf["max_depth_m"].mean()
        ),
        "storage_overlap_threshold": float(storage_overlap_threshold),
        "storage_mask_source": coastal_context["storage_mask_source"],
    }

    return {
        "coastal_context": coastal_context,
        "source_point": initial_setup["source_point"],
        "segment_geometry": initial_setup["segment_geometry"],
        "base_buffer_geometry": initial_setup["base_buffer_geometry"],
        "base_buffer_gdf": initial_setup["base_buffer_gdf"],
        "reference_points_gdf": initial_setup["reference_points_gdf"],
        "reference_lines_gdf": initial_setup["reference_lines_gdf"],
        "classified_initial_wedges_gdf": classified_initial_wedges_gdf,
        "landward_midpoint_points_gdf": landward_midpoint_points_gdf,
        "scored_landward_midpoints_gdf": scored_landward_midpoints_gdf,
        "stage_1_landward_depth_segments_gdf": (
            stage_1_landward_depth_segments_gdf
        ),
        "initial_summary": summary,
    }


# ============================================================
# Stage 1 -> 2 expansion setup
# ============================================================

def build_stage_1_expansion_candidates_from_midpoints(
    scored_landward_midpoints_gdf,
    stage_1_landward_depth_segments_gdf,
    base_buffer_geometry,
    source_point,
    stage_id=1,
    crs="EPSG:3005",
):
    """
    Convert Stage 1 landward boundary midpoint points into core-compatible
    expansion candidates.

    In coastal V2, these are not newly sampled around the whole footprint.
    They are the one-per-landward-wedge points from the initial setup.
    """

    if scored_landward_midpoints_gdf is None or scored_landward_midpoints_gdf.empty:
        raise ValueError("scored_landward_midpoints_gdf is empty.")

    if (
        stage_1_landward_depth_segments_gdf is None
        or stage_1_landward_depth_segments_gdf.empty
    ):
        raise ValueError("stage_1_landward_depth_segments_gdf is empty.")

    if base_buffer_geometry is None or base_buffer_geometry.is_empty:
        raise ValueError("base_buffer_geometry is empty.")

    if source_point is None or source_point.is_empty:
        raise ValueError("source_point is empty.")

    required_point_cols = [
        "initial_wedge_id",
        "angle_degrees",
        "source_to_point_distance_m",
        "stage_1_score",
        "stage_1_score_share",
        "geometry",
    ]

    required_segment_cols = [
        "initial_wedge_id",
        "stage_1_segment_id",
        "segment_water_budget",
        "max_depth_m",
    ]

    for col in required_point_cols:
        if col not in scored_landward_midpoints_gdf.columns:
            raise ValueError(
                f"scored_landward_midpoints_gdf is missing column: {col}"
            )

    for col in required_segment_cols:
        if col not in stage_1_landward_depth_segments_gdf.columns:
            raise ValueError(
                "stage_1_landward_depth_segments_gdf is missing column: "
                f"{col}"
            )

    segment_lookup = stage_1_landward_depth_segments_gdf[
        required_segment_cols
    ].copy()

    candidates_gdf = scored_landward_midpoints_gdf.merge(
        segment_lookup,
        on="initial_wedge_id",
        how="left",
        validate="one_to_one",
    )

    if candidates_gdf["stage_1_segment_id"].isna().any():
        raise ValueError("Some Stage 1 candidate points lack parent segments.")

    boundary_length_m = float(base_buffer_geometry.boundary.length)

    candidates_gdf = candidates_gdf.sort_values(
        "angle_degrees"
    ).reset_index(drop=True)

    candidates_gdf["candidate_id"] = np.arange(len(candidates_gdf))
    candidates_gdf["stage_id"] = stage_id
    candidates_gdf[f"stage_{stage_id}_candidate_id"] = (
        "stage_"
        + str(stage_id)
        + "_candidate_"
        + candidates_gdf["candidate_id"].astype(str)
    )
    candidates_gdf["candidate_source"] = (
        "initial_landward_wedge_boundary_midpoint"
    )
    candidates_gdf["distance_from_source_m"] = candidates_gdf[
        "source_to_point_distance_m"
    ]

    candidates_gdf["boundary_position_share"] = (
        candidates_gdf["angle_degrees"] / 360.0
    )
    candidates_gdf["boundary_distance_m"] = (
        candidates_gdf["boundary_position_share"] * boundary_length_m
    )
    candidates_gdf["boundary_length_m"] = boundary_length_m

    candidates_gdf["parent_zone_id"] = candidates_gdf[
        "stage_1_segment_id"
    ].astype(str)
    candidates_gdf["parent_stage_origin"] = "stage_1"
    candidates_gdf["parent_water_budget"] = candidates_gdf[
        "segment_water_budget"
    ].astype(float)
    candidates_gdf["parent_max_depth_m"] = candidates_gdf[
        "max_depth_m"
    ].astype(float)

    candidates_gdf[f"stage_{stage_id}_candidate_score"] = candidates_gdf[
        "stage_1_score"
    ].astype(float)
    candidates_gdf[f"stage_{stage_id}_candidate_score_share"] = (
        candidates_gdf["stage_1_score_share"].astype(float)
    )

    return gpd.GeoDataFrame(
        candidates_gdf,
        geometry="geometry",
        crs=crs,
    )


def classify_coastal_v2_stage_polygons(
    stage_polygons_gdf,
    coastal_context,
    stage_id,
    storage_overlap_threshold=0.50,
    crs="EPSG:3005",
):
    """
    Classify completed non-initial stage polygons as landward or storage.
    """

    if stage_polygons_gdf is None or stage_polygons_gdf.empty:
        raise ValueError("stage_polygons_gdf is empty.")

    if not 0.0 <= storage_overlap_threshold <= 1.0:
        raise ValueError("storage_overlap_threshold must be between 0 and 1.")

    storage_mask = coastal_context["coastal_storage_mask"]
    landward_mask = coastal_context["landward_domain_mask"]

    polygons_gdf = stage_polygons_gdf.to_crs(crs).copy()
    polygons_gdf["polygon_area_sqm"] = polygons_gdf.geometry.area
    polygons_gdf["storage_overlap_area_sqm"] = (
        polygons_gdf.geometry.intersection(storage_mask).area
    )
    polygons_gdf["landward_overlap_area_sqm"] = (
        polygons_gdf.geometry.intersection(landward_mask).area
    )

    polygons_gdf["storage_overlap_share"] = (
        polygons_gdf["storage_overlap_area_sqm"]
        / polygons_gdf["polygon_area_sqm"].replace(0, np.nan)
    ).fillna(0.0)

    polygons_gdf["landward_overlap_share"] = (
        polygons_gdf["landward_overlap_area_sqm"]
        / polygons_gdf["polygon_area_sqm"].replace(0, np.nan)
    ).fillna(0.0)

    representative_points = polygons_gdf.geometry.representative_point()

    polygons_gdf["representative_point_in_storage"] = [
        storage_mask.covers(point)
        for point in representative_points
    ]

    zone_type_col = f"stage_{stage_id}_zone_type"

    polygons_gdf[zone_type_col] = np.where(
        (
            polygons_gdf["representative_point_in_storage"]
            | (
                polygons_gdf["storage_overlap_share"]
                >= storage_overlap_threshold
            )
        ),
        "coastal_storage",
        "landward_expansion",
    )

    polygons_gdf["is_active_for_expansion"] = (
        polygons_gdf[zone_type_col] == "landward_expansion"
    )
    polygons_gdf["storage_overlap_threshold"] = storage_overlap_threshold

    return gpd.GeoDataFrame(
        polygons_gdf,
        geometry="geometry",
        crs=crs,
    )


def fill_stage_polygons_with_assigned_lobes(
    next_stage_depth_polygons_gdf,
    expansion_lobes_gdf,
    lobe_assignment_df,
    new_area_geometry,
    stage_id,
    severity_max_depth_m,
    depth_scale_factor=50_000,
    min_depth_m=0.05,
    crs="EPSG:3005",
):
    """
    Ensure next-stage depth polygons cover the lobe geometry they received
    water from.

    The core radial-run method can create a corridor/fan polygon from the old
    footprint toward a lobe without including the full rounded lobe cap. Since
    the lobe water is assigned to that polygon, the assigned lobe's new area
    should be part of the depth polygon geometry too.
    """

    if (
        next_stage_depth_polygons_gdf is None
        or next_stage_depth_polygons_gdf.empty
    ):
        raise ValueError("next_stage_depth_polygons_gdf is empty.")

    if expansion_lobes_gdf is None or expansion_lobes_gdf.empty:
        raise ValueError("expansion_lobes_gdf is empty.")

    if lobe_assignment_df is None or lobe_assignment_df.empty:
        raise ValueError("lobe_assignment_df is empty.")

    if new_area_geometry is None or new_area_geometry.is_empty:
        raise ValueError("new_area_geometry is empty.")

    if depth_scale_factor <= 0:
        raise ValueError("depth_scale_factor must be greater than 0.")

    stage_next_id = stage_id + 1
    polygon_id_col = f"stage_{stage_next_id}_depth_polygon_id"
    lobe_id_col = f"stage_{stage_id}_expansion_lobe_id"

    required_polygon_cols = [
        polygon_id_col,
        "segment_water_budget",
        "geometry",
    ]

    required_lobe_cols = [
        lobe_id_col,
        "geometry",
    ]

    required_assignment_cols = [
        polygon_id_col,
        lobe_id_col,
    ]

    for col in required_polygon_cols:
        if col not in next_stage_depth_polygons_gdf.columns:
            raise ValueError(
                f"next_stage_depth_polygons_gdf is missing column: {col}"
            )

    for col in required_lobe_cols:
        if col not in expansion_lobes_gdf.columns:
            raise ValueError(f"expansion_lobes_gdf is missing column: {col}")

    for col in required_assignment_cols:
        if col not in lobe_assignment_df.columns:
            raise ValueError(f"lobe_assignment_df is missing column: {col}")

    polygons_gdf = next_stage_depth_polygons_gdf.to_crs(crs).copy()
    lobes_gdf = expansion_lobes_gdf.to_crs(crs).copy()

    lobe_geometry_by_id = {
        str(row[lobe_id_col]): row.geometry
        for _, row in lobes_gdf.iterrows()
    }

    assigned_lobes_by_polygon = (
        lobe_assignment_df
        .groupby(polygon_id_col)[lobe_id_col]
        .apply(lambda values: [str(value) for value in values])
        .to_dict()
    )

    filled_geometries = []
    original_areas = []
    filled_areas = []
    added_areas = []
    fill_methods = []

    for _, row in polygons_gdf.iterrows():

        original_geometry = clean_geometry(row.geometry)
        polygon_id = row[polygon_id_col]
        lobe_ids = assigned_lobes_by_polygon.get(polygon_id, [])

        fill_geometries = [original_geometry]

        for lobe_id in lobe_ids:

            lobe_geometry = lobe_geometry_by_id.get(str(lobe_id))

            if lobe_geometry is None or lobe_geometry.is_empty:
                continue

            lobe_new_area_geometry = lobe_geometry.intersection(
                new_area_geometry
            )
            lobe_new_area_geometry = clean_geometry(lobe_new_area_geometry)

            if (
                lobe_new_area_geometry is not None
                and not lobe_new_area_geometry.is_empty
            ):
                fill_geometries.append(lobe_new_area_geometry)

        filled_geometry = unary_union(fill_geometries)
        filled_geometry = clean_geometry(filled_geometry)

        original_area = (
            original_geometry.area
            if original_geometry is not None and not original_geometry.is_empty
            else 0.0
        )
        filled_area = (
            filled_geometry.area
            if filled_geometry is not None and not filled_geometry.is_empty
            else 0.0
        )

        filled_geometries.append(filled_geometry)
        original_areas.append(original_area)
        filled_areas.append(filled_area)
        added_areas.append(max(filled_area - original_area, 0.0))
        fill_methods.append(
            "radial_run_plus_assigned_lobe_geometry"
            if len(lobe_ids) > 0
            else "radial_run_only_no_assigned_lobe"
        )

    polygons_gdf["pre_lobe_fill_area_sqm"] = original_areas
    polygons_gdf["segment_area_sqm"] = filled_areas
    polygons_gdf["segment_area_sqkm"] = (
        polygons_gdf["segment_area_sqm"] / 1_000_000
    )
    polygons_gdf["lobe_fill_added_area_sqm"] = added_areas
    polygons_gdf["geometry_fill_method"] = fill_methods
    polygons_gdf["geometry"] = filled_geometries

    polygons_gdf["raw_depth_proxy"] = (
        polygons_gdf["segment_water_budget"]
        / polygons_gdf["segment_area_sqm"].clip(lower=1e-6)
    )
    polygons_gdf["max_depth_m"] = (
        polygons_gdf["raw_depth_proxy"] * depth_scale_factor
    ).clip(
        lower=min_depth_m,
        upper=severity_max_depth_m,
    )

    depth_union_geometry = _union_gdf_geometry(polygons_gdf)
    uncovered_new_area_geometry = new_area_geometry.difference(
        depth_union_geometry
    )
    uncovered_new_area_geometry = clean_geometry(uncovered_new_area_geometry)

    uncovered_new_area_sqm = (
        uncovered_new_area_geometry.area
        if (
            uncovered_new_area_geometry is not None
            and not uncovered_new_area_geometry.is_empty
        )
        else 0.0
    )

    assigned_uncovered_area_sqm = 0.0

    if uncovered_new_area_sqm > 1e-6:

        if uncovered_new_area_geometry.geom_type == "Polygon":
            uncovered_parts = [uncovered_new_area_geometry]
        elif uncovered_new_area_geometry.geom_type in [
            "MultiPolygon",
            "GeometryCollection",
        ]:
            uncovered_parts = [
                geom
                for geom in uncovered_new_area_geometry.geoms
                if (
                    geom.geom_type == "Polygon"
                    and not geom.is_empty
                    and geom.area > 1e-6
                )
            ]
        else:
            uncovered_parts = []

        for uncovered_part in uncovered_parts:

            assignment_rows = []

            for polygon_index, polygon_row in polygons_gdf.iterrows():

                polygon_geometry = clean_geometry(polygon_row.geometry)

                if polygon_geometry is None or polygon_geometry.is_empty:
                    continue

                assignment_rows.append(
                    {
                        "polygon_index": polygon_index,
                        "distance_m": uncovered_part.distance(
                            polygon_geometry
                        ),
                        "shared_boundary_m": uncovered_part.boundary
                        .intersection(polygon_geometry.boundary)
                        .length,
                    }
                )

            if len(assignment_rows) == 0:
                continue

            best_assignment = (
                pd.DataFrame(assignment_rows)
                .sort_values(
                    ["distance_m", "shared_boundary_m"],
                    ascending=[True, False],
                )
                .iloc[0]
            )

            target_index = int(best_assignment["polygon_index"])
            polygons_gdf.at[target_index, "geometry"] = clean_geometry(
                unary_union(
                    [
                        polygons_gdf.at[target_index, "geometry"],
                        uncovered_part,
                    ]
                )
            )
            assigned_uncovered_area_sqm += uncovered_part.area

        polygons_gdf["segment_area_sqm"] = polygons_gdf.geometry.area
        polygons_gdf["segment_area_sqkm"] = (
            polygons_gdf["segment_area_sqm"] / 1_000_000
        )
        polygons_gdf["raw_depth_proxy"] = (
            polygons_gdf["segment_water_budget"]
            / polygons_gdf["segment_area_sqm"].clip(lower=1e-6)
        )
        polygons_gdf["max_depth_m"] = (
            polygons_gdf["raw_depth_proxy"] * depth_scale_factor
        ).clip(
            lower=min_depth_m,
            upper=severity_max_depth_m,
        )

        final_depth_union_geometry = _union_gdf_geometry(polygons_gdf)
        remaining_uncovered_geometry = new_area_geometry.difference(
            final_depth_union_geometry
        )
        remaining_uncovered_geometry = clean_geometry(
            remaining_uncovered_geometry
        )
        remaining_uncovered_area_sqm = (
            remaining_uncovered_geometry.area
            if (
                remaining_uncovered_geometry is not None
                and not remaining_uncovered_geometry.is_empty
            )
            else 0.0
        )
    else:
        remaining_uncovered_geometry = uncovered_new_area_geometry
        remaining_uncovered_area_sqm = uncovered_new_area_sqm

    summary = {
        "pre_lobe_fill_depth_area_sqm": float(sum(original_areas)),
        "post_lobe_fill_depth_area_sqm": float(
            polygons_gdf["segment_area_sqm"].sum()
        ),
        "lobe_fill_added_area_sqm": float(sum(added_areas)),
        "new_area_sqm": float(new_area_geometry.area),
        "uncovered_new_area_after_lobe_fill_sqm": float(
            uncovered_new_area_sqm
        ),
        "uncovered_new_area_after_lobe_fill_share": float(
            uncovered_new_area_sqm / max(new_area_geometry.area, 1e-9)
        ),
        "uncovered_new_area_assigned_to_polygons_sqm": float(
            assigned_uncovered_area_sqm
        ),
        "remaining_uncovered_new_area_sqm": float(
            remaining_uncovered_area_sqm
        ),
        "remaining_uncovered_new_area_share": float(
            remaining_uncovered_area_sqm / max(new_area_geometry.area, 1e-9)
        ),
    }

    return {
        "filled_next_stage_depth_polygons_gdf": gpd.GeoDataFrame(
            polygons_gdf,
            geometry="geometry",
            crs=crs,
        ),
        "uncovered_new_area_geometry": remaining_uncovered_geometry,
        "uncovered_new_area_after_lobe_fill_geometry": (
            uncovered_new_area_geometry
        ),
        "lobe_fill_summary": summary,
    }


def update_coastal_v2_active_water_zones_after_stage(
    current_active_water_zones_gdf,
    parent_water_update_df,
    classified_next_stage_depth_polygons_gdf,
    stage_id,
    min_water_budget=0.001,
    min_depth_m=0.05,
    crs="EPSG:3005",
):
    """
    Update active water zones after a coastal V2 expansion stage.

    Coastal V2 rule:
        - existing active zones can remain active with reduced water
        - newly created landward polygons become active
        - newly created coastal_storage polygons are kept in the footprint/depth
          output but are excluded from future expansion sampling
    """

    if current_active_water_zones_gdf is None or current_active_water_zones_gdf.empty:
        raise ValueError("current_active_water_zones_gdf is empty.")

    if parent_water_update_df is None or parent_water_update_df.empty:
        raise ValueError("parent_water_update_df is empty.")

    if (
        classified_next_stage_depth_polygons_gdf is None
        or classified_next_stage_depth_polygons_gdf.empty
    ):
        raise ValueError("classified_next_stage_depth_polygons_gdf is empty.")

    stage_next_id = stage_id + 1
    next_stage_polygon_id_col = f"stage_{stage_next_id}_depth_polygon_id"
    next_stage_zone_type_col = f"stage_{stage_next_id}_zone_type"

    required_current_cols = [
        "current_zone_id",
        "current_stage_origin",
        "current_water_budget",
        "current_max_depth_m",
        "geometry",
    ]

    required_parent_update_cols = [
        "parent_zone_id",
        "parent_original_water_budget",
        "parent_remaining_water_budget",
    ]

    required_next_stage_cols = [
        next_stage_polygon_id_col,
        next_stage_zone_type_col,
        "is_active_for_expansion",
        "segment_water_budget",
        "max_depth_m",
        "geometry",
    ]

    for col in required_current_cols:
        if col not in current_active_water_zones_gdf.columns:
            raise ValueError(
                f"current_active_water_zones_gdf is missing column: {col}"
            )

    for col in required_parent_update_cols:
        if col not in parent_water_update_df.columns:
            raise ValueError(f"parent_water_update_df is missing column: {col}")

    for col in required_next_stage_cols:
        if col not in classified_next_stage_depth_polygons_gdf.columns:
            raise ValueError(
                "classified_next_stage_depth_polygons_gdf is missing column: "
                f"{col}"
            )

    current_gdf = current_active_water_zones_gdf.to_crs(crs).copy()
    next_stage_gdf = classified_next_stage_depth_polygons_gdf.to_crs(crs).copy()

    parent_update_df = parent_water_update_df.copy()
    parent_update_df["parent_zone_id"] = (
        parent_update_df["parent_zone_id"].astype(str)
    )

    parent_update_lookup = (
        parent_update_df
        .set_index("parent_zone_id")
        .to_dict(orient="index")
    )

    updated_rows = []

    for _, row in current_gdf.iterrows():

        current_zone_id = str(row["current_zone_id"])
        old_water_budget = float(row["current_water_budget"])
        old_max_depth_m = float(row["current_max_depth_m"])

        if current_zone_id in parent_update_lookup:

            parent_update = parent_update_lookup[current_zone_id]

            original_water_budget = float(
                parent_update["parent_original_water_budget"]
            )
            remaining_water_budget = float(
                parent_update["parent_remaining_water_budget"]
            )

            if original_water_budget > 0:
                remaining_share = remaining_water_budget / original_water_budget
            else:
                remaining_share = 0.0

            updated_max_depth_m = old_max_depth_m * remaining_share
            update_method = "reduced_after_outflow"

        else:
            remaining_water_budget = old_water_budget
            updated_max_depth_m = old_max_depth_m
            update_method = "unchanged_no_outflow"

        if remaining_water_budget < min_water_budget:
            continue

        if updated_max_depth_m < min_depth_m:
            continue

        updated_rows.append(
            {
                "current_zone_id": current_zone_id,
                "current_stage_origin": row["current_stage_origin"],
                "current_zone_type": "landward_expansion",
                "current_water_budget": remaining_water_budget,
                "current_max_depth_m": updated_max_depth_m,
                "zone_update_method": update_method,
                "geometry": row.geometry,
            }
        )

    active_next_stage_gdf = next_stage_gdf[
        next_stage_gdf["is_active_for_expansion"]
    ].copy()

    for _, row in active_next_stage_gdf.iterrows():

        new_water_budget = float(row["segment_water_budget"])
        new_max_depth_m = float(row["max_depth_m"])

        if new_water_budget < min_water_budget:
            continue

        if new_max_depth_m < min_depth_m:
            continue

        updated_rows.append(
            {
                "current_zone_id": str(row[next_stage_polygon_id_col]),
                "current_stage_origin": f"stage_{stage_next_id}",
                "current_zone_type": row[next_stage_zone_type_col],
                "current_water_budget": new_water_budget,
                "current_max_depth_m": new_max_depth_m,
                "zone_update_method": "new_landward_stage_polygon",
                "geometry": row.geometry,
            }
        )

    updated_active_water_zones_gdf = make_geometry_gdf(
        rows=updated_rows,
        crs=crs,
    )

    excluded_storage_water_budget = float(
        next_stage_gdf.loc[
            next_stage_gdf[next_stage_zone_type_col] == "coastal_storage",
            "segment_water_budget",
        ].sum()
    )

    summary = {
        "updated_active_water_zones": int(len(updated_active_water_zones_gdf)),
        "updated_active_water_total": float(
            updated_active_water_zones_gdf["current_water_budget"].sum()
            if not updated_active_water_zones_gdf.empty
            else 0.0
        ),
        "new_landward_active_zones": int(len(active_next_stage_gdf)),
        "new_storage_zones_excluded_from_active": int(
            (
                next_stage_gdf[next_stage_zone_type_col]
                == "coastal_storage"
            ).sum()
        ),
        "new_storage_water_excluded_from_active": excluded_storage_water_budget,
    }

    return {
        "updated_active_water_zones_gdf": updated_active_water_zones_gdf,
        "active_zone_update_summary": summary,
    }


def sample_coastal_v2_landward_candidate_points(
    current_active_water_zones_gdf,
    coastal_context,
    source_point,
    stage_id,
    current_total_footprint_geometry,
    n_candidate_points_per_zone=8,
    outer_distance_share=0.97,
    boundary_sample_spacing_m=5,
    boundary_match_tolerance_m=0.5,
    crs="EPSG:3005",
):
    """
    Sample candidate points from landward active-zone outer boundaries only.

    This is the Stage 2+ coastal sampler. It should be used after the previous
    stage's polygons have already been classified and storage polygons have
    already been excluded from updated_active_water_zones_gdf.
    """

    if (
        current_active_water_zones_gdf is None
        or current_active_water_zones_gdf.empty
    ):
        raise ValueError("current_active_water_zones_gdf is empty.")

    if source_point is None or source_point.is_empty:
        raise ValueError("source_point is empty.")

    if stage_id < 2:
        raise ValueError("Use this sampler for Stage 2+ candidate points.")

    if n_candidate_points_per_zone < 1:
        raise ValueError("n_candidate_points_per_zone must be at least 1.")

    if outer_distance_share <= 0 or outer_distance_share > 1:
        raise ValueError("outer_distance_share must be between 0 and 1.")

    if boundary_sample_spacing_m <= 0:
        raise ValueError("boundary_sample_spacing_m must be greater than 0.")

    if boundary_match_tolerance_m < 0:
        raise ValueError("boundary_match_tolerance_m cannot be negative.")

    storage_mask = coastal_context["coastal_storage_mask"]

    current_total_footprint_geometry = _extract_single_geometry(
        current_total_footprint_geometry,
        "current_total_footprint_geometry",
    )

    active_zones_gdf = current_active_water_zones_gdf.to_crs(crs).copy()

    required_cols = [
        "current_zone_id",
        "current_stage_origin",
        "current_water_budget",
        "current_max_depth_m",
        "geometry",
    ]

    for col in required_cols:
        if col not in active_zones_gdf.columns:
            raise ValueError(
                f"current_active_water_zones_gdf is missing column: {col}"
            )

    if "current_zone_type" in active_zones_gdf.columns:
        active_zones_gdf = active_zones_gdf[
            active_zones_gdf["current_zone_type"] == "landward_expansion"
        ].copy()

    if active_zones_gdf.empty:
        raise ValueError("No landward active zones are available to sample.")

    n_candidate_points = max(
        int(n_candidate_points_per_zone * len(active_zones_gdf)),
        int(n_candidate_points_per_zone),
    )

    total_boundary_candidates_gdf = sample_candidate_points_from_current_footprint(
        current_footprint_geometry=current_total_footprint_geometry,
        source_point=source_point,
        stage_id=stage_id,
        n_candidate_points=n_candidate_points,
        crs=crs,
    )

    landward_domain_mask = coastal_context["landward_domain_mask"]

    if boundary_match_tolerance_m > 0:
        landward_candidate_mask = landward_domain_mask.buffer(
            boundary_match_tolerance_m
        )
    else:
        landward_candidate_mask = landward_domain_mask

    total_boundary_candidates_gdf[
        "candidate_outside_coastal_storage"
    ] = total_boundary_candidates_gdf.geometry.apply(
        lambda point: not storage_mask.covers(point)
    )
    total_boundary_candidates_gdf[
        "candidate_in_landward_domain"
    ] = total_boundary_candidates_gdf.geometry.apply(
        lambda point: landward_candidate_mask.covers(point)
    )

    candidate_points_gdf = total_boundary_candidates_gdf[
        total_boundary_candidates_gdf["candidate_outside_coastal_storage"]
        & total_boundary_candidates_gdf["candidate_in_landward_domain"]
    ].copy()

    if candidate_points_gdf.empty:
        raise ValueError(
            "No Stage 2+ coastal candidates remain on the landward "
            "total-footprint boundary."
        )

    candidate_points_gdf = candidate_points_gdf.reset_index(drop=True)
    candidate_points_gdf["total_boundary_candidate_id"] = (
        candidate_points_gdf["candidate_id"].astype(int)
    )
    candidate_points_gdf["candidate_id"] = np.arange(len(candidate_points_gdf))

    candidate_points_with_pressure_gdf = (
        attach_nearest_pressure_zone_to_candidate_points(
            candidate_points_gdf=candidate_points_gdf,
            current_water_zones_gdf=active_zones_gdf,
            stage_id=stage_id,
            crs=crs,
        )
    )

    candidate_points_with_pressure_gdf["candidate_source"] = (
        "coastal_v2_total_footprint_boundary_landward_filtered"
    )
    candidate_points_with_pressure_gdf["source_current_zone_id"] = (
        candidate_points_with_pressure_gdf["parent_zone_id"].astype(str)
    )
    candidate_points_with_pressure_gdf["source_current_stage_origin"] = (
        candidate_points_with_pressure_gdf["parent_stage_origin"]
    )
    candidate_points_with_pressure_gdf["source_current_water_budget"] = (
        candidate_points_with_pressure_gdf["parent_water_budget"].astype(float)
    )
    candidate_points_with_pressure_gdf["source_current_max_depth_m"] = (
        candidate_points_with_pressure_gdf["parent_max_depth_m"].astype(float)
    )
    candidate_points_with_pressure_gdf["distance_to_coastal_storage_m"] = (
        candidate_points_with_pressure_gdf.geometry.distance(storage_mask)
    )
    candidate_points_with_pressure_gdf["boundary_sampling_source"] = (
        "current_total_footprint_boundary"
    )
    candidate_points_with_pressure_gdf[
        "used_current_total_footprint_boundary"
    ] = True
    candidate_points_with_pressure_gdf[
        "n_total_boundary_candidate_points"
    ] = int(n_candidate_points)
    candidate_points_with_pressure_gdf[
        "n_landward_boundary_candidate_points"
    ] = int(len(candidate_points_with_pressure_gdf))
    candidate_points_with_pressure_gdf["outer_distance_share"] = (
        outer_distance_share
    )
    candidate_points_with_pressure_gdf["boundary_sample_spacing_m"] = (
        boundary_sample_spacing_m
    )
    candidate_points_with_pressure_gdf["boundary_match_tolerance_m"] = (
        boundary_match_tolerance_m
    )

    return gpd.GeoDataFrame(
        candidate_points_with_pressure_gdf,
        geometry="geometry",
        crs=crs,
    )


def prepare_coastal_v2_stage_active_points(
    stage_candidate_points_gdf,
    source_point,
    dem_path,
    expansion_boundary,
    stage_id,
    severity_max_depth_m,
    elevation_scale_m=1.0,
    n_active_points=16,
    parent_penalty_strength=0.35,
    base_distance_m=225,
    min_distance_m=40,
    max_distance_m=425,
    score_power=0.50,
    depth_power=0.75,
    water_power=0.35,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    crs="EPSG:3005",
):
    """
    Score, select, and assign distances to coastal V2 Stage 2+ candidates.

    The coastal-specific work should already be done before this point:
    candidates must come from active landward zones only. This helper then
    delegates the generic DEM scoring, active-point selection, and expansion
    distance calculation to core.py.
    """

    if stage_candidate_points_gdf is None or stage_candidate_points_gdf.empty:
        raise ValueError("stage_candidate_points_gdf is empty.")

    if source_point is None or source_point.is_empty:
        raise ValueError("source_point is empty.")

    expansion_boundary = _extract_single_geometry(
        expansion_boundary,
        "expansion_boundary",
    )

    candidate_points_gdf = stage_candidate_points_gdf.to_crs(crs).copy()

    if "candidate_outside_coastal_storage" in candidate_points_gdf.columns:
        candidate_points_gdf = candidate_points_gdf[
            candidate_points_gdf["candidate_outside_coastal_storage"] == True
        ].copy()

    if candidate_points_gdf.empty:
        raise ValueError(
            "No coastal V2 candidates remain after storage filtering."
        )

    scored_candidate_points_gdf = score_stage_candidate_points(
        candidate_points_with_pressure_gdf=candidate_points_gdf,
        source_point=source_point,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        elevation_scale_m=elevation_scale_m,
        random_seed=random_seed,
        random_low=random_low,
        random_high=random_high,
        crs=crs,
    )

    active_points_gdf = select_active_candidate_points(
        scored_candidate_points_gdf=scored_candidate_points_gdf,
        stage_id=stage_id,
        n_active_points=n_active_points,
        parent_penalty_strength=parent_penalty_strength,
        crs=crs,
    )

    active_points_with_distance_gdf = assign_stage_expansion_distances(
        active_points_gdf=active_points_gdf,
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        base_distance_m=base_distance_m,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        score_power=score_power,
        depth_power=depth_power,
        water_power=water_power,
        crs=crs,
    )

    score_col = f"stage_{stage_id}_candidate_score"
    active_share_col = f"active_stage_{stage_id}_score_share"
    distance_col = f"stage_{stage_id}_expansion_distance_m"

    parent_zones_in_candidates = (
        candidate_points_gdf["parent_zone_id"].nunique()
        if "parent_zone_id" in candidate_points_gdf.columns
        else 0
    )
    parent_zones_in_active_points = (
        active_points_with_distance_gdf["parent_zone_id"].nunique()
        if "parent_zone_id" in active_points_with_distance_gdf.columns
        else 0
    )

    summary = {
        "stage_id": int(stage_id),
        "candidate_points": int(len(candidate_points_gdf)),
        "scored_candidate_points": int(len(scored_candidate_points_gdf)),
        "active_points": int(len(active_points_gdf)),
        "parent_zones_in_candidates": int(parent_zones_in_candidates),
        "parent_zones_in_active_points": int(parent_zones_in_active_points),
        "score_col": score_col,
        "active_score_share_col": active_share_col,
        "distance_col": distance_col,
        "max_candidate_score": float(scored_candidate_points_gdf[score_col].max()),
        "min_candidate_score": float(scored_candidate_points_gdf[score_col].min()),
        "active_score_share_sum": float(
            active_points_with_distance_gdf[active_share_col].sum()
        ),
        "min_expansion_distance_m": float(
            active_points_with_distance_gdf[distance_col].min()
        ),
        "mean_expansion_distance_m": float(
            active_points_with_distance_gdf[distance_col].mean()
        ),
        "max_expansion_distance_m": float(
            active_points_with_distance_gdf[distance_col].max()
        ),
        "n_active_points_requested": int(n_active_points),
        "parent_penalty_strength": float(parent_penalty_strength),
    }

    return {
        "summary": summary,
        "candidate_points_gdf": candidate_points_gdf,
        "scored_candidate_points_gdf": scored_candidate_points_gdf,
        "active_points_gdf": active_points_gdf,
        "active_points_with_distance_gdf": active_points_with_distance_gdf,
    }


def run_coastal_v2_expansion_stage_from_active_points(
    coastal_v2_initial,
    current_footprint_geometry,
    current_active_water_zones_gdf,
    active_points_with_distance_gdf,
    dem_path,
    stage_id,
    expansion_boundary=None,
    severity_max_depth_m=3.0,
    scan_left_degrees=11.25,
    scan_right_degrees=11.25,
    n_scan_angles=9,
    elevation_scale_m=1.0,
    outflow_share=0.75,
    point_influence_radius_m=75,
    connection_buffer_m=35,
    n_radial_wedges=96,
    min_outward_gain_m=5,
    depth_scale_factor=50_000,
    min_polygon_area_sqm=1_000,
    min_water_budget=0.001,
    min_depth_m=0.05,
    stage_storage_overlap_threshold=0.50,
    crs="EPSG:3005",
):
    """
    Run a coastal V2 expansion stage after active points are selected.

    This mirrors the proven riverine flow from DEM scan onward:
    active points -> DEM scan -> water allocation -> lobes -> radial footprint
    -> depth polygons. Coastal V2 then classifies the new polygons and keeps
    only landward polygons active for the next stage.
    """

    required_initial_keys = [
        "coastal_context",
        "source_point",
    ]

    for key in required_initial_keys:
        if key not in coastal_v2_initial:
            raise ValueError(f"coastal_v2_initial is missing key: {key}")

    if current_footprint_geometry is None or current_footprint_geometry.is_empty:
        raise ValueError("current_footprint_geometry is empty.")

    if (
        current_active_water_zones_gdf is None
        or current_active_water_zones_gdf.empty
    ):
        raise ValueError("current_active_water_zones_gdf is empty.")

    if (
        active_points_with_distance_gdf is None
        or active_points_with_distance_gdf.empty
    ):
        raise ValueError("active_points_with_distance_gdf is empty.")

    stage_next_id = stage_id + 1

    coastal_context = coastal_v2_initial["coastal_context"]
    source_point = coastal_v2_initial["source_point"]

    current_footprint_geometry = clean_geometry(current_footprint_geometry)

    if expansion_boundary is None:
        expansion_boundary = coastal_context["expansion_boundary_geometry"]
    else:
        expansion_boundary = _extract_single_geometry(
            expansion_boundary,
            "expansion_boundary",
        )

    dem_scan_result = expand_active_points_with_dem_scan(
        active_points_with_distance_gdf=active_points_with_distance_gdf,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        stage_id=stage_id,
        scan_left_degrees=scan_left_degrees,
        scan_right_degrees=scan_right_degrees,
        n_scan_angles=n_scan_angles,
        elevation_scale_m=elevation_scale_m,
        crs=crs,
    )

    water_allocation_result = allocate_water_to_expanded_points(
        expanded_points_gdf=dem_scan_result["expanded_points_gdf"],
        stage_id=stage_id,
        outflow_share=outflow_share,
        crs=crs,
    )

    lobe_result = build_expansion_lobes_from_expanded_points(
        expanded_points_with_water_gdf=water_allocation_result[
            "expanded_points_with_water_gdf"
        ],
        stage_id=stage_id,
        point_influence_radius_m=point_influence_radius_m,
        connection_buffer_m=connection_buffer_m,
        crs=crs,
    )

    footprint_result = build_next_stage_footprint_from_lobes_radial_scan(
        source_point=source_point,
        current_footprint_geometry=current_footprint_geometry,
        expansion_lobes_gdf=lobe_result["expansion_lobes_gdf"],
        expansion_boundary=expansion_boundary,
        stage_id=stage_id,
        n_wedges=n_radial_wedges,
        min_outward_gain_m=min_outward_gain_m,
        crs=crs,
    )

    boundary_segments_gdf = classify_radial_scan_boundary_segments(
        selected_outer_points_gdf=footprint_result[
            "selected_outer_points_gdf"
        ],
        stage_id=stage_id,
        crs=crs,
    )

    next_stage_polygons_gdf = create_next_stage_polygons_from_boundary_segment_runs(
        source_point=source_point,
        new_area_geometry=footprint_result["new_area_geometry"],
        selected_outer_points_gdf=footprint_result[
            "selected_outer_points_gdf"
        ],
        boundary_segments_gdf=boundary_segments_gdf,
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_area_sqm=min_polygon_area_sqm,
        crs=crs,
    )

    depth_assignment_result = assign_depths_to_next_stage_polygons_from_lobes(
        next_stage_polygons_gdf=next_stage_polygons_gdf,
        expansion_lobes_gdf=lobe_result["expansion_lobes_gdf"],
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    lobe_fill_result = fill_stage_polygons_with_assigned_lobes(
        next_stage_depth_polygons_gdf=depth_assignment_result[
            "next_stage_depth_polygons_gdf"
        ],
        expansion_lobes_gdf=lobe_result["expansion_lobes_gdf"],
        lobe_assignment_df=depth_assignment_result["lobe_assignment_df"],
        new_area_geometry=footprint_result["new_area_geometry"],
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    classified_next_stage_depth_polygons_gdf = (
        classify_coastal_v2_stage_polygons(
            stage_polygons_gdf=lobe_fill_result[
                "filled_next_stage_depth_polygons_gdf"
            ],
            coastal_context=coastal_context,
            stage_id=stage_next_id,
            storage_overlap_threshold=stage_storage_overlap_threshold,
            crs=crs,
        )
    )

    next_stage_zone_type_col = f"stage_{stage_next_id}_zone_type"
    next_stage_landward_depth_polygons_gdf = (
        classified_next_stage_depth_polygons_gdf[
            classified_next_stage_depth_polygons_gdf[next_stage_zone_type_col]
            == "landward_expansion"
        ].copy()
    )
    next_stage_storage_depth_polygons_gdf = (
        classified_next_stage_depth_polygons_gdf[
            classified_next_stage_depth_polygons_gdf[next_stage_zone_type_col]
            == "coastal_storage"
        ].copy()
    )

    active_zone_update_result = (
        update_coastal_v2_active_water_zones_after_stage(
            current_active_water_zones_gdf=current_active_water_zones_gdf,
            parent_water_update_df=water_allocation_result[
                "parent_water_update_df"
            ],
            classified_next_stage_depth_polygons_gdf=(
                classified_next_stage_depth_polygons_gdf
            ),
            stage_id=stage_id,
            min_water_budget=min_water_budget,
            min_depth_m=min_depth_m,
            crs=crs,
        )
    )

    summary = {
        "stage_id": int(stage_id),
        "stage_next_id": int(stage_next_id),
        "active_points": int(len(active_points_with_distance_gdf)),
        "expanded_points": int(len(dem_scan_result["expanded_points_gdf"])),
        "expansion_lobes": int(len(lobe_result["expansion_lobes_gdf"])),
        "next_stage_depth_polygons": int(
            len(classified_next_stage_depth_polygons_gdf)
        ),
        "next_stage_landward_polygons": int(
            len(next_stage_landward_depth_polygons_gdf)
        ),
        "next_stage_storage_polygons": int(
            len(next_stage_storage_depth_polygons_gdf)
        ),
        "current_area_sqkm": float(
            current_footprint_geometry.area / 1_000_000
        ),
        "next_area_sqkm": float(
            footprint_result["next_footprint_geometry"].area / 1_000_000
        ),
        "new_area_sqkm": float(
            footprint_result["new_area_geometry"].area / 1_000_000
        ),
        "current_active_water_budget": float(
            current_active_water_zones_gdf["current_water_budget"].sum()
        ),
        "parent_outflow_water_budget": float(
            water_allocation_result["parent_water_update_df"][
                "parent_outflow_water_budget"
            ].sum()
        ),
        "expanded_point_water_budget": float(
            water_allocation_result["expanded_points_with_water_gdf"][
                f"stage_{stage_id}_allocated_water_budget"
            ].sum()
        ),
        "lobe_water_budget": float(
            lobe_result["expansion_lobes_gdf"][
                "total_allocated_water_budget"
            ].sum()
        ),
        "next_stage_polygon_water_budget": float(
            classified_next_stage_depth_polygons_gdf[
                "segment_water_budget"
            ].sum()
        ),
        "next_stage_landward_water_budget": float(
            next_stage_landward_depth_polygons_gdf[
                "segment_water_budget"
            ].sum()
            if not next_stage_landward_depth_polygons_gdf.empty
            else 0.0
        ),
        "next_stage_storage_water_budget": float(
            next_stage_storage_depth_polygons_gdf[
                "segment_water_budget"
            ].sum()
            if not next_stage_storage_depth_polygons_gdf.empty
            else 0.0
        ),
        "stage_storage_overlap_threshold": float(
            stage_storage_overlap_threshold
        ),
        **lobe_fill_result["lobe_fill_summary"],
        **active_zone_update_result["active_zone_update_summary"],
    }

    summary["expanded_minus_parent_outflow_water_budget"] = (
        summary["expanded_point_water_budget"]
        - summary["parent_outflow_water_budget"]
    )
    summary["lobe_minus_expanded_water_budget"] = (
        summary["lobe_water_budget"]
        - summary["expanded_point_water_budget"]
    )
    summary["polygon_minus_lobe_water_budget"] = (
        summary["next_stage_polygon_water_budget"]
        - summary["lobe_water_budget"]
    )

    result = {
        "summary": summary,
        "current_footprint_geometry": current_footprint_geometry,
        "current_active_water_zones_gdf": current_active_water_zones_gdf,
        "active_points_with_distance_gdf": active_points_with_distance_gdf,
        "expanded_points_gdf": dem_scan_result["expanded_points_gdf"],
        "scan_lines_gdf": dem_scan_result["scan_lines_gdf"],
        "expanded_points_with_water_gdf": water_allocation_result[
            "expanded_points_with_water_gdf"
        ],
        "parent_water_update_df": water_allocation_result[
            "parent_water_update_df"
        ],
        "expanded_points_with_lobes_gdf": lobe_result[
            "expanded_points_with_lobes_gdf"
        ],
        "point_influence_gdf": lobe_result["point_influence_gdf"],
        "connection_lines_gdf": lobe_result["connection_lines_gdf"],
        "expansion_lobes_gdf": lobe_result["expansion_lobes_gdf"],
        "next_footprint_geometry": footprint_result[
            "next_footprint_geometry"
        ],
        "new_area_geometry": footprint_result["new_area_geometry"],
        "selected_outer_points_gdf": footprint_result[
            "selected_outer_points_gdf"
        ],
        "lobe_union_geometry": footprint_result["lobe_union_geometry"],
        "boundary_segments_gdf": boundary_segments_gdf,
        "next_stage_depth_polygons_gdf": (
            classified_next_stage_depth_polygons_gdf
        ),
        "next_stage_depth_polygons_before_lobe_fill_gdf": (
            depth_assignment_result["next_stage_depth_polygons_gdf"]
        ),
        "uncovered_new_area_geometry": lobe_fill_result[
            "uncovered_new_area_geometry"
        ],
        "lobe_fill_summary": lobe_fill_result["lobe_fill_summary"],
        "next_stage_landward_depth_polygons_gdf": (
            next_stage_landward_depth_polygons_gdf
        ),
        "next_stage_storage_depth_polygons_gdf": (
            next_stage_storage_depth_polygons_gdf
        ),
        "updated_active_water_zones_gdf": active_zone_update_result[
            "updated_active_water_zones_gdf"
        ],
        "active_zone_update_summary": active_zone_update_result[
            "active_zone_update_summary"
        ],
        "lobe_assignment_df": depth_assignment_result[
            "lobe_assignment_df"
        ],
    }

    result[f"stage_{stage_next_id}_depth_polygons_gdf"] = (
        classified_next_stage_depth_polygons_gdf
    )
    result[f"stage_{stage_next_id}_landward_depth_polygons_gdf"] = (
        next_stage_landward_depth_polygons_gdf
    )
    result[f"stage_{stage_next_id}_storage_depth_polygons_gdf"] = (
        next_stage_storage_depth_polygons_gdf
    )

    return result


def run_stage_1_to_2_coastal_v2_expansion(
    coastal_v2_initial,
    dem_path,
    expansion_boundary=None,
    severity_max_depth_m=3.0,
    n_active_points=16,
    parent_penalty_strength=0.35,
    base_distance_m=225,
    min_distance_m=40,
    max_distance_m=425,
    score_power=0.50,
    depth_power=0.75,
    water_power=0.35,
    scan_left_degrees=11.25,
    scan_right_degrees=11.25,
    n_scan_angles=9,
    elevation_scale_m=1.0,
    outflow_share=0.75,
    point_influence_radius_m=75,
    connection_buffer_m=35,
    n_radial_wedges=96,
    min_outward_gain_m=5,
    depth_scale_factor=50_000,
    min_polygon_area_sqm=1_000,
    min_water_budget=0.001,
    min_depth_m=0.05,
    stage_storage_overlap_threshold=0.50,
    crs="EPSG:3005",
):
    """
    Run the first repeatable coastal V2 expansion slice: Stage 1 -> Stage 2.

    This function intentionally starts from the landward boundary midpoint
    points created during initialization. It does not resample the whole initial
    footprint boundary.
    """

    required_keys = [
        "coastal_context",
        "source_point",
        "base_buffer_geometry",
        "scored_landward_midpoints_gdf",
        "stage_1_landward_depth_segments_gdf",
    ]

    for key in required_keys:
        if key not in coastal_v2_initial:
            raise ValueError(f"coastal_v2_initial is missing key: {key}")

    stage_id = 1
    stage_next_id = 2

    coastal_context = coastal_v2_initial["coastal_context"]
    source_point = coastal_v2_initial["source_point"]
    base_buffer_geometry = coastal_v2_initial["base_buffer_geometry"]
    stage_1_landward_depth_segments_gdf = coastal_v2_initial[
        "stage_1_landward_depth_segments_gdf"
    ]

    if expansion_boundary is None:
        expansion_boundary = coastal_context["expansion_boundary_geometry"]
    else:
        expansion_boundary = _extract_single_geometry(
            expansion_boundary,
            "expansion_boundary",
        )

    current_active_water_zones_gdf = initialize_current_water_zones_from_stage_1(
        stage_1_depth_segments_gdf=stage_1_landward_depth_segments_gdf,
        min_water_budget=min_water_budget,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    # Use the full initial buffer as the geometric subtraction footprint.
    # Active water remains landward-only, but the old bay-side buffer area
    # should not be treated as newly flooded Stage 2 area.
    current_footprint_geometry = base_buffer_geometry
    current_footprint_geometry = clean_geometry(current_footprint_geometry)

    stage_1_candidate_points_gdf = (
        build_stage_1_expansion_candidates_from_midpoints(
            scored_landward_midpoints_gdf=coastal_v2_initial[
                "scored_landward_midpoints_gdf"
            ],
            stage_1_landward_depth_segments_gdf=(
                stage_1_landward_depth_segments_gdf
            ),
            base_buffer_geometry=base_buffer_geometry,
            source_point=source_point,
            stage_id=stage_id,
            crs=crs,
        )
    )

    active_stage_1_points_gdf = select_active_candidate_points(
        scored_candidate_points_gdf=stage_1_candidate_points_gdf,
        stage_id=stage_id,
        n_active_points=n_active_points,
        parent_penalty_strength=parent_penalty_strength,
        crs=crs,
    )

    active_points_with_distance_gdf = assign_stage_expansion_distances(
        active_points_gdf=active_stage_1_points_gdf,
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        base_distance_m=base_distance_m,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        score_power=score_power,
        depth_power=depth_power,
        water_power=water_power,
        crs=crs,
    )

    dem_scan_result = expand_active_points_with_dem_scan(
        active_points_with_distance_gdf=active_points_with_distance_gdf,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        stage_id=stage_id,
        scan_left_degrees=scan_left_degrees,
        scan_right_degrees=scan_right_degrees,
        n_scan_angles=n_scan_angles,
        elevation_scale_m=elevation_scale_m,
        crs=crs,
    )

    water_allocation_result = allocate_water_to_expanded_points(
        expanded_points_gdf=dem_scan_result["expanded_points_gdf"],
        stage_id=stage_id,
        outflow_share=outflow_share,
        crs=crs,
    )

    lobe_result = build_expansion_lobes_from_expanded_points(
        expanded_points_with_water_gdf=water_allocation_result[
            "expanded_points_with_water_gdf"
        ],
        stage_id=stage_id,
        point_influence_radius_m=point_influence_radius_m,
        connection_buffer_m=connection_buffer_m,
        crs=crs,
    )

    footprint_result = build_next_stage_footprint_from_lobes_radial_scan(
        source_point=source_point,
        current_footprint_geometry=current_footprint_geometry,
        expansion_lobes_gdf=lobe_result["expansion_lobes_gdf"],
        expansion_boundary=expansion_boundary,
        stage_id=stage_id,
        n_wedges=n_radial_wedges,
        min_outward_gain_m=min_outward_gain_m,
        crs=crs,
    )

    boundary_segments_gdf = classify_radial_scan_boundary_segments(
        selected_outer_points_gdf=footprint_result[
            "selected_outer_points_gdf"
        ],
        stage_id=stage_id,
        crs=crs,
    )

    stage_2_polygons_gdf = create_next_stage_polygons_from_boundary_segment_runs(
        source_point=source_point,
        new_area_geometry=footprint_result["new_area_geometry"],
        selected_outer_points_gdf=footprint_result[
            "selected_outer_points_gdf"
        ],
        boundary_segments_gdf=boundary_segments_gdf,
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_area_sqm=min_polygon_area_sqm,
        crs=crs,
    )

    depth_assignment_result = assign_depths_to_next_stage_polygons_from_lobes(
        next_stage_polygons_gdf=stage_2_polygons_gdf,
        expansion_lobes_gdf=lobe_result["expansion_lobes_gdf"],
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    lobe_fill_result = fill_stage_polygons_with_assigned_lobes(
        next_stage_depth_polygons_gdf=depth_assignment_result[
            "next_stage_depth_polygons_gdf"
        ],
        expansion_lobes_gdf=lobe_result["expansion_lobes_gdf"],
        lobe_assignment_df=depth_assignment_result["lobe_assignment_df"],
        new_area_geometry=footprint_result["new_area_geometry"],
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    classified_stage_2_depth_polygons_gdf = (
        classify_coastal_v2_stage_polygons(
            stage_polygons_gdf=lobe_fill_result[
                "filled_next_stage_depth_polygons_gdf"
            ],
            coastal_context=coastal_context,
            stage_id=stage_next_id,
            storage_overlap_threshold=stage_storage_overlap_threshold,
            crs=crs,
        )
    )

    stage_2_zone_type_col = f"stage_{stage_next_id}_zone_type"
    stage_2_landward_depth_polygons_gdf = (
        classified_stage_2_depth_polygons_gdf[
            classified_stage_2_depth_polygons_gdf[stage_2_zone_type_col]
            == "landward_expansion"
        ].copy()
    )
    stage_2_storage_depth_polygons_gdf = (
        classified_stage_2_depth_polygons_gdf[
            classified_stage_2_depth_polygons_gdf[stage_2_zone_type_col]
            == "coastal_storage"
        ].copy()
    )

    active_zone_update_result = (
        update_coastal_v2_active_water_zones_after_stage(
            current_active_water_zones_gdf=current_active_water_zones_gdf,
            parent_water_update_df=water_allocation_result[
                "parent_water_update_df"
            ],
            classified_next_stage_depth_polygons_gdf=(
                classified_stage_2_depth_polygons_gdf
            ),
            stage_id=stage_id,
            min_water_budget=min_water_budget,
            min_depth_m=min_depth_m,
            crs=crs,
        )
    )

    summary = {
        "stage_id": stage_id,
        "stage_next_id": stage_next_id,
        "stage_1_candidate_points": int(len(stage_1_candidate_points_gdf)),
        "active_points": int(len(active_stage_1_points_gdf)),
        "expanded_points": int(len(dem_scan_result["expanded_points_gdf"])),
        "expansion_lobes": int(len(lobe_result["expansion_lobes_gdf"])),
        "stage_2_depth_polygons": int(
            len(classified_stage_2_depth_polygons_gdf)
        ),
        "stage_2_landward_polygons": int(
            len(stage_2_landward_depth_polygons_gdf)
        ),
        "stage_2_storage_polygons": int(
            len(stage_2_storage_depth_polygons_gdf)
        ),
        "stage_1_active_water_budget": float(
            current_active_water_zones_gdf["current_water_budget"].sum()
        ),
        "stage_1_parent_outflow_water_budget": float(
            water_allocation_result["parent_water_update_df"][
                "parent_outflow_water_budget"
            ].sum()
        ),
        "expanded_point_water_budget": float(
            water_allocation_result["expanded_points_with_water_gdf"][
                f"stage_{stage_id}_allocated_water_budget"
            ].sum()
        ),
        "lobe_water_budget": float(
            lobe_result["expansion_lobes_gdf"][
                "total_allocated_water_budget"
            ].sum()
        ),
        "stage_2_polygon_water_budget": float(
            classified_stage_2_depth_polygons_gdf[
                "segment_water_budget"
            ].sum()
        ),
        "stage_2_landward_water_budget": float(
            stage_2_landward_depth_polygons_gdf[
                "segment_water_budget"
            ].sum()
            if not stage_2_landward_depth_polygons_gdf.empty
            else 0.0
        ),
        "stage_2_storage_water_budget": float(
            stage_2_storage_depth_polygons_gdf[
                "segment_water_budget"
            ].sum()
            if not stage_2_storage_depth_polygons_gdf.empty
            else 0.0
        ),
        "stage_storage_overlap_threshold": float(
            stage_storage_overlap_threshold
        ),
        **lobe_fill_result["lobe_fill_summary"],
        **active_zone_update_result["active_zone_update_summary"],
    }

    summary["expanded_minus_parent_outflow_water_budget"] = (
        summary["expanded_point_water_budget"]
        - summary["stage_1_parent_outflow_water_budget"]
    )
    summary["lobe_minus_expanded_water_budget"] = (
        summary["lobe_water_budget"]
        - summary["expanded_point_water_budget"]
    )
    summary["polygon_minus_lobe_water_budget"] = (
        summary["stage_2_polygon_water_budget"]
        - summary["lobe_water_budget"]
    )

    return {
        "summary": summary,
        "current_footprint_geometry": current_footprint_geometry,
        "current_active_water_zones_gdf": current_active_water_zones_gdf,
        "stage_1_candidate_points_gdf": stage_1_candidate_points_gdf,
        "active_stage_1_points_gdf": active_stage_1_points_gdf,
        "active_points_with_distance_gdf": active_points_with_distance_gdf,
        "expanded_points_gdf": dem_scan_result["expanded_points_gdf"],
        "scan_lines_gdf": dem_scan_result["scan_lines_gdf"],
        "expanded_points_with_water_gdf": water_allocation_result[
            "expanded_points_with_water_gdf"
        ],
        "parent_water_update_df": water_allocation_result[
            "parent_water_update_df"
        ],
        "expanded_points_with_lobes_gdf": lobe_result[
            "expanded_points_with_lobes_gdf"
        ],
        "point_influence_gdf": lobe_result["point_influence_gdf"],
        "connection_lines_gdf": lobe_result["connection_lines_gdf"],
        "expansion_lobes_gdf": lobe_result["expansion_lobes_gdf"],
        "next_footprint_geometry": footprint_result[
            "next_footprint_geometry"
        ],
        "new_area_geometry": footprint_result["new_area_geometry"],
        "selected_outer_points_gdf": footprint_result[
            "selected_outer_points_gdf"
        ],
        "lobe_union_geometry": footprint_result["lobe_union_geometry"],
        "boundary_segments_gdf": boundary_segments_gdf,
        "stage_2_depth_polygons_gdf": classified_stage_2_depth_polygons_gdf,
        "stage_2_depth_polygons_before_lobe_fill_gdf": (
            depth_assignment_result["next_stage_depth_polygons_gdf"]
        ),
        "uncovered_new_area_geometry": lobe_fill_result[
            "uncovered_new_area_geometry"
        ],
        "lobe_fill_summary": lobe_fill_result["lobe_fill_summary"],
        "stage_2_landward_depth_polygons_gdf": (
            stage_2_landward_depth_polygons_gdf
        ),
        "stage_2_storage_depth_polygons_gdf": (
            stage_2_storage_depth_polygons_gdf
        ),
        "updated_active_water_zones_gdf": active_zone_update_result[
            "updated_active_water_zones_gdf"
        ],
        "active_zone_update_summary": active_zone_update_result[
            "active_zone_update_summary"
        ],
        "lobe_assignment_df": depth_assignment_result[
            "lobe_assignment_df"
        ],
    }


def build_coastal_v2_flood_footprint(
    selected_segment,
    dem_path,
    expansion_boundary,
    coastal_storage_gdf=None,
    coastal_storage_side_gdf=None,
    severity="severe",
    event_water_budget=7.0,
    severity_max_depth_m=3.0,
    n_growth_stages=5,
    initial_buffer_radius_m=250,
    n_stage_1_reference_points=16,
    initial_storage_overlap_threshold=0.99,
    stage_storage_overlap_threshold=0.50,
    elevation_scale_m=1.0,
    depth_scale_factor=50_000,
    n_candidate_points_per_zone=8,
    outer_distance_share=0.97,
    boundary_sample_spacing_m=5,
    boundary_match_tolerance_m=0.5,
    n_active_points=16,
    parent_penalty_strength=0.35,
    base_distance_m=225,
    min_distance_m=40,
    max_distance_m=425,
    score_power=0.50,
    depth_power=0.75,
    water_power=0.35,
    scan_left_degrees=11.25,
    scan_right_degrees=11.25,
    n_scan_angles=9,
    outflow_share=0.75,
    point_influence_radius_m=75,
    connection_buffer_m=35,
    n_radial_wedges=96,
    min_outward_gain_m=5,
    min_polygon_area_sqm=1_000,
    min_water_budget=0.001,
    min_depth_m=0.05,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    crs="EPSG:3005",
):
    """
    Build a full coastal V2 flood footprint.

    Coastal V2 keeps two separate state objects:
        - current total footprint geometry for radial footprint construction
        - current active landward water zones for pressure and future expansion
    """

    if n_growth_stages < 1:
        raise ValueError("n_growth_stages must be at least 1.")

    coastal_v2_initial = build_initial_coastal_landward_setup(
        selected_segment=selected_segment,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        coastal_storage_gdf=coastal_storage_gdf,
        coastal_storage_side_gdf=coastal_storage_side_gdf,
        event_water_budget=event_water_budget,
        severity_max_depth_m=severity_max_depth_m,
        initial_buffer_radius_m=initial_buffer_radius_m,
        n_stage_1_reference_points=n_stage_1_reference_points,
        storage_overlap_threshold=initial_storage_overlap_threshold,
        elevation_scale_m=elevation_scale_m,
        depth_scale_factor=depth_scale_factor,
        random_seed=random_seed,
        random_low=random_low,
        random_high=random_high,
        crs=crs,
    )

    stage_outputs = {}
    stage_active_point_outputs = {}
    stage_summary_rows = []

    stage_1_depth_segments_gdf = coastal_v2_initial[
        "stage_1_landward_depth_segments_gdf"
    ].copy()
    stage_1_depth_segments_gdf["depth_stage"] = 1
    stage_1_depth_segments_gdf["depth_polygon_id"] = (
        "stage_1_"
        + stage_1_depth_segments_gdf["stage_1_segment_id"].astype(str)
    )
    stage_1_depth_segments_gdf["event_severity"] = severity
    stage_1_depth_segments_gdf["coastal_v2_zone_type"] = (
        "landward_expansion"
    )

    all_depth_polygon_gdfs = [stage_1_depth_segments_gdf]

    def _summary_value(summary, names, default=0.0):
        for name in names:
            if name in summary:
                return summary[name]
        return default

    def _build_stage_summary_row(
        stage_output,
        stage_id,
        stage_next_id,
        candidate_points_count,
        scored_candidate_points_count,
    ):
        summary = stage_output["summary"]

        return {
            "stage_id": int(stage_id),
            "stage_next_id": int(stage_next_id),
            "event_severity": severity,
            "candidate_points": int(candidate_points_count),
            "scored_candidate_points": int(scored_candidate_points_count),
            "active_points": int(summary["active_points"]),
            "expanded_points": int(summary["expanded_points"]),
            "expansion_lobes": int(summary["expansion_lobes"]),
            "depth_polygons": int(
                _summary_value(
                    summary,
                    [
                        f"stage_{stage_next_id}_depth_polygons",
                        "next_stage_depth_polygons",
                    ],
                )
            ),
            "landward_polygons": int(
                _summary_value(
                    summary,
                    [
                        f"stage_{stage_next_id}_landward_polygons",
                        "next_stage_landward_polygons",
                    ],
                )
            ),
            "storage_polygons": int(
                _summary_value(
                    summary,
                    [
                        f"stage_{stage_next_id}_storage_polygons",
                        "next_stage_storage_polygons",
                    ],
                )
            ),
            "current_area_sqkm": float(
                stage_output["current_footprint_geometry"].area / 1_000_000
            ),
            "next_area_sqkm": float(
                stage_output["next_footprint_geometry"].area / 1_000_000
            ),
            "new_area_sqkm": float(
                stage_output["new_area_geometry"].area / 1_000_000
            ),
            "current_active_water_budget": float(
                _summary_value(
                    summary,
                    [
                        "stage_1_active_water_budget",
                        "current_active_water_budget",
                    ],
                )
            ),
            "parent_outflow_water_budget": float(
                _summary_value(
                    summary,
                    [
                        "stage_1_parent_outflow_water_budget",
                        "parent_outflow_water_budget",
                    ],
                )
            ),
            "expanded_point_water_budget": float(
                summary["expanded_point_water_budget"]
            ),
            "lobe_water_budget": float(summary["lobe_water_budget"]),
            "polygon_water_budget": float(
                _summary_value(
                    summary,
                    [
                        f"stage_{stage_next_id}_polygon_water_budget",
                        "next_stage_polygon_water_budget",
                    ],
                )
            ),
            "landward_water_budget": float(
                _summary_value(
                    summary,
                    [
                        f"stage_{stage_next_id}_landward_water_budget",
                        "next_stage_landward_water_budget",
                    ],
                )
            ),
            "storage_water_budget": float(
                _summary_value(
                    summary,
                    [
                        f"stage_{stage_next_id}_storage_water_budget",
                        "next_stage_storage_water_budget",
                    ],
                )
            ),
            "updated_active_water_zones": int(
                summary["updated_active_water_zones"]
            ),
            "updated_active_water_total": float(
                summary["updated_active_water_total"]
            ),
            "new_landward_active_zones": int(
                summary["new_landward_active_zones"]
            ),
            "new_storage_zones_excluded_from_active": int(
                summary["new_storage_zones_excluded_from_active"]
            ),
            "new_storage_water_excluded_from_active": float(
                summary["new_storage_water_excluded_from_active"]
            ),
            "stage_storage_overlap_threshold": float(
                summary["stage_storage_overlap_threshold"]
            ),
            "pre_lobe_fill_depth_area_sqm": float(
                summary["pre_lobe_fill_depth_area_sqm"]
            ),
            "post_lobe_fill_depth_area_sqm": float(
                summary["post_lobe_fill_depth_area_sqm"]
            ),
            "lobe_fill_added_area_sqm": float(
                summary["lobe_fill_added_area_sqm"]
            ),
            "uncovered_new_area_after_lobe_fill_sqm": float(
                summary["uncovered_new_area_after_lobe_fill_sqm"]
            ),
            "uncovered_new_area_after_lobe_fill_share": float(
                summary["uncovered_new_area_after_lobe_fill_share"]
            ),
            "uncovered_new_area_assigned_to_polygons_sqm": float(
                summary.get(
                    "uncovered_new_area_assigned_to_polygons_sqm",
                    0.0,
                )
            ),
            "remaining_uncovered_new_area_sqm": float(
                summary.get("remaining_uncovered_new_area_sqm", 0.0)
            ),
            "remaining_uncovered_new_area_share": float(
                summary.get("remaining_uncovered_new_area_share", 0.0)
            ),
            "expanded_minus_parent_outflow_water_budget": float(
                summary["expanded_minus_parent_outflow_water_budget"]
            ),
            "lobe_minus_expanded_water_budget": float(
                summary["lobe_minus_expanded_water_budget"]
            ),
            "polygon_minus_lobe_water_budget": float(
                summary["polygon_minus_lobe_water_budget"]
            ),
        }

    stage_1_to_2_output = run_stage_1_to_2_coastal_v2_expansion(
        coastal_v2_initial=coastal_v2_initial,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        severity_max_depth_m=severity_max_depth_m,
        n_active_points=n_active_points,
        parent_penalty_strength=parent_penalty_strength,
        base_distance_m=base_distance_m,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        score_power=score_power,
        depth_power=depth_power,
        water_power=water_power,
        scan_left_degrees=scan_left_degrees,
        scan_right_degrees=scan_right_degrees,
        n_scan_angles=n_scan_angles,
        elevation_scale_m=elevation_scale_m,
        outflow_share=outflow_share,
        point_influence_radius_m=point_influence_radius_m,
        connection_buffer_m=connection_buffer_m,
        n_radial_wedges=n_radial_wedges,
        min_outward_gain_m=min_outward_gain_m,
        depth_scale_factor=depth_scale_factor,
        min_polygon_area_sqm=min_polygon_area_sqm,
        min_water_budget=min_water_budget,
        min_depth_m=min_depth_m,
        stage_storage_overlap_threshold=stage_storage_overlap_threshold,
        crs=crs,
    )

    stage_outputs[1] = stage_1_to_2_output

    stage_2_depth_polygons_gdf = stage_1_to_2_output[
        "stage_2_depth_polygons_gdf"
    ].copy()
    stage_2_depth_polygons_gdf["depth_stage"] = 2
    stage_2_depth_polygons_gdf["depth_polygon_id"] = (
        stage_2_depth_polygons_gdf["stage_2_depth_polygon_id"].astype(str)
    )
    stage_2_depth_polygons_gdf["event_severity"] = severity
    stage_2_depth_polygons_gdf["coastal_v2_zone_type"] = (
        stage_2_depth_polygons_gdf["stage_2_zone_type"]
    )
    all_depth_polygon_gdfs.append(stage_2_depth_polygons_gdf)

    stage_summary_rows.append(
        _build_stage_summary_row(
            stage_output=stage_1_to_2_output,
            stage_id=1,
            stage_next_id=2,
            candidate_points_count=stage_1_to_2_output["summary"][
                "stage_1_candidate_points"
            ],
            scored_candidate_points_count=stage_1_to_2_output["summary"][
                "stage_1_candidate_points"
            ],
        )
    )

    current_footprint_geometry = stage_1_to_2_output[
        "next_footprint_geometry"
    ]
    current_active_water_zones_gdf = stage_1_to_2_output[
        "updated_active_water_zones_gdf"
    ]

    for stage_id in range(2, n_growth_stages + 1):

        candidate_points_gdf = sample_coastal_v2_landward_candidate_points(
            current_active_water_zones_gdf=current_active_water_zones_gdf,
            coastal_context=coastal_v2_initial["coastal_context"],
            source_point=coastal_v2_initial["source_point"],
            stage_id=stage_id,
            current_total_footprint_geometry=current_footprint_geometry,
            n_candidate_points_per_zone=n_candidate_points_per_zone,
            outer_distance_share=outer_distance_share,
            boundary_sample_spacing_m=boundary_sample_spacing_m,
            boundary_match_tolerance_m=boundary_match_tolerance_m,
            crs=crs,
        )

        active_point_output = prepare_coastal_v2_stage_active_points(
            stage_candidate_points_gdf=candidate_points_gdf,
            source_point=coastal_v2_initial["source_point"],
            dem_path=dem_path,
            expansion_boundary=expansion_boundary,
            stage_id=stage_id,
            severity_max_depth_m=severity_max_depth_m,
            elevation_scale_m=elevation_scale_m,
            n_active_points=n_active_points,
            parent_penalty_strength=parent_penalty_strength,
            base_distance_m=base_distance_m,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            score_power=score_power,
            depth_power=depth_power,
            water_power=water_power,
            random_seed=random_seed + stage_id,
            random_low=random_low,
            random_high=random_high,
            crs=crs,
        )

        stage_active_point_outputs[stage_id] = active_point_output

        stage_output = run_coastal_v2_expansion_stage_from_active_points(
            coastal_v2_initial=coastal_v2_initial,
            current_footprint_geometry=current_footprint_geometry,
            current_active_water_zones_gdf=current_active_water_zones_gdf,
            active_points_with_distance_gdf=active_point_output[
                "active_points_with_distance_gdf"
            ],
            dem_path=dem_path,
            stage_id=stage_id,
            expansion_boundary=expansion_boundary,
            severity_max_depth_m=severity_max_depth_m,
            scan_left_degrees=scan_left_degrees,
            scan_right_degrees=scan_right_degrees,
            n_scan_angles=n_scan_angles,
            elevation_scale_m=elevation_scale_m,
            outflow_share=outflow_share,
            point_influence_radius_m=point_influence_radius_m,
            connection_buffer_m=connection_buffer_m,
            n_radial_wedges=n_radial_wedges,
            min_outward_gain_m=min_outward_gain_m,
            depth_scale_factor=depth_scale_factor,
            min_polygon_area_sqm=min_polygon_area_sqm,
            min_water_budget=min_water_budget,
            min_depth_m=min_depth_m,
            stage_storage_overlap_threshold=stage_storage_overlap_threshold,
            crs=crs,
        )

        stage_output["candidate_points_gdf"] = candidate_points_gdf
        stage_output["scored_candidate_points_gdf"] = active_point_output[
            "scored_candidate_points_gdf"
        ]
        stage_output["active_points_gdf"] = active_point_output[
            "active_points_gdf"
        ]

        stage_outputs[stage_id] = stage_output

        stage_next_id = stage_id + 1
        depth_key = f"stage_{stage_next_id}_depth_polygons_gdf"
        zone_type_col = f"stage_{stage_next_id}_zone_type"
        polygon_id_col = f"stage_{stage_next_id}_depth_polygon_id"

        next_stage_depth_polygons_gdf = stage_output[depth_key].copy()
        next_stage_depth_polygons_gdf["depth_stage"] = stage_next_id
        next_stage_depth_polygons_gdf["depth_polygon_id"] = (
            next_stage_depth_polygons_gdf[polygon_id_col].astype(str)
        )
        next_stage_depth_polygons_gdf["event_severity"] = severity
        next_stage_depth_polygons_gdf["coastal_v2_zone_type"] = (
            next_stage_depth_polygons_gdf[zone_type_col]
        )
        all_depth_polygon_gdfs.append(next_stage_depth_polygons_gdf)

        stage_summary_rows.append(
            _build_stage_summary_row(
                stage_output=stage_output,
                stage_id=stage_id,
                stage_next_id=stage_next_id,
                candidate_points_count=len(candidate_points_gdf),
                scored_candidate_points_count=len(
                    active_point_output["scored_candidate_points_gdf"]
                ),
            )
        )

        current_footprint_geometry = stage_output["next_footprint_geometry"]
        current_active_water_zones_gdf = stage_output[
            "updated_active_water_zones_gdf"
        ]

        if current_active_water_zones_gdf.empty:
            break

    all_depth_polygons_gdf = gpd.GeoDataFrame(
        pd.concat(
            all_depth_polygon_gdfs,
            ignore_index=True,
            sort=False,
        ),
        geometry="geometry",
        crs=crs,
    )

    stage_summary_df = pd.DataFrame(stage_summary_rows)

    final_footprint_geometry = clean_geometry(current_footprint_geometry)

    final_footprint_gdf = make_geometry_gdf(
        rows=[
            {
                "event_severity": severity,
                "n_growth_stages_requested": int(n_growth_stages),
                "n_growth_stages_completed": int(len(stage_outputs)),
                "final_area_sqm": final_footprint_geometry.area,
                "final_area_sqkm": final_footprint_geometry.area / 1_000_000,
                "geometry": final_footprint_geometry,
            }
        ],
        crs=crs,
    )

    final_summary = {
        "event_severity": severity,
        "event_water_budget": float(event_water_budget),
        "n_growth_stages_requested": int(n_growth_stages),
        "n_growth_stages_completed": int(len(stage_outputs)),
        "total_depth_polygons": int(len(all_depth_polygons_gdf)),
        "final_footprint_area_sqkm": float(
            final_footprint_geometry.area / 1_000_000
        ),
        "final_active_water_zones": int(len(current_active_water_zones_gdf)),
        "final_active_water_budget": float(
            current_active_water_zones_gdf["current_water_budget"].sum()
            if not current_active_water_zones_gdf.empty
            else 0.0
        ),
        "total_depth_polygon_water_budget": float(
            all_depth_polygons_gdf["segment_water_budget"].sum()
        ),
        "max_depth_m": float(all_depth_polygons_gdf["max_depth_m"].max()),
        "mean_depth_m": float(all_depth_polygons_gdf["max_depth_m"].mean()),
        "initial_storage_overlap_threshold": float(
            initial_storage_overlap_threshold
        ),
        "stage_storage_overlap_threshold": float(
            stage_storage_overlap_threshold
        ),
    }

    return {
        "coastal_v2_initial": coastal_v2_initial,
        "stage_outputs": stage_outputs,
        "stage_active_point_outputs": stage_active_point_outputs,
        "stage_summary_df": stage_summary_df,
        "final_summary": final_summary,
        "final_footprint_geometry": final_footprint_geometry,
        "final_footprint_gdf": final_footprint_gdf,
        "all_depth_polygons_gdf": all_depth_polygons_gdf,
        "final_active_water_zones_gdf": current_active_water_zones_gdf,
    }
