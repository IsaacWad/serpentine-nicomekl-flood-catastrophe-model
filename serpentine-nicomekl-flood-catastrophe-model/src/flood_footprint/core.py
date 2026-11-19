# ============================================================
# flood_footprint/core.py
# Shared flood footprint core mechanics
# Derived from flood_footprintV2.py
# ============================================================

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

from shapely.geometry import Point, LineString, Polygon
from shapely.validation import make_valid


# ============================================================
# Geometry helpers
# ============================================================

def clean_geometry(geometry):
    """
    Clean geometry after buffer/intersection operations.
    """

    if geometry is None:
        return geometry

    if geometry.is_empty:
        return geometry

    try:
        geometry = make_valid(geometry)
    except Exception:
        geometry = geometry.buffer(0)

    try:
        geometry = geometry.buffer(0)
    except Exception:
        pass

    return geometry


def make_geometry_gdf(rows, crs="EPSG:3005"):
    """
    Safely create a GeoDataFrame from rows containing a 'geometry' key.

    This uses WKB to avoid GeoPandas/Shapely array-interface issues.
    """

    if len(rows) == 0:
        return gpd.GeoDataFrame(
            pd.DataFrame(),
            geometry=[],
            crs=crs,
        )

    df = pd.DataFrame(rows).copy()

    geometries = list(df["geometry"])

    df = df.drop(columns=["geometry"])

    geometry_series = gpd.GeoSeries.from_wkb(
        [geom.wkb for geom in geometries],
        crs=crs,
    )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry_series,
        crs=crs,
    )

    return gdf


def extract_segment_geometry(selected_segment):
    """
    Extract geometry from selected segment.

    Accepts:
        - shapely geometry
        - one-row GeoDataFrame
        - GeoSeries row
        - pandas Series row
    """

    if hasattr(selected_segment, "geom_type"):
        return selected_segment

    if isinstance(selected_segment, gpd.GeoDataFrame):
        if selected_segment.empty:
            raise ValueError("selected_segment GeoDataFrame is empty.")
        return selected_segment.iloc[0].geometry

    if isinstance(selected_segment, (gpd.GeoSeries, pd.Series)):
        return selected_segment.geometry

    raise TypeError(
        "selected_segment must be a shapely geometry, GeoDataFrame row, "
        "GeoSeries row, pandas Series row, or one-row GeoDataFrame."
    )


def get_segment_midpoint(segment_geometry):
    """
    Get midpoint/source point from segment geometry.
    """

    if segment_geometry is None or segment_geometry.is_empty:
        raise ValueError("segment_geometry is empty.")

    if segment_geometry.geom_type in ["LineString", "LinearRing"]:
        return segment_geometry.interpolate(0.5, normalized=True)

    if segment_geometry.geom_type == "MultiLineString":
        longest_line = max(
            list(segment_geometry.geoms),
            key=lambda geom: geom.length,
        )
        return longest_line.interpolate(0.5, normalized=True)

    if segment_geometry.geom_type == "Point":
        return segment_geometry

    return segment_geometry.representative_point()



# ============================================================
# Stage 0 / initial flood setup
# ============================================================

def create_initial_flood_setup(
    selected_segment,
    buffer_radius_m,
    expansion_boundary,
    n_reference_points=16,
    crs="EPSG:3005",
):
    """
    Create the initial flood setup.

    This creates:
        - source midpoint
        - initial base buffer
        - reference points around the buffer
        - radial reference lines from source to reference points

    This is the setup phase before Stage 1 depth zones.
    """

    # ------------------------------------------------------------
    # 1. Segment geometry and source point
    # ------------------------------------------------------------

    segment_geometry = extract_segment_geometry(selected_segment)

    if segment_geometry is None or segment_geometry.is_empty:
        raise ValueError("selected_segment geometry is empty.")

    source_point = get_segment_midpoint(segment_geometry)

    # ------------------------------------------------------------
    # 2. Initial base buffer clipped to expansion boundary
    # ------------------------------------------------------------

    base_buffer_geometry = source_point.buffer(buffer_radius_m)
    base_buffer_geometry = base_buffer_geometry.intersection(expansion_boundary)
    base_buffer_geometry = clean_geometry(base_buffer_geometry)

    if base_buffer_geometry is None or base_buffer_geometry.is_empty:
        raise ValueError(
            "base_buffer_geometry is empty after clipping to expansion_boundary."
        )

    # ------------------------------------------------------------
    # 3. Base buffer GeoDataFrame
    # ------------------------------------------------------------

    base_buffer_rows = [
        {
            "setup_object": "base_buffer",
            "buffer_radius_m": buffer_radius_m,
            "source_x": source_point.x,
            "source_y": source_point.y,
            "base_area_sqm": base_buffer_geometry.area,
            "base_area_sqkm": base_buffer_geometry.area / 1_000_000,
            "geometry": base_buffer_geometry,
        }
    ]

    base_buffer_gdf = make_geometry_gdf(
        rows=base_buffer_rows,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 4. Reference points and radial lines
    # ------------------------------------------------------------

    reference_point_rows = []
    reference_line_rows = []

    reference_angles = np.linspace(
        0,
        360,
        n_reference_points,
        endpoint=False,
    )

    for reference_id, angle_degrees in enumerate(reference_angles):

        angle_radians = np.deg2rad(angle_degrees)

        raw_reference_point = Point(
            source_point.x + buffer_radius_m * np.cos(angle_radians),
            source_point.y + buffer_radius_m * np.sin(angle_radians),
        )

        if (
            base_buffer_geometry.contains(raw_reference_point)
            or base_buffer_geometry.touches(raw_reference_point)
        ):
            reference_point = raw_reference_point

        else:
            reference_point = base_buffer_geometry.boundary.interpolate(
                base_buffer_geometry.boundary.project(raw_reference_point)
            )

        reference_point_rows.append(
            {
                "reference_id": reference_id,
                "angle_degrees": angle_degrees,
                "distance_from_source_m": source_point.distance(reference_point),
                "geometry": reference_point,
            }
        )

        reference_line_rows.append(
            {
                "reference_id": reference_id,
                "angle_degrees": angle_degrees,
                "line_length_m": source_point.distance(reference_point),
                "geometry": LineString(
            [
                (source_point.x, source_point.y),
                (reference_point.x, reference_point.y),
            ]),
            }
        )

    reference_points_gdf = make_geometry_gdf(
        rows=reference_point_rows,
        crs=crs,
    )

    reference_lines_gdf = make_geometry_gdf(
        rows=reference_line_rows,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 5. Return setup package
    # ------------------------------------------------------------

    return {
        "source_point": source_point,
        "segment_geometry": segment_geometry,
        "base_buffer_geometry": base_buffer_geometry,
        "base_buffer_gdf": base_buffer_gdf,
        "reference_points_gdf": reference_points_gdf,
        "reference_lines_gdf": reference_lines_gdf,
    }


# ============================================================
# DEM helpers
# ============================================================


def sample_dem_elevation_at_point(point, dem_path):
    """
    Sample DEM elevation at a single point.
    """

    with rasterio.open(dem_path) as src:
        value = list(src.sample([(point.x, point.y)]))[0][0]
        nodata = src.nodata

    if nodata is not None and value == nodata:
        return np.nan

    return float(value)


# ============================================================
# Stage 1 scored points
# ============================================================

def create_stage_1_scored_points(
    reference_points_gdf,
    source_point,
    dem_path,
    expansion_boundary,
    event_water_budget,
    elevation_scale_m=1.0,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    min_score=0.01,
    crs="EPSG:3005",
):
    """
    Create and score Stage 1 points.

    Stage 1 is special because these points come from the initial
    setup reference points, not from a previous flood footprint boundary.
    """

    if reference_points_gdf.empty:
        raise ValueError("reference_points_gdf is empty.")

    required_cols = [
        "reference_id",
        "angle_degrees",
        "distance_from_source_m",
        "geometry",
    ]

    for col in required_cols:
        if col not in reference_points_gdf.columns:
            raise ValueError(f"reference_points_gdf is missing column: {col}")

    source_elevation_m = sample_dem_elevation_at_point(
        point=source_point,
        dem_path=dem_path,
    )

    rng = np.random.default_rng(random_seed)

    stage_1_rows = []

    for _, row in reference_points_gdf.iterrows():

        stage_1_point = row.geometry

        inside_expansion_boundary = (
            expansion_boundary.contains(stage_1_point)
            or expansion_boundary.touches(stage_1_point)
        )

        stage_1_point_elevation_m = sample_dem_elevation_at_point(
            point=stage_1_point,
            dem_path=dem_path,
        )

        elevation_difference_m = (
            stage_1_point_elevation_m - source_elevation_m
        )

        if (
            np.isnan(source_elevation_m)
            or np.isnan(stage_1_point_elevation_m)
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
                    abs(elevation_difference_m) / max(elevation_scale_m, 1e-6),
                    1.0,
                )

                elevation_score = elevation_score * (1 + 0.25 * downhill_bonus)

        random_factor = rng.uniform(random_low, random_high)

        stage_1_score = max(
            elevation_score * random_factor,
            min_score,
        )

        stage_1_rows.append(
            {
                "stage_id": 1,
                "stage_1_point_id": f"stage_1_point_{int(row['reference_id'])}",
                "reference_id": int(row["reference_id"]),
                "angle_degrees": row["angle_degrees"],
                "distance_from_source_m": row["distance_from_source_m"],
                "source_elevation_m": source_elevation_m,
                "stage_1_point_elevation_m": stage_1_point_elevation_m,
                "elevation_difference_from_source_m": elevation_difference_m,
                "inside_expansion_boundary": inside_expansion_boundary,
                "stage_1_score": stage_1_score,
                "geometry": stage_1_point,
            }
        )

    stage_1_points_gdf = make_geometry_gdf(
        rows=stage_1_rows,
        crs=crs,
    )

    total_score = stage_1_points_gdf["stage_1_score"].sum()

    if total_score <= 0 or np.isnan(total_score):
        stage_1_points_gdf["stage_1_score_share"] = (
            1 / len(stage_1_points_gdf)
        )
    else:
        stage_1_points_gdf["stage_1_score_share"] = (
            stage_1_points_gdf["stage_1_score"] / total_score
        )

    stage_1_points_gdf["stage_1_allocated_water_budget"] = (
        stage_1_points_gdf["stage_1_score_share"] * event_water_budget
    )

    return stage_1_points_gdf


# ============================================================
# Stage 1 depth segments
# ============================================================

def create_stage_1_depth_segments(
    source_point,
    base_buffer_geometry,
    stage_1_points_gdf,
    severity_max_depth_m,
    depth_scale_factor=50_000,
    crs="EPSG:3005",
):
    """
    Create Stage 1 depth/pressure segments.

    Each Stage 1 point owns a radial wedge inside the base buffer.
    Water allocated to the point becomes water allocated to that wedge.
    """

    if stage_1_points_gdf.empty:
        raise ValueError("stage_1_points_gdf is empty.")

    required_cols = [
        "stage_1_point_id",
        "angle_degrees",
        "stage_1_allocated_water_budget",
        "geometry",
    ]

    for col in required_cols:
        if col not in stage_1_points_gdf.columns:
            raise ValueError(f"stage_1_points_gdf is missing column: {col}")

    if base_buffer_geometry is None or base_buffer_geometry.is_empty:
        raise ValueError("base_buffer_geometry is empty.")

    stage_1_points_sorted = (
        stage_1_points_gdf
        .sort_values("angle_degrees")
        .reset_index(drop=True)
        .copy()
    )

    n_points = len(stage_1_points_sorted)

    # Use the max reference distance so the wedge fully covers clipped/base area.
    max_radius_m = stage_1_points_sorted["distance_from_source_m"].max() * 2

    segment_rows = []

    for i, row in stage_1_points_sorted.iterrows():

        current_angle = row["angle_degrees"]

        previous_angle = stage_1_points_sorted.iloc[i - 1]["angle_degrees"]
        next_angle = stage_1_points_sorted.iloc[(i + 1) % n_points]["angle_degrees"]

        # Handle wraparound.
        if i == 0:
            previous_angle = previous_angle - 360

        if i == n_points - 1:
            next_angle = next_angle + 360

        left_boundary_angle = (previous_angle + current_angle) / 2
        right_boundary_angle = (current_angle + next_angle) / 2

        left_rad = np.deg2rad(left_boundary_angle)
        right_rad = np.deg2rad(right_boundary_angle)

        left_point = Point(
            source_point.x + max_radius_m * np.cos(left_rad),
            source_point.y + max_radius_m * np.sin(left_rad),
        )

        current_outer_point = Point(
            source_point.x + max_radius_m * np.cos(np.deg2rad(current_angle)),
            source_point.y + max_radius_m * np.sin(np.deg2rad(current_angle)),
        )

        right_point = Point(
            source_point.x + max_radius_m * np.cos(right_rad),
            source_point.y + max_radius_m * np.sin(right_rad),
        )

        wedge_geometry = Polygon(
            [
                (source_point.x, source_point.y),
                (left_point.x, left_point.y),
                (current_outer_point.x, current_outer_point.y),
                (right_point.x, right_point.y),
                (source_point.x, source_point.y),
            ]
        )

        stage_1_segment_geometry = wedge_geometry.intersection(
            base_buffer_geometry
        )

        stage_1_segment_geometry = clean_geometry(stage_1_segment_geometry)

        if stage_1_segment_geometry is None or stage_1_segment_geometry.is_empty:
            continue

        segment_water_budget = row["stage_1_allocated_water_budget"]
        segment_area_sqm = stage_1_segment_geometry.area

        raw_depth_proxy = segment_water_budget / max(segment_area_sqm, 1e-6)

        max_depth_m = min(
            raw_depth_proxy * depth_scale_factor,
            severity_max_depth_m,
        )

        segment_rows.append(
            {
                "stage_id": 1,
                "stage_1_segment_id": f"stage_1_segment_{i}",
                "stage_1_point_id": row["stage_1_point_id"],
                "reference_id": row["reference_id"],
                "angle_degrees": current_angle,
                "left_boundary_angle": left_boundary_angle % 360,
                "right_boundary_angle": right_boundary_angle % 360,
                "stage_1_score": row["stage_1_score"],
                "stage_1_score_share": row["stage_1_score_share"],
                "segment_water_budget": segment_water_budget,
                "segment_area_sqm": segment_area_sqm,
                "raw_depth_proxy": raw_depth_proxy,
                "max_depth_m": max_depth_m,
                "geometry": stage_1_segment_geometry,
            }
        )

    stage_1_depth_segments_gdf = make_geometry_gdf(
        rows=segment_rows,
        crs=crs,
    )

    return stage_1_depth_segments_gdf


# ============================================================
# Current water zone initialization
# ============================================================

def initialize_current_water_zones_from_stage_1(
    stage_1_depth_segments_gdf,
    min_water_budget=0.001,
    min_depth_m=0.05,
    crs="EPSG:3005",
):
    """
    Convert Stage 1 depth segments into current water/pressure zones.

    These zones become the pressure source for Stage 2 expansion.
    """

    if stage_1_depth_segments_gdf.empty:
        raise ValueError("stage_1_depth_segments_gdf is empty.")

    required_cols = [
        "stage_1_segment_id",
        "segment_water_budget",
        "max_depth_m",
        "geometry",
    ]

    for col in required_cols:
        if col not in stage_1_depth_segments_gdf.columns:
            raise ValueError(f"stage_1_depth_segments_gdf is missing column: {col}")

    current_rows = []

    for _, row in stage_1_depth_segments_gdf.iterrows():

        current_water_budget = row["segment_water_budget"]
        current_max_depth_m = row["max_depth_m"]

        if current_water_budget < min_water_budget:
            continue

        if current_max_depth_m < min_depth_m:
            continue

        current_rows.append(
            {
                "current_zone_id": row["stage_1_segment_id"],
                "current_stage_origin": "stage_1",
                "current_water_budget": current_water_budget,
                "current_max_depth_m": current_max_depth_m,
                "geometry": row.geometry,
            }
        )

    current_water_zones_gdf = make_geometry_gdf(
        rows=current_rows,
        crs=crs,
    )

    return current_water_zones_gdf




###----------###
###--Part 2--###
###----------###

# Now this is where we create the robust repeatable process to expand the distance of our flood and the transfer of the depth


#HELPER FUNCTIONS

# ============================================================
# Stage expansion helper 1:
# Sample candidate points from current footprint boundary
# ============================================================

def sample_candidate_points_from_current_footprint(
    current_footprint_geometry,
    source_point,
    stage_id,
    n_candidate_points=128,
    crs="EPSG:3005",
):
    """
    Sample candidate points along the current flood footprint boundary.

    These candidate points represent possible outward expansion locations
    for the current stage.

    Example:
        stage_id = 1 means these are Stage 1 candidate points used to
        expand from the Stage 1 footprint into the Stage 2 footprint.

    Parameters
    ----------
    current_footprint_geometry:
        Current flood footprint geometry.

    source_point:
        Original source midpoint of the selected segment.

    stage_id:
        Current stage id.

    n_candidate_points:
        Number of candidate points to sample around the current footprint boundary.

    crs:
        Projected CRS used for distance calculations.

    Returns
    -------
    candidate_points_gdf:
        GeoDataFrame of candidate boundary points.
    """

    if current_footprint_geometry is None or current_footprint_geometry.is_empty:
        raise ValueError("current_footprint_geometry is empty.")

    if source_point is None or source_point.is_empty:
        raise ValueError("source_point is empty.")

    boundary_geometry = current_footprint_geometry.boundary

    if boundary_geometry is None or boundary_geometry.is_empty:
        raise ValueError("current_footprint_geometry boundary is empty.")

    candidate_rows = []

    sample_distances = np.linspace(
        0,
        boundary_geometry.length,
        n_candidate_points,
        endpoint=False,
    )

    for candidate_id, boundary_distance in enumerate(sample_distances):

        candidate_point = boundary_geometry.interpolate(boundary_distance)

        dx = candidate_point.x - source_point.x
        dy = candidate_point.y - source_point.y

        angle_degrees = np.degrees(
            np.arctan2(dy, dx)
        ) % 360

        distance_from_source_m = source_point.distance(candidate_point)

        candidate_rows.append(
            {
                "stage_id": stage_id,
                "candidate_id": candidate_id,
                "boundary_distance_m": boundary_distance,
                "boundary_position_share": boundary_distance / boundary_geometry.length,
                "boundary_length_m": boundary_geometry.length,
                "angle_degrees": angle_degrees,
                "distance_from_source_m": distance_from_source_m,
                "geometry": candidate_point,
            }
        )

    candidate_points_gdf = make_geometry_gdf(
        rows=candidate_rows,
        crs=crs,
    )

    return candidate_points_gdf

# ============================================================
# Stage expansion helper 2:
# Attach nearest current pressure zone to candidate points
# ============================================================

def attach_nearest_pressure_zone_to_candidate_points(
    candidate_points_gdf,
    current_water_zones_gdf,
    stage_id,
    candidate_id_col="candidate_id",
    current_zone_id_col="current_zone_id",
    current_stage_origin_col="current_stage_origin",
    current_water_col="current_water_budget",
    current_depth_col="current_max_depth_m",
    crs="EPSG:3005",
):
    """
    Attach the nearest current water/pressure zone to each candidate point.

    Candidate points are sampled from the current footprint boundary.
    Current water zones provide the pressure surface.

    This function does not score points yet. It only tells each candidate
    which pressure zone is closest to it.
    """

    if candidate_points_gdf.empty:
        raise ValueError("candidate_points_gdf is empty.")

    if current_water_zones_gdf.empty:
        raise ValueError("current_water_zones_gdf is empty.")

    candidate_required_cols = [
        candidate_id_col,
        "stage_id",
        "angle_degrees",
        "distance_from_source_m",
        "geometry",
    ]

    for col in candidate_required_cols:
        if col not in candidate_points_gdf.columns:
            raise ValueError(f"candidate_points_gdf is missing column: {col}")

    pressure_required_cols = [
        current_zone_id_col,
        current_stage_origin_col,
        current_water_col,
        current_depth_col,
        "geometry",
    ]

    for col in pressure_required_cols:
        if col not in current_water_zones_gdf.columns:
            raise ValueError(f"current_water_zones_gdf is missing column: {col}")

    candidate_points = candidate_points_gdf.copy()
    pressure_zones = current_water_zones_gdf[
        [
            current_zone_id_col,
            current_stage_origin_col,
            current_water_col,
            current_depth_col,
            "geometry",
        ]
    ].copy()

    candidate_points = gpd.GeoDataFrame(
        candidate_points,
        geometry="geometry",
        crs=crs,
    )

    pressure_zones = gpd.GeoDataFrame(
        pressure_zones,
        geometry="geometry",
        crs=crs,
    )

    # ------------------------------------------------------------
    # Preferred method: GeoPandas nearest spatial join
    # ------------------------------------------------------------

    try:
        joined_gdf = gpd.sjoin_nearest(
            candidate_points,
            pressure_zones,
            how="left",
            distance_col="distance_to_pressure_zone_m",
        )

        # If multiple zones tie, keep the nearest/first per candidate.
        joined_gdf = (
            joined_gdf
            .sort_values(
                [candidate_id_col, "distance_to_pressure_zone_m"]
            )
            .drop_duplicates(
                subset=[candidate_id_col],
                keep="first",
            )
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------
    # Fallback method: manual nearest distance
    # ------------------------------------------------------------

    except Exception:

        joined_rows = []

        for _, candidate_row in candidate_points.iterrows():

            distances = pressure_zones.geometry.distance(
                candidate_row.geometry
            )

            nearest_idx = distances.idxmin()
            nearest_zone = pressure_zones.loc[nearest_idx]

            out_row = candidate_row.to_dict()

            out_row[current_zone_id_col] = nearest_zone[current_zone_id_col]
            out_row[current_stage_origin_col] = nearest_zone[current_stage_origin_col]
            out_row[current_water_col] = nearest_zone[current_water_col]
            out_row[current_depth_col] = nearest_zone[current_depth_col]
            out_row["distance_to_pressure_zone_m"] = float(
                distances.loc[nearest_idx]
            )

            joined_rows.append(out_row)

        joined_gdf = make_geometry_gdf(
            rows=joined_rows,
            crs=crs,
        )

    # ------------------------------------------------------------
    # Standard parent naming
    # ------------------------------------------------------------

    joined_gdf = joined_gdf.rename(
        columns={
            current_zone_id_col: "parent_zone_id",
            current_stage_origin_col: "parent_stage_origin",
            current_water_col: "parent_water_budget",
            current_depth_col: "parent_max_depth_m",
        }
    )

    # Drop spatial join index column if it exists.
    joined_gdf = joined_gdf.drop(
        columns=["index_right"],
        errors="ignore",
    )

    # Add clean stage-specific helper id
    joined_gdf[f"stage_{stage_id}_candidate_id"] = (
        "stage_"
        + str(stage_id)
        + "_candidate_"
        + joined_gdf[candidate_id_col].astype(str)
    )

    return gpd.GeoDataFrame(
        joined_gdf,
        geometry="geometry",
        crs=crs,
    )


# ============================================================
# Stage expansion helper 3:
# Score stage candidate points
# ============================================================

def score_stage_candidate_points(
    candidate_points_with_pressure_gdf,
    source_point,
    dem_path,
    expansion_boundary,
    stage_id,
    severity_max_depth_m,
    elevation_scale_m=1.0,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    min_score=0.01,
    crs="EPSG:3005",
):
    """
    Score candidate points for the current expansion stage.

    Candidate score combines:
        - DEM favorability
        - parent pressure depth
        - parent water budget
        - small random factor

    The output is still a candidate pool. We have not selected active
    expansion points yet.
    """

    if candidate_points_with_pressure_gdf.empty:
        raise ValueError("candidate_points_with_pressure_gdf is empty.")

    required_cols = [
        "candidate_id",
        "stage_id",
        "angle_degrees",
        "distance_from_source_m",
        "parent_zone_id",
        "parent_stage_origin",
        "parent_water_budget",
        "parent_max_depth_m",
        "geometry",
    ]

    for col in required_cols:
        if col not in candidate_points_with_pressure_gdf.columns:
            raise ValueError(
                f"candidate_points_with_pressure_gdf is missing column: {col}"
            )

    source_elevation_m = sample_dem_elevation_at_point(
        point=source_point,
        dem_path=dem_path,
    )

    rng = np.random.default_rng(random_seed)

    scored_rows = []

    mean_parent_water_budget = max(
        candidate_points_with_pressure_gdf["parent_water_budget"].mean(),
        1e-9,
    )

    for _, row in candidate_points_with_pressure_gdf.iterrows():

        candidate_point = row.geometry

        inside_expansion_boundary = (
            expansion_boundary.contains(candidate_point)
            or expansion_boundary.touches(candidate_point)
        )

        candidate_elevation_m = sample_dem_elevation_at_point(
            point=candidate_point,
            dem_path=dem_path,
        )

        elevation_difference_from_source_m = (
            candidate_elevation_m - source_elevation_m
        )

        # --------------------------------------------------------
        # DEM score
        # --------------------------------------------------------

        if (
            np.isnan(source_elevation_m)
            or np.isnan(candidate_elevation_m)
            or not inside_expansion_boundary
        ):
            dem_score = min_score

        else:
            uphill_m = max(elevation_difference_from_source_m, 0)

            # Penalize uphill candidate locations.
            dem_score = np.exp(
                -uphill_m / max(elevation_scale_m, 1e-6)
            )

            # Modest capped bonus for downhill/flat movement.
            if elevation_difference_from_source_m < 0:
                downhill_bonus = min(
                    abs(elevation_difference_from_source_m)
                    / max(elevation_scale_m, 1e-6),
                    1.0,
                )

                dem_score = dem_score * (1 + 0.25 * downhill_bonus)

        # --------------------------------------------------------
        # Parent pressure multipliers
        # --------------------------------------------------------

        parent_depth_pressure_factor = (
            row["parent_max_depth_m"] / max(severity_max_depth_m, 1e-9)
        )

        parent_depth_pressure_factor = max(
            parent_depth_pressure_factor,
            0.0,
        )

        parent_depth_multiplier = np.sqrt(parent_depth_pressure_factor)

        parent_water_factor = (
            row["parent_water_budget"] / mean_parent_water_budget
        )

        parent_water_factor = max(parent_water_factor, 0.0)

        parent_water_multiplier = np.sqrt(parent_water_factor)

        # --------------------------------------------------------
        # Final candidate score
        # --------------------------------------------------------

        random_factor = rng.uniform(random_low, random_high)

        candidate_score = (
            dem_score
            * parent_depth_multiplier
            * parent_water_multiplier
            * random_factor
        )

        candidate_score = max(candidate_score, min_score)

        out_row = row.to_dict()

        out_row.update(
            {
                f"stage_{stage_id}_candidate_elevation_m": candidate_elevation_m,
                f"stage_{stage_id}_elevation_difference_from_source_m": elevation_difference_from_source_m,
                f"stage_{stage_id}_dem_score": dem_score,
                f"stage_{stage_id}_parent_depth_pressure_factor": parent_depth_pressure_factor,
                f"stage_{stage_id}_parent_depth_multiplier": parent_depth_multiplier,
                f"stage_{stage_id}_parent_water_factor": parent_water_factor,
                f"stage_{stage_id}_parent_water_multiplier": parent_water_multiplier,
                f"stage_{stage_id}_random_factor": random_factor,
                f"stage_{stage_id}_candidate_score": candidate_score,
                "inside_expansion_boundary": inside_expansion_boundary,
                "source_elevation_m": source_elevation_m,
            }
        )

        scored_rows.append(out_row)

    scored_candidates_gdf = make_geometry_gdf(
        rows=scored_rows,
        crs=crs,
    )

    score_col = f"stage_{stage_id}_candidate_score"
    score_share_col = f"stage_{stage_id}_candidate_score_share"

    total_score = scored_candidates_gdf[score_col].sum()

    if total_score <= 0 or np.isnan(total_score):
        scored_candidates_gdf[score_share_col] = (
            1 / len(scored_candidates_gdf)
        )
    else:
        scored_candidates_gdf[score_share_col] = (
            scored_candidates_gdf[score_col] / total_score
        )

    return scored_candidates_gdf



# ============================================================
# Stage expansion helper 4:
# Select active candidate points with parent-zone penalty
# ============================================================

def select_active_candidate_points(
    scored_candidate_points_gdf,
    stage_id,
    n_active_points=16,
    parent_penalty_strength=0.35,
    min_adjusted_score=0.0,
    score_col=None,
    parent_zone_col="parent_zone_id",
    crs="EPSG:3005",
):
    """
    Select active expansion points from scored candidate points.

    Applies a same-parent-zone penalty so one pressure zone does not
    dominate all active expansion outlets.

    adjusted_score = raw_score / (1 + parent_penalty_strength * n_prior_selected_from_parent)
    """

    if scored_candidate_points_gdf.empty:
        raise ValueError("scored_candidate_points_gdf is empty.")

    if score_col is None:
        score_col = f"stage_{stage_id}_candidate_score"

    required_cols = [
        "candidate_id",
        score_col,
        parent_zone_col,
        "geometry",
    ]

    for col in required_cols:
        if col not in scored_candidate_points_gdf.columns:
            raise ValueError(f"scored_candidate_points_gdf is missing column: {col}")

    remaining_gdf = scored_candidate_points_gdf.copy()
    remaining_gdf[parent_zone_col] = remaining_gdf[parent_zone_col].astype(str)

    selected_rows = []
    selected_parent_counts = {}

    for selection_rank in range(1, n_active_points + 1):

        if remaining_gdf.empty:
            break

        adjusted_scores = []

        for _, row in remaining_gdf.iterrows():

            parent_zone_id = row[parent_zone_col]

            n_prior_selected_from_parent = selected_parent_counts.get(
                parent_zone_id,
                0,
            )

            adjusted_score = row[score_col] / (
                1 + parent_penalty_strength * n_prior_selected_from_parent
            )

            adjusted_scores.append(adjusted_score)

        remaining_gdf["_adjusted_selection_score"] = adjusted_scores

        best_idx = remaining_gdf["_adjusted_selection_score"].idxmax()
        best_row = remaining_gdf.loc[best_idx].copy()

        if best_row["_adjusted_selection_score"] <= min_adjusted_score:
            break

        parent_zone_id = best_row[parent_zone_col]

        n_prior_selected_from_parent = selected_parent_counts.get(
            parent_zone_id,
            0,
        )

        # --------------------------------------------------------
        # Important: preserve geometry explicitly
        # --------------------------------------------------------

        best_geometry = best_row.geometry

        best_row_dict = best_row.drop(
            labels=["_adjusted_selection_score"],
            errors="ignore",
        ).to_dict()

        best_row_dict["geometry"] = best_geometry
        best_row_dict["selection_rank"] = selection_rank
        best_row_dict["same_parent_prior_selected"] = n_prior_selected_from_parent
        best_row_dict["adjusted_selection_score"] = best_row["_adjusted_selection_score"]

        selected_rows.append(best_row_dict)

        selected_parent_counts[parent_zone_id] = n_prior_selected_from_parent + 1

        remaining_gdf = remaining_gdf.drop(index=best_idx)

    active_points_gdf = make_geometry_gdf(
        rows=selected_rows,
        crs=crs,
    )

    active_score_share_col = f"active_stage_{stage_id}_score_share"

    total_active_score = active_points_gdf[score_col].sum()

    if total_active_score <= 0 or np.isnan(total_active_score):
        active_points_gdf[active_score_share_col] = (
            1 / len(active_points_gdf)
        )
    else:
        active_points_gdf[active_score_share_col] = (
            active_points_gdf[score_col] / total_active_score
        )

    return active_points_gdf


# ============================================================
# Stage expansion helper 5:
# Assign expansion distances to active points
# ============================================================

def assign_stage_expansion_distances(
    active_points_gdf,
    stage_id,
    severity_max_depth_m,
    base_distance_m,
    min_distance_m,
    max_distance_m,
    active_score_share_col=None,
    parent_water_col="parent_water_budget",
    parent_depth_col="parent_max_depth_m",
    score_power=0.50,
    depth_power=0.75,
    water_power=0.35,
    crs="EPSG:3005",
):
    """
    Assign an expansion distance to each active point.

    Expansion distance is larger when:
        - active point has a larger score share
        - parent pressure zone is deeper
        - parent pressure zone has more water

    This function does not move the points yet.
    It only decides how far each point will try to expand.
    """

    if active_points_gdf.empty:
        raise ValueError("active_points_gdf is empty.")

    if active_score_share_col is None:
        active_score_share_col = f"active_stage_{stage_id}_score_share"

    required_cols = [
        active_score_share_col,
        parent_water_col,
        parent_depth_col,
        "geometry",
    ]

    for col in required_cols:
        if col not in active_points_gdf.columns:
            raise ValueError(f"active_points_gdf is missing column: {col}")

    out_gdf = active_points_gdf.copy()

    n_active_points = len(out_gdf)
    equal_score_share = 1 / max(n_active_points, 1)

    # ------------------------------------------------------------
    # Score multiplier
    # ------------------------------------------------------------

    out_gdf[f"stage_{stage_id}_distance_score_multiplier"] = (
        out_gdf[active_score_share_col].clip(lower=0) / equal_score_share
    ) ** score_power

    # ------------------------------------------------------------
    # Depth multiplier
    # ------------------------------------------------------------

    out_gdf[f"stage_{stage_id}_distance_depth_pressure_factor"] = (
        out_gdf[parent_depth_col] / max(severity_max_depth_m, 1e-9)
    ).clip(lower=0)

    out_gdf[f"stage_{stage_id}_distance_depth_multiplier"] = (
        out_gdf[f"stage_{stage_id}_distance_depth_pressure_factor"]
    ) ** depth_power

    # ------------------------------------------------------------
    # Water multiplier
    # ------------------------------------------------------------

    mean_parent_water = max(
        out_gdf[parent_water_col].mean(),
        1e-9,
    )

    out_gdf[f"stage_{stage_id}_distance_water_factor"] = (
        out_gdf[parent_water_col].clip(lower=0) / mean_parent_water
    )

    out_gdf[f"stage_{stage_id}_distance_water_multiplier"] = (
        out_gdf[f"stage_{stage_id}_distance_water_factor"]
    ) ** water_power

    # ------------------------------------------------------------
    # Final expansion distance
    # ------------------------------------------------------------

    distance_col = f"stage_{stage_id}_expansion_distance_m"

    out_gdf[distance_col] = (
        base_distance_m
        * out_gdf[f"stage_{stage_id}_distance_score_multiplier"]
        * out_gdf[f"stage_{stage_id}_distance_depth_multiplier"]
        * out_gdf[f"stage_{stage_id}_distance_water_multiplier"]
    )

    out_gdf[distance_col] = out_gdf[distance_col].clip(
        lower=min_distance_m,
        upper=max_distance_m,
    )

    return gpd.GeoDataFrame(
        out_gdf,
        geometry="geometry",
        crs=crs,
    )

# ============================================================
# Stage expansion helper 6:
# Expand active points using DEM scan
# ============================================================

def expand_active_points_with_dem_scan(
    active_points_with_distance_gdf,
    dem_path,
    expansion_boundary,
    stage_id,
    scan_left_degrees=11.25,
    scan_right_degrees=11.25,
    n_scan_angles=9,
    elevation_scale_m=1.0,
    min_score=0.01,
    crs="EPSG:3005",
):
    """
    Expand active points outward using a DEM-guided angular scan.

    For each active point:
        - take its outward angle
        - scan several nearby angles
        - create possible destination points at the assigned distance
        - score each destination by DEM favourability
        - keep the best destination

    This function creates expanded points but does not allocate water yet.
    """

    if active_points_with_distance_gdf.empty:
        raise ValueError("active_points_with_distance_gdf is empty.")

    distance_col = f"stage_{stage_id}_expansion_distance_m"
    active_score_share_col = f"active_stage_{stage_id}_score_share"

    required_cols = [
    "candidate_id",
    f"stage_{stage_id}_candidate_id",
    "angle_degrees",
    distance_col,
    active_score_share_col,
    "parent_zone_id",
    "parent_water_budget",
    "parent_max_depth_m",
    "geometry",
    "boundary_distance_m",
    "boundary_position_share",
    "boundary_length_m",
    ]

    for col in required_cols:
        if col not in active_points_with_distance_gdf.columns:
            raise ValueError(f"active_points_with_distance_gdf is missing column: {col}")

    expanded_point_rows = []
    scan_line_rows = []

    scan_offsets = np.linspace(
        -scan_left_degrees,
        scan_right_degrees,
        n_scan_angles,
    )

    for _, row in active_points_with_distance_gdf.iterrows():

        active_point = row.geometry
        active_elevation_m = sample_dem_elevation_at_point(
            point=active_point,
            dem_path=dem_path,
        )

        base_angle_degrees = row["angle_degrees"]
        expansion_distance_m = row[distance_col]

        best_scan = None
        all_scan_results = []

        for scan_offset_degrees in scan_offsets:

            scan_angle_degrees = (
                base_angle_degrees + scan_offset_degrees
            ) % 360

            scan_angle_radians = np.deg2rad(scan_angle_degrees)

            candidate_expanded_point = Point(
                active_point.x
                + expansion_distance_m * np.cos(scan_angle_radians),
                active_point.y
                + expansion_distance_m * np.sin(scan_angle_radians),
            )

            inside_expansion_boundary = (
                expansion_boundary.contains(candidate_expanded_point)
                or expansion_boundary.touches(candidate_expanded_point)
            )

            if inside_expansion_boundary:
                candidate_elevation_m = sample_dem_elevation_at_point(
                    point=candidate_expanded_point,
                    dem_path=dem_path,
                )
            else:
                candidate_elevation_m = np.nan

            elevation_difference_m = candidate_elevation_m - active_elevation_m

            # ----------------------------------------------------
            # DEM scan score
            # ----------------------------------------------------

            if (
                np.isnan(active_elevation_m)
                or np.isnan(candidate_elevation_m)
                or not inside_expansion_boundary
            ):
                scan_score = min_score

            else:
                uphill_m = max(elevation_difference_m, 0)

                scan_score = np.exp(
                    -uphill_m / max(elevation_scale_m, 1e-6)
                )

                # Modest bonus for downhill paths.
                if elevation_difference_m < 0:
                    downhill_bonus = min(
                        abs(elevation_difference_m)
                        / max(elevation_scale_m, 1e-6),
                        1.0,
                    )

                    scan_score = scan_score * (1 + 0.25 * downhill_bonus)

                scan_score = max(scan_score, min_score)

            scan_result = {
                "scan_offset_degrees": scan_offset_degrees,
                "scan_angle_degrees": scan_angle_degrees,
                "candidate_expanded_point": candidate_expanded_point,
                "candidate_elevation_m": candidate_elevation_m,
                "elevation_difference_m": elevation_difference_m,
                "inside_expansion_boundary": inside_expansion_boundary,
                "scan_score": scan_score,
            }

            all_scan_results.append(scan_result)

            if best_scan is None or scan_score > best_scan["scan_score"]:
                best_scan = scan_result

        expanded_point = best_scan["candidate_expanded_point"]

        expanded_point_rows.append(
        {
        "stage_id": stage_id,
        f"stage_{stage_id}_expanded_point_id": (
            f"stage_{stage_id}_expanded_point_{int(row['candidate_id'])}"
        ),
        f"stage_{stage_id}_candidate_id": row[f"stage_{stage_id}_candidate_id"],
        "candidate_id": row["candidate_id"],
        "selection_rank": row["selection_rank"],

        # carry active score share forward
        active_score_share_col: row[active_score_share_col],

        "parent_zone_id": row["parent_zone_id"],
        "parent_stage_origin": row["parent_stage_origin"],
        "parent_water_budget": row["parent_water_budget"],
        "parent_max_depth_m": row["parent_max_depth_m"],
        "base_angle_degrees": base_angle_degrees,
        "selected_scan_offset_degrees": best_scan["scan_offset_degrees"],
        "selected_scan_angle_degrees": best_scan["scan_angle_degrees"],
        "expansion_distance_m": expansion_distance_m,
        "active_point_elevation_m": active_elevation_m,
        "expanded_point_elevation_m": best_scan["candidate_elevation_m"],
        "elevation_difference_m": best_scan["elevation_difference_m"],
        "dem_scan_score": best_scan["scan_score"],
        "inside_expansion_boundary": best_scan["inside_expansion_boundary"],
        "active_point_x": active_point.x,
        "active_point_y": active_point.y,
        "geometry": expanded_point,
        "boundary_distance_m": row["boundary_distance_m"],
        "boundary_position_share": row["boundary_position_share"],
        "boundary_length_m": row["boundary_length_m"],
            }
        )

        scan_line_rows.append(
            {
                "stage_id": stage_id,
                f"stage_{stage_id}_candidate_id": row[f"stage_{stage_id}_candidate_id"],
                f"stage_{stage_id}_expanded_point_id": (
                    f"stage_{stage_id}_expanded_point_{int(row['candidate_id'])}"
                ),
                "candidate_id": row["candidate_id"],
                "selection_rank": row["selection_rank"],
                "selected_scan_angle_degrees": best_scan["scan_angle_degrees"],
                "expansion_distance_m": expansion_distance_m,
                "dem_scan_score": best_scan["scan_score"],
                "geometry": LineString(
                    [
                        (active_point.x, active_point.y),
                        (expanded_point.x, expanded_point.y),
                    ]
                ),
            }
        )

    expanded_points_gdf = make_geometry_gdf(
        rows=expanded_point_rows,
        crs=crs,
    )

    scan_lines_gdf = make_geometry_gdf(
        rows=scan_line_rows,
        crs=crs,
    )

    return {
        "expanded_points_gdf": expanded_points_gdf,
        "scan_lines_gdf": scan_lines_gdf,
    }


# ============================================================
# Stage expansion helper 7:
# Allocate water to expanded points
# ============================================================

def allocate_water_to_expanded_points(
    expanded_points_gdf,
    stage_id,
    outflow_share=0.60,
    parent_zone_col="parent_zone_id",
    parent_water_col="parent_water_budget",
    dem_scan_score_col="dem_scan_score",
    active_score_share_col=None,
    min_child_strength=1e-9,
    crs="EPSG:3005",
):
    """
    Allocate outflow water from parent pressure zones to expanded points.

    For each parent zone:
        parent_outflow_water = parent_water_budget * outflow_share

    Then split that outflow among expanded child points from the same parent
    using child strength.

    child_strength = dem_scan_score * active_score_share

    This function does not update the parent zones yet.
    It only attaches allocated water to expanded points.
    """

    if expanded_points_gdf.empty:
        raise ValueError("expanded_points_gdf is empty.")

    if active_score_share_col is None:
        active_score_share_col = f"active_stage_{stage_id}_score_share"

    required_cols = [
        parent_zone_col,
        parent_water_col,
        dem_scan_score_col,
        active_score_share_col,
        "geometry",
    ]

    for col in required_cols:
        if col not in expanded_points_gdf.columns:
            raise ValueError(f"expanded_points_gdf is missing column: {col}")

    out_gdf = expanded_points_gdf.copy()

    out_gdf[parent_zone_col] = out_gdf[parent_zone_col].astype(str)

    # ------------------------------------------------------------
    # Child strength
    # ------------------------------------------------------------

    out_gdf[f"stage_{stage_id}_child_water_strength"] = (
        out_gdf[dem_scan_score_col].clip(lower=0)
        * out_gdf[active_score_share_col].clip(lower=0)
    )

    out_gdf[f"stage_{stage_id}_child_water_strength"] = out_gdf[
        f"stage_{stage_id}_child_water_strength"
    ].clip(lower=min_child_strength)

    # ------------------------------------------------------------
    # Parent-level totals
    # ------------------------------------------------------------

    parent_summary_df = (
        out_gdf
        .groupby(parent_zone_col)
        .agg(
            parent_original_water_budget=(parent_water_col, "first"),
            total_child_water_strength=(
                f"stage_{stage_id}_child_water_strength",
                "sum",
            ),
            n_expanded_children=("geometry", "count"),
        )
        .reset_index()
    )

    parent_summary_df["parent_outflow_share"] = outflow_share

    parent_summary_df["parent_outflow_water_budget"] = (
        parent_summary_df["parent_original_water_budget"] * outflow_share
    )

    parent_summary_df["parent_remaining_water_budget"] = (
        parent_summary_df["parent_original_water_budget"]
        - parent_summary_df["parent_outflow_water_budget"]
    )

    # ------------------------------------------------------------
    # Join parent summary back to expanded points
    # ------------------------------------------------------------

    out_gdf = out_gdf.merge(
        parent_summary_df,
        on=parent_zone_col,
        how="left",
    )

    out_gdf[f"stage_{stage_id}_child_relative_water_share"] = (
        out_gdf[f"stage_{stage_id}_child_water_strength"]
        / out_gdf["total_child_water_strength"].clip(lower=min_child_strength)
    )

    out_gdf[f"stage_{stage_id}_allocated_water_budget"] = (
        out_gdf[f"stage_{stage_id}_child_relative_water_share"]
        * out_gdf["parent_outflow_water_budget"]
    )

    # ------------------------------------------------------------
    # Build parent update table
    # ------------------------------------------------------------

    parent_update_gdf = parent_summary_df.copy()

    return {
        "expanded_points_with_water_gdf": gpd.GeoDataFrame(
            out_gdf,
            geometry="geometry",
            crs=crs,
        ),
        "parent_water_update_df": parent_update_gdf,
    }


# ============================================================
# Stage expansion helper 8:
# Build next footprint from radial scan
# ============================================================

from shapely.geometry import MultiPoint
from shapely.ops import unary_union


def get_point_angle_and_distance_from_source(point, source_point):
    """
    Get radial angle and distance from source point to another point.
    """

    dx = point.x - source_point.x
    dy = point.y - source_point.y

    angle_degrees = np.degrees(np.arctan2(dy, dx)) % 360
    distance_m = source_point.distance(point)

    return angle_degrees, distance_m


def get_boundary_point_at_angle(
    source_point,
    boundary_geometry,
    angle_degrees,
    search_distance_m,
):
    """
    Get approximate current footprint boundary point along a radial angle.

    Creates a long ray from the source point and intersects it with the
    current footprint boundary. Keeps the farthest intersection point.
    """

    angle_radians = np.deg2rad(angle_degrees)

    ray_end_point = Point(
        source_point.x + search_distance_m * np.cos(angle_radians),
        source_point.y + search_distance_m * np.sin(angle_radians),
    )

    ray_line = LineString(
        [
            (source_point.x, source_point.y),
            (ray_end_point.x, ray_end_point.y),
        ]
    )

    intersection = ray_line.intersection(boundary_geometry)

    if intersection.is_empty:
        return None

    points = []

    if intersection.geom_type == "Point":
        points = [intersection]

    elif intersection.geom_type == "MultiPoint":
        points = list(intersection.geoms)

    elif intersection.geom_type in ["LineString", "LinearRing"]:
        points = [
            Point(intersection.coords[0]),
            Point(intersection.coords[-1]),
        ]

    elif intersection.geom_type == "MultiLineString":
        for line in intersection.geoms:
            points.append(Point(line.coords[0]))
            points.append(Point(line.coords[-1]))

    elif intersection.geom_type == "GeometryCollection":
        for geom in intersection.geoms:
            if geom.geom_type == "Point":
                points.append(geom)
            elif geom.geom_type in ["LineString", "LinearRing"]:
                points.append(Point(geom.coords[0]))
                points.append(Point(geom.coords[-1]))

    if len(points) == 0:
        return None

    farthest_point = max(
        points,
        key=lambda p: source_point.distance(p),
    )

    return farthest_point



# ============================================================
# Stage expansion helper:
# Explode_geometry_to_parts
# ============================================================
def explode_geometry_to_parts(geometry):
    """
    Split a Shapely geometry into individual polygon parts.

    Used when polygon creation returns a Polygon, MultiPolygon, or GeometryCollection.
    """

    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type == "Polygon":
        return [geometry]

    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)

    if geometry.geom_type == "GeometryCollection":
        polygon_parts = []

        for part in geometry.geoms:
            if part is None or part.is_empty:
                continue

            if part.geom_type == "Polygon":
                polygon_parts.append(part)

            elif part.geom_type == "MultiPolygon":
                polygon_parts.extend(list(part.geoms))

        return polygon_parts

    return []

# ============================================================
# Stage expansion helper:
# Build expansion lobes from expanded points
# ============================================================

def build_expansion_lobes_from_expanded_points(
    expanded_points_with_water_gdf,
    stage_id,
    point_influence_radius_m=75,
    connection_buffer_m=35,
    allocated_water_col=None,
    crs="EPSG:3005",
):
    """
    Build connected expansion lobes from expanded points.

    Logic:
        - create a small buffer around each expanded point
        - if two point buffers overlap, connect those points
        - connected points become one expansion lobe
        - lobe geometry is the union of point buffers and buffered connection lines

    This creates local expansion blobs before the radial footprint scan.
    """

    if expanded_points_with_water_gdf.empty:
        raise ValueError("expanded_points_with_water_gdf is empty.")

    expanded_point_id_col = f"stage_{stage_id}_expanded_point_id"

    if allocated_water_col is None:
        allocated_water_col = f"stage_{stage_id}_allocated_water_budget"

    required_cols = [
        expanded_point_id_col,
        allocated_water_col,
        "geometry",
    ]

    for col in required_cols:
        if col not in expanded_points_with_water_gdf.columns:
            raise ValueError(f"expanded_points_with_water_gdf is missing column: {col}")

    points_gdf = expanded_points_with_water_gdf.copy().reset_index(drop=True)

    n_points = len(points_gdf)

    # ------------------------------------------------------------
    # Build influence buffers around each expanded point
    # ------------------------------------------------------------

    influence_rows = []

    for i, row in points_gdf.iterrows():

        influence_geometry = row.geometry.buffer(point_influence_radius_m)
        influence_geometry = clean_geometry(influence_geometry)

        influence_rows.append(
            {
                "point_index": i,
                expanded_point_id_col: row[expanded_point_id_col],
                allocated_water_col: row[allocated_water_col],
                "point_influence_radius_m": point_influence_radius_m,
                "geometry": influence_geometry,
            }
        )

    point_influence_gdf = make_geometry_gdf(
        rows=influence_rows,
        crs=crs,
    )

    # ------------------------------------------------------------
    # Find overlapping influence buffers and build connection graph
    # ------------------------------------------------------------

    adjacency = {i: set() for i in range(n_points)}
    connection_rows = []

    for i in range(n_points):

        point_i = points_gdf.loc[i].geometry
        buffer_i = point_influence_gdf.loc[i].geometry

        for j in range(i + 1, n_points):

            point_j = points_gdf.loc[j].geometry
            buffer_j = point_influence_gdf.loc[j].geometry

            buffers_overlap = (
                buffer_i.intersects(buffer_j)
                or buffer_i.touches(buffer_j)
            )

            if buffers_overlap:

                adjacency[i].add(j)
                adjacency[j].add(i)

                connection_line = LineString(
                    [
                        (point_i.x, point_i.y),
                        (point_j.x, point_j.y),
                    ]
                )

                connection_geometry = connection_line.buffer(connection_buffer_m)
                connection_geometry = clean_geometry(connection_geometry)

                connection_rows.append(
                    {
                        "from_point_index": i,
                        "to_point_index": j,
                        "from_expanded_point_id": points_gdf.loc[i, expanded_point_id_col],
                        "to_expanded_point_id": points_gdf.loc[j, expanded_point_id_col],
                        "connection_distance_m": point_i.distance(point_j),
                        "connection_buffer_m": connection_buffer_m,
                        "geometry": connection_geometry,
                    }
                )

    if len(connection_rows) > 0:
        connection_lines_gdf = make_geometry_gdf(
            rows=connection_rows,
            crs=crs,
        )
    else:
        connection_lines_gdf = gpd.GeoDataFrame(
            pd.DataFrame(),
            geometry=[],
            crs=crs,
        )

    # ------------------------------------------------------------
    # Find connected components / lobes
    # ------------------------------------------------------------

    visited = set()
    component_id_by_point_index = {}

    current_component_id = 0

    for i in range(n_points):

        if i in visited:
            continue

        stack = [i]
        component_members = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component_members.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

        for member_index in component_members:
            component_id_by_point_index[member_index] = current_component_id

        current_component_id += 1

    points_gdf[f"stage_{stage_id}_expansion_lobe_number"] = (
        points_gdf.index.map(component_id_by_point_index)
    )

    points_gdf[f"stage_{stage_id}_expansion_lobe_id"] = (
        "stage_"
        + str(stage_id)
        + "_lobe_"
        + points_gdf[f"stage_{stage_id}_expansion_lobe_number"].astype(str)
    )

    # ------------------------------------------------------------
    # Build lobe geometries
    # ------------------------------------------------------------

    lobe_rows = []

    for lobe_number, lobe_points_gdf in points_gdf.groupby(
        f"stage_{stage_id}_expansion_lobe_number"
    ):

        lobe_point_indices = list(lobe_points_gdf.index)

        lobe_geometries = []

        # Add point influence buffers.
        for point_index in lobe_point_indices:
            lobe_geometries.append(point_influence_gdf.loc[point_index].geometry)

        # Add connection geometries for connections inside this lobe.
        if not connection_lines_gdf.empty:
            lobe_connections_gdf = connection_lines_gdf[
                connection_lines_gdf["from_point_index"].isin(lobe_point_indices)
                & connection_lines_gdf["to_point_index"].isin(lobe_point_indices)
            ].copy()

            for geom in lobe_connections_gdf.geometry:
                lobe_geometries.append(geom)

        lobe_geometry = unary_union(lobe_geometries)
        lobe_geometry = clean_geometry(lobe_geometry)

        lobe_id = f"stage_{stage_id}_lobe_{int(lobe_number)}"

        lobe_rows.append(
            {
                "stage_id": stage_id,
                f"stage_{stage_id}_expansion_lobe_number": int(lobe_number),
                f"stage_{stage_id}_expansion_lobe_id": lobe_id,
                "n_expanded_points": len(lobe_points_gdf),
                "total_allocated_water_budget": lobe_points_gdf[
                    allocated_water_col
                ].sum(),
                "mean_allocated_water_budget": lobe_points_gdf[
                    allocated_water_col
                ].mean(),
                "point_influence_radius_m": point_influence_radius_m,
                "connection_buffer_m": connection_buffer_m,
                "geometry": lobe_geometry,
            }
        )

    expansion_lobes_gdf = make_geometry_gdf(
        rows=lobe_rows,
        crs=crs,
    )

    points_with_lobes_gdf = gpd.GeoDataFrame(
        points_gdf,
        geometry="geometry",
        crs=crs,
    )

    return {
        "expanded_points_with_lobes_gdf": points_with_lobes_gdf,
        "point_influence_gdf": point_influence_gdf,
        "connection_lines_gdf": connection_lines_gdf,
        "expansion_lobes_gdf": expansion_lobes_gdf,
    }


# ============================================================
# Stage expansion helper:
# Build next footprint from expansion lobes using radial scan
# ============================================================

def extract_points_from_intersection_geometry(intersection_geometry):
    """
    Extract point-like coordinates from a ray/geometry intersection.
    """

    if intersection_geometry is None or intersection_geometry.is_empty:
        return []

    points = []

    if intersection_geometry.geom_type == "Point":
        points.append(intersection_geometry)

    elif intersection_geometry.geom_type == "MultiPoint":
        points.extend(list(intersection_geometry.geoms))

    elif intersection_geometry.geom_type in ["LineString", "LinearRing"]:
        points.append(Point(intersection_geometry.coords[0]))
        points.append(Point(intersection_geometry.coords[-1]))

    elif intersection_geometry.geom_type == "MultiLineString":
        for line in intersection_geometry.geoms:
            points.append(Point(line.coords[0]))
            points.append(Point(line.coords[-1]))

    elif intersection_geometry.geom_type == "Polygon":
        for coord in intersection_geometry.exterior.coords:
            points.append(Point(coord))

    elif intersection_geometry.geom_type == "MultiPolygon":
        for polygon in intersection_geometry.geoms:
            for coord in polygon.exterior.coords:
                points.append(Point(coord))

    elif intersection_geometry.geom_type == "GeometryCollection":
        for geom in intersection_geometry.geoms:
            points.extend(extract_points_from_intersection_geometry(geom))

    return points


def get_farthest_intersection_point_from_source(
    source_point,
    target_geometry,
    angle_degrees,
    search_distance_m,
):
    """
    Cast a ray from source_point and return the farthest intersection point
    with target_geometry.
    """

    angle_radians = np.deg2rad(angle_degrees)

    ray_end_point = Point(
        source_point.x + search_distance_m * np.cos(angle_radians),
        source_point.y + search_distance_m * np.sin(angle_radians),
    )

    ray_line = LineString(
        [
            (source_point.x, source_point.y),
            (ray_end_point.x, ray_end_point.y),
        ]
    )

    intersection_geometry = ray_line.intersection(target_geometry)

    points = extract_points_from_intersection_geometry(
        intersection_geometry=intersection_geometry
    )

    if len(points) == 0:
        return None

    farthest_point = max(
        points,
        key=lambda point: source_point.distance(point),
    )

    return farthest_point


def build_next_stage_footprint_from_lobes_radial_scan(
    source_point,
    current_footprint_geometry,
    expansion_lobes_gdf,
    expansion_boundary,
    stage_id,
    n_wedges=64,
    min_outward_gain_m=5,
    crs="EPSG:3005",
):
    """
    Build the next flood footprint using radial scan against expansion lobes.

    For each radial direction:
        - find current footprint boundary point
        - find farthest expansion lobe intersection point
        - if the lobe point is farther outward, use it
        - otherwise keep current boundary point

    This creates a connected footprint from expansion lobe geometry.
    """

    if current_footprint_geometry is None or current_footprint_geometry.is_empty:
        raise ValueError("current_footprint_geometry is empty.")

    if expansion_lobes_gdf.empty:
        raise ValueError("expansion_lobes_gdf is empty.")

    if expansion_boundary is None or expansion_boundary.is_empty:
        raise ValueError("expansion_boundary is empty.")

    stage_next_id = stage_id + 1

    current_footprint_geometry = clean_geometry(current_footprint_geometry)

    lobe_union_geometry = unary_union(list(expansion_lobes_gdf.geometry))
    lobe_union_geometry = clean_geometry(lobe_union_geometry)

    if lobe_union_geometry is None or lobe_union_geometry.is_empty:
        raise ValueError("lobe_union_geometry is empty.")

    max_current_distance_m = max(
        source_point.distance(Point(x, y))
        for x, y in current_footprint_geometry.envelope.exterior.coords
    )

    max_lobe_distance_m = max(
        source_point.distance(Point(x, y))
        for x, y in lobe_union_geometry.envelope.exterior.coords
    )

    search_distance_m = max(
        max_current_distance_m,
        max_lobe_distance_m,
    ) * 2

    wedge_center_angles = np.linspace(
        0,
        360,
        n_wedges,
        endpoint=False,
    )

    selected_rows = []

    for wedge_id, wedge_center_angle in enumerate(wedge_center_angles):

        current_boundary_point = get_farthest_intersection_point_from_source(
            source_point=source_point,
            target_geometry=current_footprint_geometry.boundary,
            angle_degrees=wedge_center_angle,
            search_distance_m=search_distance_m,
        )

        if current_boundary_point is None:
            continue

        current_boundary_distance_m = source_point.distance(
            current_boundary_point
        )

        lobe_point = get_farthest_intersection_point_from_source(
            source_point=source_point,
            target_geometry=lobe_union_geometry.boundary,
            angle_degrees=wedge_center_angle,
            search_distance_m=search_distance_m,
        )

        selected_point = current_boundary_point
        selected_source = "current_boundary"
        selected_distance_m = current_boundary_distance_m
        lobe_distance_m = np.nan
        outward_gain_m = 0.0

        if lobe_point is not None:

            lobe_distance_m = source_point.distance(lobe_point)
            outward_gain_m = lobe_distance_m - current_boundary_distance_m

            if outward_gain_m >= min_outward_gain_m:
                selected_point = lobe_point
                selected_source = "expansion_lobe"
                selected_distance_m = lobe_distance_m

        selected_rows.append(
            {
                "stage_id": stage_id,
                "stage_next_id": stage_next_id,
                "wedge_id": wedge_id,
                "wedge_center_angle_degrees": wedge_center_angle,
                "selected_source": selected_source,
                "current_boundary_distance_m": current_boundary_distance_m,
                "lobe_distance_m": lobe_distance_m,
                "selected_distance_from_source_m": selected_distance_m,
                "outward_gain_m": outward_gain_m,
                "geometry": selected_point,
            }
        )

    selected_outer_points_gdf = make_geometry_gdf(
        rows=selected_rows,
        crs=crs,
    )

    if len(selected_outer_points_gdf) < 3:
        raise ValueError("Not enough selected outer points to build footprint polygon.")

    selected_outer_points_sorted = (
        selected_outer_points_gdf
        .sort_values("wedge_center_angle_degrees")
        .reset_index(drop=True)
        .copy()
    )

    polygon_coords = [
        (point.x, point.y)
        for point in selected_outer_points_sorted.geometry
    ]

    polygon_coords.append(polygon_coords[0])

    next_footprint_geometry = Polygon(polygon_coords)
    next_footprint_geometry = clean_geometry(next_footprint_geometry)

    # Never lose already-flooded area.
    next_footprint_geometry = next_footprint_geometry.union(
        current_footprint_geometry
    )

    # Also include the lobe geometry itself, so local lobe areas are not lost
    # between radial scan lines.
    next_footprint_geometry = next_footprint_geometry.union(
        lobe_union_geometry
    )

    next_footprint_geometry = next_footprint_geometry.intersection(
        expansion_boundary
    )

    next_footprint_geometry = clean_geometry(next_footprint_geometry)

    new_area_geometry = next_footprint_geometry.difference(
        current_footprint_geometry
    )

    new_area_geometry = clean_geometry(new_area_geometry)

    return {
        "next_footprint_geometry": next_footprint_geometry,
        "new_area_geometry": new_area_geometry,
        "selected_outer_points_gdf": selected_outer_points_gdf,
        "lobe_union_geometry": lobe_union_geometry,
    }



# ============================================================
# Stage expansion helper:
# Classify radial scan boundary segments
# ============================================================

def classify_radial_scan_boundary_segments(
    selected_outer_points_gdf,
    stage_id,
    crs="EPSG:3005",
):
    """
    Classify boundary segments from the radial scan output.

    Segment types:
        - current_boundary:
            both endpoints are on the previous/current footprint boundary

        - edge:
            one endpoint is on the previous/current footprint boundary
            and one endpoint is on the new Stage 2 / expansion lobe boundary

        - connector:
            both endpoints are on the new Stage 2 / expansion lobe boundary

    This is used before building next-stage pressure polygons.
    """

    if selected_outer_points_gdf.empty:
        raise ValueError("selected_outer_points_gdf is empty.")

    required_cols = [
        "wedge_id",
        "wedge_center_angle_degrees",
        "selected_source",
        "geometry",
    ]

    for col in required_cols:
        if col not in selected_outer_points_gdf.columns:
            raise ValueError(f"selected_outer_points_gdf is missing column: {col}")

    points_gdf = (
        selected_outer_points_gdf
        .sort_values("wedge_center_angle_degrees")
        .reset_index(drop=True)
        .copy()
    )

    n_points = len(points_gdf)

    segment_rows = []

    for i in range(n_points):

        next_i = (i + 1) % n_points

        point_a = points_gdf.loc[i].geometry
        point_b = points_gdf.loc[next_i].geometry

        source_a = points_gdf.loc[i, "selected_source"]
        source_b = points_gdf.loc[next_i, "selected_source"]

        if source_a == "current_boundary" and source_b == "current_boundary":
            boundary_segment_type = "current_boundary"

        elif source_a != "current_boundary" and source_b != "current_boundary":
            boundary_segment_type = "connector"

        else:
            boundary_segment_type = "edge"

        line_geometry = LineString(
            [
                (point_a.x, point_a.y),
                (point_b.x, point_b.y),
            ]
        )

        segment_rows.append(
            {
                "stage_id": stage_id,
                "stage_next_id": stage_id + 1,
                "boundary_segment_id": f"stage_{stage_id}_boundary_segment_{i}",
                "from_point_index": i,
                "to_point_index": next_i,
                "from_wedge_id": points_gdf.loc[i, "wedge_id"],
                "to_wedge_id": points_gdf.loc[next_i, "wedge_id"],
                "from_angle_degrees": points_gdf.loc[i, "wedge_center_angle_degrees"],
                "to_angle_degrees": points_gdf.loc[next_i, "wedge_center_angle_degrees"],
                "from_selected_source": source_a,
                "to_selected_source": source_b,
                "boundary_segment_type": boundary_segment_type,
                "segment_length_m": line_geometry.length,
                "geometry": line_geometry,
            }
        )

    boundary_segments_gdf = make_geometry_gdf(
        rows=segment_rows,
        crs=crs,
    )

    return boundary_segments_gdf


# ============================================================
# Stage expansion helper:
# Build depth-zone polygons from edge/connector runs
# ============================================================

def create_next_stage_polygons_from_boundary_segment_runs(
    source_point,
    new_area_geometry,
    selected_outer_points_gdf,
    boundary_segments_gdf,
    stage_id,
    severity_max_depth_m,
    depth_scale_factor=50_000,
    min_area_sqm=1.0,
    crs="EPSG:3005",
):
    """
    Create next-stage pressure/depth polygons from classified radial boundary segments.

    A new pressure polygon is created from each run:

        edge -> connector(s) -> edge

    The polygon is first built as a radial fan from the source point, then
    clipped to the actual new_area_geometry. This avoids sliver depth issues
    from nearest-lobe assignment.
    """

    if source_point is None or source_point.is_empty:
        raise ValueError("source_point is empty.")

    if new_area_geometry is None or new_area_geometry.is_empty:
        raise ValueError("new_area_geometry is empty.")

    if selected_outer_points_gdf.empty:
        raise ValueError("selected_outer_points_gdf is empty.")

    if boundary_segments_gdf.empty:
        raise ValueError("boundary_segments_gdf is empty.")

    required_segment_cols = [
        "from_point_index",
        "to_point_index",
        "boundary_segment_type",
        "geometry",
    ]

    for col in required_segment_cols:
        if col not in boundary_segments_gdf.columns:
            raise ValueError(f"boundary_segments_gdf is missing column: {col}")

    required_point_cols = [
        "wedge_id",
        "wedge_center_angle_degrees",
        "selected_source",
        "geometry",
    ]

    for col in required_point_cols:
        if col not in selected_outer_points_gdf.columns:
            raise ValueError(f"selected_outer_points_gdf is missing column: {col}")

    stage_next_id = stage_id + 1

    points_gdf = (
        selected_outer_points_gdf
        .sort_values("wedge_center_angle_degrees")
        .reset_index(drop=True)
        .copy()
    )

    segments_gdf = (
        boundary_segments_gdf
        .sort_values("from_point_index")
        .reset_index(drop=True)
        .copy()
    )

    n_segments = len(segments_gdf)

    # ------------------------------------------------------------
    # Find connector runs.
    # A run starts after an edge and continues through connector
    # segments until the next edge.
    # ------------------------------------------------------------

    connector_run_rows = []
    visited_segments = set()
    run_id = 0

    for start_idx in range(n_segments):

        if start_idx in visited_segments:
            continue

        if segments_gdf.loc[start_idx, "boundary_segment_type"] != "edge":
            continue

        run_segment_indices = [start_idx]
        visited_segments.add(start_idx)

        current_idx = (start_idx + 1) % n_segments

        while current_idx != start_idx:

            segment_type = segments_gdf.loc[current_idx, "boundary_segment_type"]

            if segment_type == "current_boundary":
                break

            run_segment_indices.append(current_idx)
            visited_segments.add(current_idx)

            if segment_type == "edge" and current_idx != start_idx:
                break

            current_idx = (current_idx + 1) % n_segments

        segment_types = list(
            segments_gdf.loc[run_segment_indices, "boundary_segment_type"]
        )

        # We only keep proper expansion runs:
        # edge -> connector(s) -> edge
        if (
            len(run_segment_indices) >= 3
            and segment_types[0] == "edge"
            and segment_types[-1] == "edge"
            and "connector" in segment_types[1:-1]
        ):
            connector_run_rows.append(
                {
                    "run_id": run_id,
                    "segment_indices": run_segment_indices,
                    "segment_types": segment_types,
                }
            )
            run_id += 1

    # ------------------------------------------------------------
    # Build one clipped polygon per connector run.
    # ------------------------------------------------------------

    polygon_rows = []

    for run in connector_run_rows:

        run_segment_indices = run["segment_indices"]

        # Collect boundary point indices touched by this run.
        point_indices = []

        for segment_idx in run_segment_indices:
            from_idx = int(segments_gdf.loc[segment_idx, "from_point_index"])
            to_idx = int(segments_gdf.loc[segment_idx, "to_point_index"])

            if len(point_indices) == 0:
                point_indices.append(from_idx)

            point_indices.append(to_idx)

        # Remove consecutive duplicates while preserving order.
        ordered_point_indices = []

        for idx in point_indices:
            if len(ordered_point_indices) == 0 or ordered_point_indices[-1] != idx:
                ordered_point_indices.append(idx)

        boundary_points = [
            points_gdf.loc[idx].geometry
            for idx in ordered_point_indices
            if idx in points_gdf.index
        ]

        if len(boundary_points) < 2:
            continue

        fan_coords = [(source_point.x, source_point.y)]

        for point in boundary_points:
            fan_coords.append((point.x, point.y))

        fan_coords.append((source_point.x, source_point.y))

        fan_geometry = Polygon(fan_coords)
        fan_geometry = clean_geometry(fan_geometry)

        depth_polygon_geometry = new_area_geometry.intersection(fan_geometry)
        depth_polygon_geometry = clean_geometry(depth_polygon_geometry)

        if depth_polygon_geometry is None or depth_polygon_geometry.is_empty:
            continue

        polygon_parts = explode_geometry_to_parts(depth_polygon_geometry)

        for polygon_part in polygon_parts:

            polygon_part = clean_geometry(polygon_part)

            if polygon_part is None or polygon_part.is_empty:
                continue

            polygon_area_sqm = polygon_part.area

            if polygon_area_sqm < min_area_sqm:
                continue

            # For now, use area-based water proxy.
            # We will replace this with lobe water allocation after geometry works.
            segment_water_budget = polygon_area_sqm / new_area_geometry.area

            raw_depth_proxy = segment_water_budget / max(polygon_area_sqm, 1e-6)

            max_depth_m = min(
                raw_depth_proxy * depth_scale_factor,
                severity_max_depth_m,
            )

            polygon_rows.append(
                {
                    "stage_id": stage_next_id,
                    "parent_stage_id": stage_id,
                    f"stage_{stage_next_id}_depth_polygon_id": (
                        f"stage_{stage_next_id}_depth_polygon_"
                        + str(len(polygon_rows))
                    ),
                    "boundary_run_id": run["run_id"],
                    "n_boundary_segments": len(run_segment_indices),
                    "segment_sequence": " -> ".join(run["segment_types"]),
                    "assignment_method": "edge_connector_run",
                    "segment_water_budget": segment_water_budget,
                    "segment_area_sqm": polygon_area_sqm,
                    "segment_area_sqkm": polygon_area_sqm / 1_000_000,
                    "raw_depth_proxy": raw_depth_proxy,
                    "max_depth_m": max_depth_m,
                    "geometry": polygon_part,
                }
            )

    next_stage_depth_polygons_gdf = make_geometry_gdf(
        rows=polygon_rows,
        crs=crs,
    )

    return next_stage_depth_polygons_gdf



# ============================================================
# Stage expansion helper:
# Assign water/depth to next-stage polygons from expansion lobes
# ============================================================

def assign_depths_to_next_stage_polygons_from_lobes(
    next_stage_polygons_gdf,
    expansion_lobes_gdf,
    stage_id,
    severity_max_depth_m,
    depth_scale_factor=50_000,
    lobe_id_col=None,
    lobe_water_col="total_allocated_water_budget",
    polygon_id_col=None,
    min_depth_m=0.0,
    crs="EPSG:3005",
):
    """
    Assign water budgets and depths to already-built next-stage polygons.

    Each expansion lobe is assigned to exactly one next-stage polygon:
        1. first by largest overlap with the polygon
        2. if no overlap, by nearest polygon

    Then each polygon depth is calculated from:

        depth proxy = assigned water budget / polygon area

    This avoids tiny leftover sliver polygons receiving full lobe water.
    """

    if next_stage_polygons_gdf.empty:
        raise ValueError("next_stage_polygons_gdf is empty.")

    if expansion_lobes_gdf.empty:
        raise ValueError("expansion_lobes_gdf is empty.")

    stage_next_id = stage_id + 1

    if lobe_id_col is None:
        lobe_id_col = f"stage_{stage_id}_expansion_lobe_id"

    if polygon_id_col is None:
        polygon_id_col = f"stage_{stage_next_id}_depth_polygon_id"

    required_polygon_cols = [
        polygon_id_col,
        "geometry",
    ]

    required_lobe_cols = [
        lobe_id_col,
        lobe_water_col,
        "geometry",
    ]

    for col in required_polygon_cols:
        if col not in next_stage_polygons_gdf.columns:
            raise ValueError(f"next_stage_polygons_gdf is missing column: {col}")

    for col in required_lobe_cols:
        if col not in expansion_lobes_gdf.columns:
            raise ValueError(f"expansion_lobes_gdf is missing column: {col}")

    # ------------------------------------------------------------
    # Copy inputs
    # ------------------------------------------------------------

    polygons_gdf = next_stage_polygons_gdf.copy().reset_index(drop=True)
    lobes_gdf = expansion_lobes_gdf.copy().reset_index(drop=True)

    # ------------------------------------------------------------
    # Drop any old placeholder water/depth columns.
    # These may have been created during the polygon-construction step.
    # We recalculate them cleanly here.
    # ------------------------------------------------------------

    polygons_gdf = polygons_gdf.drop(
        columns=[
            "segment_water_budget",
            "segment_area_sqm",
            "segment_area_sqkm",
            "raw_depth_proxy",
            "max_depth_m",
            "n_assigned_lobes",
            "assigned_lobe_ids",
            "lobe_assignment_methods",
        ],
        errors="ignore",
    )

    polygons_gdf["_polygon_index"] = polygons_gdf.index

    # ------------------------------------------------------------
    # Assign each lobe to one polygon
    # ------------------------------------------------------------

    lobe_assignment_rows = []

    for _, lobe_row in lobes_gdf.iterrows():

        lobe_geometry = clean_geometry(lobe_row.geometry)

        if lobe_geometry is None or lobe_geometry.is_empty:
            continue

        overlap_rows = []

        for _, polygon_row in polygons_gdf.iterrows():

            polygon_geometry = clean_geometry(polygon_row.geometry)

            if polygon_geometry is None or polygon_geometry.is_empty:
                overlap_area_sqm = 0.0
                distance_to_polygon_m = np.inf

            else:
                overlap_geometry = polygon_geometry.intersection(lobe_geometry)

                if overlap_geometry is None or overlap_geometry.is_empty:
                    overlap_area_sqm = 0.0
                else:
                    overlap_area_sqm = overlap_geometry.area

                distance_to_polygon_m = lobe_geometry.distance(polygon_geometry)

            overlap_rows.append(
                {
                    "_polygon_index": polygon_row["_polygon_index"],
                    polygon_id_col: polygon_row[polygon_id_col],
                    "overlap_area_sqm": overlap_area_sqm,
                    "distance_to_polygon_m": distance_to_polygon_m,
                }
            )

        overlap_df = pd.DataFrame(overlap_rows)

        max_overlap_area = overlap_df["overlap_area_sqm"].max()

        if max_overlap_area > 0:
            best_row = (
                overlap_df
                .sort_values(
                    ["overlap_area_sqm", "distance_to_polygon_m"],
                    ascending=[False, True],
                )
                .iloc[0]
            )

            assignment_method = "largest_lobe_overlap"

        else:
            best_row = (
                overlap_df
                .sort_values("distance_to_polygon_m", ascending=True)
                .iloc[0]
            )

            assignment_method = "nearest_polygon_fallback"

        lobe_assignment_rows.append(
            {
                lobe_id_col: lobe_row[lobe_id_col],
                lobe_water_col: float(lobe_row[lobe_water_col]),
                "_polygon_index": int(best_row["_polygon_index"]),
                polygon_id_col: best_row[polygon_id_col],
                "lobe_assignment_method": assignment_method,
                "lobe_overlap_area_sqm": float(best_row["overlap_area_sqm"]),
                "lobe_distance_to_polygon_m": float(best_row["distance_to_polygon_m"]),
            }
        )

    lobe_assignment_df = pd.DataFrame(lobe_assignment_rows)

    if lobe_assignment_df.empty:
        raise ValueError("No lobes could be assigned to polygons.")

    # ------------------------------------------------------------
    # Sum lobe water by polygon
    # ------------------------------------------------------------

    polygon_water_df = (
        lobe_assignment_df
        .groupby("_polygon_index")
        .agg(
            segment_water_budget=(lobe_water_col, "sum"),
            n_assigned_lobes=(lobe_id_col, "count"),
            assigned_lobe_ids=(
                lobe_id_col,
                lambda x: ", ".join(x.astype(str)),
            ),
            lobe_assignment_methods=(
                "lobe_assignment_method",
                lambda x: ", ".join(sorted(set(x.astype(str)))),
            ),
        )
        .reset_index()
    )

    polygons_gdf = polygons_gdf.merge(
        polygon_water_df,
        on="_polygon_index",
        how="left",
    )

    # Fill polygons with no assigned lobe.
    polygons_gdf["segment_water_budget"] = polygons_gdf[
        "segment_water_budget"
    ].fillna(0.0)

    polygons_gdf["n_assigned_lobes"] = polygons_gdf[
        "n_assigned_lobes"
    ].fillna(0).astype(int)

    polygons_gdf["assigned_lobe_ids"] = polygons_gdf[
        "assigned_lobe_ids"
    ].fillna("")

    polygons_gdf["lobe_assignment_methods"] = polygons_gdf[
        "lobe_assignment_methods"
    ].fillna("none")

    # ------------------------------------------------------------
    # Calculate area and depth
    # ------------------------------------------------------------

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

    polygons_gdf = polygons_gdf.drop(
        columns=["_polygon_index"],
        errors="ignore",
    )

    next_stage_depth_polygons_gdf = gpd.GeoDataFrame(
        polygons_gdf,
        geometry="geometry",
        crs=crs,
    )

    return {
        "next_stage_depth_polygons_gdf": next_stage_depth_polygons_gdf,
        "lobe_assignment_df": lobe_assignment_df,
    }


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
    Add each assigned lobe's new-area geometry to its depth polygon.

    The radial-run polygons assign water correctly, but they can miss the
    rounded lobe caps that were included in the stage footprint. This keeps the
    footprint geometry and depth geometry aligned.
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

    depth_union_geometry = unary_union(list(polygons_gdf.geometry))
    depth_union_geometry = clean_geometry(depth_union_geometry)

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

        final_depth_union_geometry = unary_union(list(polygons_gdf.geometry))
        final_depth_union_geometry = clean_geometry(final_depth_union_geometry)
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



# ============================================================
# Stage expansion helper:
# Update current water zones after expansion
# ============================================================

def update_current_water_zones_after_expansion(
    current_water_zones_gdf,
    parent_water_update_df,
    next_stage_depth_polygons_gdf,
    stage_id,
    min_water_budget=0.001,
    min_depth_m=0.05,
    crs="EPSG:3005",
):
    """
    Update current water/pressure zones after one expansion stage.

    Existing parent zones lose the water that flowed outward.
    New depth polygons become new current water zones for the next stage.

    Returns
    -------
    updated_current_water_zones_gdf
    """

    if current_water_zones_gdf.empty:
        raise ValueError("current_water_zones_gdf is empty.")

    if parent_water_update_df.empty:
        raise ValueError("parent_water_update_df is empty.")

    if next_stage_depth_polygons_gdf.empty:
        raise ValueError("next_stage_depth_polygons_gdf is empty.")

    stage_next_id = stage_id + 1

    required_current_cols = [
        "current_zone_id",
        "current_stage_origin",
        "current_water_budget",
        "current_max_depth_m",
        "geometry",
    ]

    for col in required_current_cols:
        if col not in current_water_zones_gdf.columns:
            raise ValueError(f"current_water_zones_gdf is missing column: {col}")

    required_parent_update_cols = [
        "parent_zone_id",
        "parent_original_water_budget",
        "parent_remaining_water_budget",
    ]

    for col in required_parent_update_cols:
        if col not in parent_water_update_df.columns:
            raise ValueError(f"parent_water_update_df is missing column: {col}")

    next_stage_polygon_id_col = f"stage_{stage_next_id}_depth_polygon_id"

    required_next_stage_cols = [
        next_stage_polygon_id_col,
        "segment_water_budget",
        "max_depth_m",
        "geometry",
    ]

    for col in required_next_stage_cols:
        if col not in next_stage_depth_polygons_gdf.columns:
            raise ValueError(f"next_stage_depth_polygons_gdf is missing column: {col}")

    # ------------------------------------------------------------
    # Build lookup for parent remaining water
    # ------------------------------------------------------------

    parent_update_lookup = (
        parent_water_update_df
        .set_index("parent_zone_id")
        .to_dict(orient="index")
    )

    updated_rows = []

    # ------------------------------------------------------------
    # Update old/current zones
    # ------------------------------------------------------------

    for _, row in current_water_zones_gdf.iterrows():

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
                "current_water_budget": remaining_water_budget,
                "current_max_depth_m": updated_max_depth_m,
                "zone_update_method": update_method,
                "geometry": row.geometry,
            }
        )

    # ------------------------------------------------------------
    # Add new next-stage depth polygons as current zones
    # ------------------------------------------------------------

    for _, row in next_stage_depth_polygons_gdf.iterrows():

        new_water_budget = float(row["segment_water_budget"])
        new_max_depth_m = float(row["max_depth_m"])

        if new_water_budget < min_water_budget:
            continue

        if new_max_depth_m < min_depth_m:
            continue

        updated_rows.append(
            {
                "current_zone_id": row[next_stage_polygon_id_col],
                "current_stage_origin": f"stage_{stage_next_id}",
                "current_water_budget": new_water_budget,
                "current_max_depth_m": new_max_depth_m,
                "zone_update_method": "new_stage_depth_polygon",
                "geometry": row.geometry,
            }
        )

    updated_current_water_zones_gdf = make_geometry_gdf(
        rows=updated_rows,
        crs=crs,
    )

    return updated_current_water_zones_gdf





# ============================================================
# Orchestrator:
# Run one flood footprint expansion stage
# ============================================================

def run_one_flood_expansion_stage(
    source_point,
    current_footprint_geometry,
    current_water_zones_gdf,
    dem_path,
    expansion_boundary,
    stage_id,
    severity_max_depth_m,
    # candidate sampling
    n_candidate_points=128,
    # candidate scoring
    elevation_scale_m=1.0,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    # active point selection
    n_active_points=16,
    parent_penalty_strength=0.35,
    # distance assignment
    base_distance_m=225,
    min_distance_m=40,
    max_distance_m=425,
    score_power=0.50,
    depth_power=0.75,
    water_power=0.35,
    # DEM scan
    scan_left_degrees=11.25,
    scan_right_degrees=11.25,
    n_scan_angles=9,
    # water allocation
    outflow_share=0.60,
    # lobe construction
    point_influence_radius_m=50,
    connection_buffer_m=20,
    # radial footprint construction
    n_radial_wedges=96,
    min_outward_gain_m=5,
    # depth polygons
    depth_scale_factor=50_000,
    min_polygon_area_sqm=1.0,
    min_depth_m=0.0,
    crs="EPSG:3005",
):
    """
    Run one complete flood expansion stage.

    This takes the current footprint and current water/pressure zones,
    expands outward using DEM-guided active points, builds expansion lobes,
    creates the next footprint, then builds and depths the next-stage
    pressure polygons.

    Returns a dictionary containing all intermediate objects for QA/debugging.
    """

    if source_point is None or source_point.is_empty:
        raise ValueError("source_point is empty.")

    if current_footprint_geometry is None or current_footprint_geometry.is_empty:
        raise ValueError("current_footprint_geometry is empty.")

    if current_water_zones_gdf.empty:
        raise ValueError("current_water_zones_gdf is empty.")

    if expansion_boundary is None or expansion_boundary.is_empty:
        raise ValueError("expansion_boundary is empty.")

    stage_next_id = stage_id + 1

    # ------------------------------------------------------------
    # 1. Sample candidate points from current footprint boundary
    # ------------------------------------------------------------

    candidate_points_gdf = sample_candidate_points_from_current_footprint(
        current_footprint_geometry=current_footprint_geometry,
        source_point=source_point,
        stage_id=stage_id,
        n_candidate_points=n_candidate_points,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 2. Attach nearest pressure zone
    # ------------------------------------------------------------

    candidate_points_with_pressure_gdf = (
        attach_nearest_pressure_zone_to_candidate_points(
            candidate_points_gdf=candidate_points_gdf,
            current_water_zones_gdf=current_water_zones_gdf,
            stage_id=stage_id,
            crs=crs,
        )
    )

    # ------------------------------------------------------------
    # 3. Score candidates
    # ------------------------------------------------------------

    scored_candidate_points_gdf = score_stage_candidate_points(
        candidate_points_with_pressure_gdf=candidate_points_with_pressure_gdf,
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

    # ------------------------------------------------------------
    # 4. Select active candidate points
    # ------------------------------------------------------------

    active_points_gdf = select_active_candidate_points(
        scored_candidate_points_gdf=scored_candidate_points_gdf,
        stage_id=stage_id,
        n_active_points=n_active_points,
        parent_penalty_strength=parent_penalty_strength,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 5. Assign expansion distances
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # 6. DEM scan to expanded points
    # ------------------------------------------------------------

    dem_scan_output = expand_active_points_with_dem_scan(
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

    expanded_points_gdf = dem_scan_output["expanded_points_gdf"]
    scan_lines_gdf = dem_scan_output["scan_lines_gdf"]

    # ------------------------------------------------------------
    # 7. Allocate water to expanded points
    # ------------------------------------------------------------

    water_output = allocate_water_to_expanded_points(
        expanded_points_gdf=expanded_points_gdf,
        stage_id=stage_id,
        outflow_share=outflow_share,
        crs=crs,
    )

    expanded_points_with_water_gdf = water_output[
        "expanded_points_with_water_gdf"
    ]

    parent_water_update_df = water_output[
        "parent_water_update_df"
    ]

    # ------------------------------------------------------------
    # 8. Build expansion lobes from expanded points
    # ------------------------------------------------------------

    lobe_output = build_expansion_lobes_from_expanded_points(
        expanded_points_with_water_gdf=expanded_points_with_water_gdf,
        stage_id=stage_id,
        point_influence_radius_m=point_influence_radius_m,
        connection_buffer_m=connection_buffer_m,
        crs=crs,
    )

    expanded_points_with_lobes_gdf = lobe_output[
        "expanded_points_with_lobes_gdf"
    ]

    point_influence_gdf = lobe_output[
        "point_influence_gdf"
    ]

    connection_lines_gdf = lobe_output[
        "connection_lines_gdf"
    ]

    expansion_lobes_gdf = lobe_output[
        "expansion_lobes_gdf"
    ]

    # ------------------------------------------------------------
    # 9. Build next footprint using lobe radial scan
    # ------------------------------------------------------------

    footprint_output = build_next_stage_footprint_from_lobes_radial_scan(
        source_point=source_point,
        current_footprint_geometry=current_footprint_geometry,
        expansion_lobes_gdf=expansion_lobes_gdf,
        expansion_boundary=expansion_boundary,
        stage_id=stage_id,
        n_wedges=n_radial_wedges,
        min_outward_gain_m=min_outward_gain_m,
        crs=crs,
    )

    next_footprint_geometry = footprint_output["next_footprint_geometry"]
    new_area_geometry = footprint_output["new_area_geometry"]
    selected_outer_points_gdf = footprint_output["selected_outer_points_gdf"]
    lobe_union_geometry = footprint_output["lobe_union_geometry"]

    # ------------------------------------------------------------
    # 10. Classify boundary segments
    # ------------------------------------------------------------

    boundary_segments_gdf = classify_radial_scan_boundary_segments(
        selected_outer_points_gdf=selected_outer_points_gdf,
        stage_id=stage_id,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 11. Build next-stage pressure polygons from edge/connector runs
    # ------------------------------------------------------------

    next_stage_polygons_gdf = (
        create_next_stage_polygons_from_boundary_segment_runs(
            source_point=source_point,
            new_area_geometry=new_area_geometry,
            selected_outer_points_gdf=selected_outer_points_gdf,
            boundary_segments_gdf=boundary_segments_gdf,
            stage_id=stage_id,
            severity_max_depth_m=severity_max_depth_m,
            depth_scale_factor=depth_scale_factor,
            min_area_sqm=min_polygon_area_sqm,
            crs=crs,
        )
    )

    # ------------------------------------------------------------
    # 12. Assign depths to next-stage polygons from expansion lobes
    # ------------------------------------------------------------

    depth_output = assign_depths_to_next_stage_polygons_from_lobes(
        next_stage_polygons_gdf=next_stage_polygons_gdf,
        expansion_lobes_gdf=expansion_lobes_gdf,
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    next_stage_depth_polygons_gdf = depth_output[
        "next_stage_depth_polygons_gdf"
    ]

    lobe_assignment_df = depth_output[
        "lobe_assignment_df"
    ]

    # ------------------------------------------------------------
    # 13. Fill depth polygons with their assigned lobe geometry
    # ------------------------------------------------------------

    lobe_fill_output = fill_stage_polygons_with_assigned_lobes(
        next_stage_depth_polygons_gdf=next_stage_depth_polygons_gdf,
        expansion_lobes_gdf=expansion_lobes_gdf,
        lobe_assignment_df=lobe_assignment_df,
        new_area_geometry=new_area_geometry,
        stage_id=stage_id,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    next_stage_depth_polygons_before_lobe_fill_gdf = (
        next_stage_depth_polygons_gdf
    )

    next_stage_depth_polygons_gdf = lobe_fill_output[
        "filled_next_stage_depth_polygons_gdf"
    ]

    # ------------------------------------------------------------
    # 14. Update current water zones for next expansion stage
    # ------------------------------------------------------------

    updated_current_water_zones_gdf = update_current_water_zones_after_expansion(
        current_water_zones_gdf=current_water_zones_gdf,
        parent_water_update_df=parent_water_update_df,
        next_stage_depth_polygons_gdf=next_stage_depth_polygons_gdf,
        stage_id=stage_id,
        min_water_budget=0.001,
        min_depth_m=0.05,
        crs=crs,
    )

    # ------------------------------------------------------------
    # Return everything useful
    # ------------------------------------------------------------

    return {
        "stage_id": stage_id,
        "stage_next_id": stage_next_id,
        "candidate_points_gdf": candidate_points_gdf,
        "candidate_points_with_pressure_gdf": candidate_points_with_pressure_gdf,
        "scored_candidate_points_gdf": scored_candidate_points_gdf,
        "active_points_gdf": active_points_gdf,
        "active_points_with_distance_gdf": active_points_with_distance_gdf,
        "expanded_points_gdf": expanded_points_gdf,
        "scan_lines_gdf": scan_lines_gdf,
        "expanded_points_with_water_gdf": expanded_points_with_water_gdf,
        "parent_water_update_df": parent_water_update_df,
        "expanded_points_with_lobes_gdf": expanded_points_with_lobes_gdf,
        "point_influence_gdf": point_influence_gdf,
        "connection_lines_gdf": connection_lines_gdf,
        "expansion_lobes_gdf": expansion_lobes_gdf,
        "next_footprint_geometry": next_footprint_geometry,
        "new_area_geometry": new_area_geometry,
        "selected_outer_points_gdf": selected_outer_points_gdf,
        "lobe_union_geometry": lobe_union_geometry,
        "boundary_segments_gdf": boundary_segments_gdf,
        "next_stage_polygons_gdf": next_stage_polygons_gdf,
        "next_stage_depth_polygons_gdf": next_stage_depth_polygons_gdf,
        "next_stage_depth_polygons_before_lobe_fill_gdf": (
            next_stage_depth_polygons_before_lobe_fill_gdf
        ),
        "uncovered_new_area_geometry": lobe_fill_output[
            "uncovered_new_area_geometry"
        ],
        "lobe_fill_summary": lobe_fill_output["lobe_fill_summary"],
        "lobe_assignment_df": lobe_assignment_df,
        "updated_current_water_zones_gdf": updated_current_water_zones_gdf,
    }




# ============================================================
# Main builder:
# Build full stochastic flood footprint V2
# ============================================================

def build_flood_footprint_v2(
    selected_segment,
    dem_path,
    expansion_boundary,
    severity,
    event_water_budget,
    severity_max_depth_m,
    n_growth_stages=8,
    # initial setup
    initial_buffer_radius_m=225,
    n_stage_1_reference_points=16,
    # stage 1 scoring/depth
    elevation_scale_m=1.0,
    depth_scale_factor=50_000,
    # repeated expansion
    n_candidate_points=128,
    n_active_points=16,
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
    point_influence_radius_m=50,
    connection_buffer_m=20,
    n_radial_wedges=96,
    min_outward_gain_m=5,
    # thresholds
    min_water_budget=0.001,
    min_depth_m=0.05,
    min_polygon_area_sqm=1.0,
    # randomness
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    crs="EPSG:3005",
):
    """
    Build a full stochastic flood footprint using the V2 expansion model.

    Main output for loss modelling:
        all_depth_polygons_gdf

    This contains every depth polygon created during the event, including:
        - Stage 1 initial pressure zones
        - Stage 2+ expansion depth polygons

    Property-level flood depth should later be assigned as:
        max depth reached across all_depth_polygons_gdf.
    """

    if expansion_boundary is None or expansion_boundary.is_empty:
        raise ValueError("expansion_boundary is empty.")

    if n_growth_stages < 1:
        raise ValueError("n_growth_stages must be at least 1.")

    # ------------------------------------------------------------
    # 1. Initial flood setup
    # ------------------------------------------------------------

    initial_setup = create_initial_flood_setup(
        selected_segment=selected_segment,
        buffer_radius_m=initial_buffer_radius_m,
        expansion_boundary=expansion_boundary,
        n_reference_points=n_stage_1_reference_points,
        crs=crs,
    )

    source_point = initial_setup["source_point"]
    base_buffer_geometry = initial_setup["base_buffer_geometry"]

    # ------------------------------------------------------------
    # 2. Stage 1 scored points
    # ------------------------------------------------------------

    stage_1_points_gdf = create_stage_1_scored_points(
        reference_points_gdf=initial_setup["reference_points_gdf"],
        source_point=source_point,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        event_water_budget=event_water_budget,
        elevation_scale_m=elevation_scale_m,
        random_seed=random_seed,
        random_low=random_low,
        random_high=random_high,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 3. Stage 1 depth / pressure polygons
    # ------------------------------------------------------------

    stage_1_depth_segments_gdf = create_stage_1_depth_segments(
        source_point=source_point,
        base_buffer_geometry=base_buffer_geometry,
        stage_1_points_gdf=stage_1_points_gdf,
        severity_max_depth_m=severity_max_depth_m,
        depth_scale_factor=depth_scale_factor,
        crs=crs,
    )

    stage_1_depth_segments_gdf["depth_stage"] = 1
    stage_1_depth_segments_gdf["depth_polygon_id"] = (
        "stage_1_"
        + stage_1_depth_segments_gdf["stage_1_segment_id"].astype(str)
    )
    stage_1_depth_segments_gdf["event_severity"] = severity

    # ------------------------------------------------------------
    # 4. Initialize current water zones
    # ------------------------------------------------------------

    current_water_zones_gdf = initialize_current_water_zones_from_stage_1(
        stage_1_depth_segments_gdf=stage_1_depth_segments_gdf,
        min_water_budget=min_water_budget,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 5. Repeated expansion stages
    # ------------------------------------------------------------

    stage_outputs = {}
    stage_summary_rows = []

    current_footprint_geometry = base_buffer_geometry
    current_stage_water_zones_gdf = current_water_zones_gdf.copy()

    for stage_id in range(1, n_growth_stages + 1):

        stage_output = run_one_flood_expansion_stage(
            source_point=source_point,
            current_footprint_geometry=current_footprint_geometry,
            current_water_zones_gdf=current_stage_water_zones_gdf,
            dem_path=dem_path,
            expansion_boundary=expansion_boundary,
            stage_id=stage_id,
            severity_max_depth_m=severity_max_depth_m,
            n_candidate_points=n_candidate_points,
            elevation_scale_m=elevation_scale_m,
            random_seed=random_seed + stage_id,
            random_low=random_low,
            random_high=random_high,
            n_active_points=n_active_points,
            parent_penalty_strength=0.35,
            base_distance_m=base_distance_m,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            score_power=score_power,
            depth_power=depth_power,
            water_power=water_power,
            scan_left_degrees=scan_left_degrees,
            scan_right_degrees=scan_right_degrees,
            n_scan_angles=n_scan_angles,
            outflow_share=outflow_share,
            point_influence_radius_m=point_influence_radius_m,
            connection_buffer_m=connection_buffer_m,
            n_radial_wedges=n_radial_wedges,
            min_outward_gain_m=min_outward_gain_m,
            depth_scale_factor=depth_scale_factor,
            min_polygon_area_sqm=min_polygon_area_sqm,
            min_depth_m=min_depth_m,
            crs=crs,
        )

        stage_outputs[stage_id] = stage_output

        next_footprint_geometry = stage_output["next_footprint_geometry"]
        new_area_geometry = stage_output["new_area_geometry"]
        next_stage_depth_polygons_gdf = stage_output[
            "next_stage_depth_polygons_gdf"
        ]
        expansion_lobes_gdf = stage_output["expansion_lobes_gdf"]

        stage_summary_rows.append(
            {
                "stage_id": stage_id,
                "stage_next_id": stage_id + 1,
                "event_severity": severity,
                "candidate_points": len(stage_output["candidate_points_gdf"]),
                "active_points": len(stage_output["active_points_gdf"]),
                "expanded_points": len(stage_output["expanded_points_gdf"]),
                "expansion_lobes": len(expansion_lobes_gdf),
                "depth_polygons": len(next_stage_depth_polygons_gdf),
                "current_area_sqkm": current_footprint_geometry.area / 1_000_000,
                "next_area_sqkm": next_footprint_geometry.area / 1_000_000,
                "new_area_sqkm": new_area_geometry.area / 1_000_000,
                "assigned_lobe_water": next_stage_depth_polygons_gdf[
                    "segment_water_budget"
                ].sum(),
                "original_lobe_water": expansion_lobes_gdf[
                    "total_allocated_water_budget"
                ].sum(),
                "updated_water_zones": len(
                    stage_output["updated_current_water_zones_gdf"]
                ),
                "updated_water_total": stage_output[
                    "updated_current_water_zones_gdf"
                ]["current_water_budget"].sum(),
            }
        )

        # Update state for next expansion stage.
        current_footprint_geometry = stage_output["next_footprint_geometry"]
        current_stage_water_zones_gdf = stage_output[
            "updated_current_water_zones_gdf"
        ]

    stage_summary_df = pd.DataFrame(stage_summary_rows)

    # ------------------------------------------------------------
    # 6. Build final footprint outputs
    # ------------------------------------------------------------

    final_footprint_geometry = current_footprint_geometry

    final_footprint_gdf = make_geometry_gdf(
        rows=[
            {
                "event_severity": severity,
                "n_growth_stages": n_growth_stages,
                "final_area_sqm": final_footprint_geometry.area,
                "final_area_sqkm": final_footprint_geometry.area / 1_000_000,
                "geometry": final_footprint_geometry,
            }
        ],
        crs=crs,
    )

    # ------------------------------------------------------------
    # 7. Build all_depth_polygons_gdf
    # ------------------------------------------------------------

    all_depth_polygon_layers = []

    stage_1_depth_copy = stage_1_depth_segments_gdf.copy()

    stage_1_depth_copy["depth_stage"] = 1
    stage_1_depth_copy["depth_polygon_id"] = (
        "stage_1_"
        + stage_1_depth_copy["stage_1_segment_id"].astype(str)
    )
    stage_1_depth_copy["event_severity"] = severity

    all_depth_polygon_layers.append(stage_1_depth_copy)

    for stage_id, stage_output in stage_outputs.items():

        stage_number = stage_id + 1

        depth_gdf = stage_output["next_stage_depth_polygons_gdf"].copy()

        stage_depth_polygon_id_col = f"stage_{stage_number}_depth_polygon_id"

        depth_gdf["depth_stage"] = stage_number

        if stage_depth_polygon_id_col in depth_gdf.columns:
            depth_gdf["depth_polygon_id"] = depth_gdf[
                stage_depth_polygon_id_col
            ].astype(str)
        else:
            depth_gdf["depth_polygon_id"] = (
                "stage_"
                + str(stage_number)
                + "_depth_polygon_"
                + depth_gdf.index.astype(str)
            )

        depth_gdf["event_severity"] = severity

        all_depth_polygon_layers.append(depth_gdf)

    all_depth_polygons_gdf = pd.concat(
        all_depth_polygon_layers,
        ignore_index=True,
    )

    all_depth_polygons_gdf = gpd.GeoDataFrame(
        all_depth_polygons_gdf,
        geometry="geometry",
        crs=crs,
    )

    # Keep a clean common set of columns near the front.
    front_cols = [
        "depth_stage",
        "depth_polygon_id",
        "event_severity",
        "max_depth_m",
        "segment_water_budget",
        "segment_area_sqm",
        "segment_area_sqkm",
        "geometry",
    ]

    existing_front_cols = [
        col for col in front_cols
        if col in all_depth_polygons_gdf.columns
    ]

    remaining_cols = [
        col for col in all_depth_polygons_gdf.columns
        if col not in existing_front_cols
    ]

    all_depth_polygons_gdf = all_depth_polygons_gdf[
        existing_front_cols + remaining_cols
    ]

    # ------------------------------------------------------------
    # 8. Return official output package
    # ------------------------------------------------------------

    return {
        "severity": severity,
        "event_water_budget": event_water_budget,
        "severity_max_depth_m": severity_max_depth_m,
        "n_growth_stages": n_growth_stages,
        "source_point": source_point,
        "initial_setup": initial_setup,
        "stage_1_points_gdf": stage_1_points_gdf,
        "stage_1_depth_segments_gdf": stage_1_depth_segments_gdf,
        "initial_current_water_zones_gdf": current_water_zones_gdf,
        "stage_outputs": stage_outputs,
        "stage_summary_df": stage_summary_df,
        "final_footprint_geometry": final_footprint_geometry,
        "final_footprint_gdf": final_footprint_gdf,
        "all_depth_polygons_gdf": all_depth_polygons_gdf,
        "final_current_water_zones_gdf": current_stage_water_zones_gdf,
    }
