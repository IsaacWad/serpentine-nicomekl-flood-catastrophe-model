# ============================================================
# flood_footprint/riverine_interior_drainage.py
# Riverine and interior-drainage flood footprint wrappers
# ============================================================

import geopandas as gpd
import pandas as pd

try:
    from .core import (
        make_geometry_gdf,
        create_initial_flood_setup,
        create_stage_1_scored_points,
        create_stage_1_depth_segments,
        initialize_current_water_zones_from_stage_1,
        run_one_flood_expansion_stage,
    )
except ImportError:
    from core import (
        make_geometry_gdf,
        create_initial_flood_setup,
        create_stage_1_scored_points,
        create_stage_1_depth_segments,
        initialize_current_water_zones_from_stage_1,
        run_one_flood_expansion_stage,
    )


def _build_generic_line_source_flood_footprint(
    selected_segment,
    dem_path,
    expansion_boundary,
    flood_type,
    severity,
    event_water_budget,
    severity_max_depth_m,
    n_growth_stages=8,
    initial_buffer_radius_m=225,
    n_stage_1_reference_points=16,
    elevation_scale_m=1.0,
    depth_scale_factor=50_000,
    n_candidate_points=128,
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
    point_influence_radius_m=50,
    connection_buffer_m=20,
    n_radial_wedges=96,
    min_outward_gain_m=5,
    min_water_budget=0.001,
    min_depth_m=0.05,
    min_polygon_area_sqm=1.0,
    random_seed=42,
    random_low=0.85,
    random_high=1.15,
    crs="EPSG:3005",
):
    """
    Shared builder for riverine and interior-drainage line-source events.

    Both event types currently use the same generic V2 footprint mechanics.
    The flood_type argument only labels outputs so downstream reporting can
    keep the event families separate.
    """

    if flood_type not in ["riverine", "interior_drainage"]:
        raise ValueError(
            "flood_type must be 'riverine' or 'interior_drainage'."
        )

    if expansion_boundary is None or expansion_boundary.is_empty:
        raise ValueError("expansion_boundary is empty.")

    if n_growth_stages < 1:
        raise ValueError("n_growth_stages must be at least 1.")

    initial_setup = create_initial_flood_setup(
        selected_segment=selected_segment,
        buffer_radius_m=initial_buffer_radius_m,
        expansion_boundary=expansion_boundary,
        n_reference_points=n_stage_1_reference_points,
        crs=crs,
    )

    source_point = initial_setup["source_point"]
    base_buffer_geometry = initial_setup["base_buffer_geometry"]

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

    stage_1_candidate_points_gdf = stage_1_points_gdf.copy()
    stage_1_candidate_points_gdf["point_position"] = (
        "initial_wedge_outer_boundary_midpoint"
    )

    stage_1_depth_segments_gdf = create_stage_1_depth_segments(
        source_point=source_point,
        base_buffer_geometry=base_buffer_geometry,
        stage_1_points_gdf=stage_1_candidate_points_gdf,
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
    stage_1_depth_segments_gdf["flood_type"] = flood_type

    current_water_zones_gdf = initialize_current_water_zones_from_stage_1(
        stage_1_depth_segments_gdf=stage_1_depth_segments_gdf,
        min_water_budget=min_water_budget,
        min_depth_m=min_depth_m,
        crs=crs,
    )

    initial_stage_1_summary = {
        "flood_type": flood_type,
        "event_severity": severity,
        "stage_id": 1,
        "stage_1_candidate_points": int(len(stage_1_candidate_points_gdf)),
        "stage_1_depth_polygons": int(len(stage_1_depth_segments_gdf)),
        "stage_1_active_water_zones": int(len(current_water_zones_gdf)),
        "event_water_budget": float(event_water_budget),
        "stage_1_allocated_water_budget": float(
            stage_1_depth_segments_gdf["segment_water_budget"].sum()
        ),
        "stage_1_water_balance_error": float(
            stage_1_depth_segments_gdf["segment_water_budget"].sum()
            - event_water_budget
        ),
        "largest_stage_1_share": float(
            stage_1_depth_segments_gdf["stage_1_score_share"].max()
        ),
        "smallest_stage_1_share": float(
            stage_1_depth_segments_gdf["stage_1_score_share"].min()
        ),
    }

    stage_outputs = {}
    stage_summary_rows = []
    all_depth_polygon_layers = [stage_1_depth_segments_gdf.copy()]

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

        next_stage_number = stage_id + 1
        next_stage_depth_polygons_gdf = stage_output[
            "next_stage_depth_polygons_gdf"
        ].copy()
        stage_depth_polygon_id_col = (
            f"stage_{next_stage_number}_depth_polygon_id"
        )

        next_stage_depth_polygons_gdf["depth_stage"] = next_stage_number
        next_stage_depth_polygons_gdf["depth_polygon_id"] = (
            next_stage_depth_polygons_gdf[stage_depth_polygon_id_col]
            .astype(str)
            if stage_depth_polygon_id_col in next_stage_depth_polygons_gdf
            else (
                "stage_"
                + str(next_stage_number)
                + "_depth_polygon_"
                + next_stage_depth_polygons_gdf.index.astype(str)
            )
        )
        next_stage_depth_polygons_gdf["event_severity"] = severity
        next_stage_depth_polygons_gdf["flood_type"] = flood_type

        all_depth_polygon_layers.append(next_stage_depth_polygons_gdf)

        expansion_lobes_gdf = stage_output["expansion_lobes_gdf"]
        lobe_fill_summary = stage_output.get("lobe_fill_summary", {})

        stage_summary_rows.append(
            {
                "stage_id": int(stage_id),
                "stage_next_id": int(next_stage_number),
                "flood_type": flood_type,
                "event_severity": severity,
                "candidate_points": int(
                    len(stage_output["candidate_points_gdf"])
                ),
                "active_points": int(len(stage_output["active_points_gdf"])),
                "expanded_points": int(
                    len(stage_output["expanded_points_gdf"])
                ),
                "expansion_lobes": int(len(expansion_lobes_gdf)),
                "depth_polygons": int(len(next_stage_depth_polygons_gdf)),
                "current_area_sqkm": float(
                    current_footprint_geometry.area / 1_000_000
                ),
                "next_area_sqkm": float(
                    stage_output["next_footprint_geometry"].area / 1_000_000
                ),
                "new_area_sqkm": float(
                    stage_output["new_area_geometry"].area / 1_000_000
                ),
                "assigned_lobe_water": float(
                    next_stage_depth_polygons_gdf[
                        "segment_water_budget"
                    ].sum()
                ),
                "original_lobe_water": float(
                    expansion_lobes_gdf[
                        "total_allocated_water_budget"
                    ].sum()
                ),
                "updated_water_zones": int(
                    len(stage_output["updated_current_water_zones_gdf"])
                ),
                "updated_water_total": float(
                    stage_output["updated_current_water_zones_gdf"][
                        "current_water_budget"
                    ].sum()
                ),
                "pre_lobe_fill_depth_area_sqm": float(
                    lobe_fill_summary.get(
                        "pre_lobe_fill_depth_area_sqm",
                        0.0,
                    )
                ),
                "post_lobe_fill_depth_area_sqm": float(
                    lobe_fill_summary.get(
                        "post_lobe_fill_depth_area_sqm",
                        0.0,
                    )
                ),
                "lobe_fill_added_area_sqm": float(
                    lobe_fill_summary.get(
                        "lobe_fill_added_area_sqm",
                        0.0,
                    )
                ),
                "uncovered_new_area_after_lobe_fill_sqm": float(
                    lobe_fill_summary.get(
                        "uncovered_new_area_after_lobe_fill_sqm",
                        0.0,
                    )
                ),
                "uncovered_new_area_after_lobe_fill_share": float(
                    lobe_fill_summary.get(
                        "uncovered_new_area_after_lobe_fill_share",
                        0.0,
                    )
                ),
                "uncovered_new_area_assigned_to_polygons_sqm": float(
                    lobe_fill_summary.get(
                        "uncovered_new_area_assigned_to_polygons_sqm",
                        0.0,
                    )
                ),
                "remaining_uncovered_new_area_sqm": float(
                    lobe_fill_summary.get(
                        "remaining_uncovered_new_area_sqm",
                        0.0,
                    )
                ),
                "remaining_uncovered_new_area_share": float(
                    lobe_fill_summary.get(
                        "remaining_uncovered_new_area_share",
                        0.0,
                    )
                ),
            }
        )

        current_footprint_geometry = stage_output["next_footprint_geometry"]
        current_stage_water_zones_gdf = stage_output[
            "updated_current_water_zones_gdf"
        ]

        if current_stage_water_zones_gdf.empty:
            break

    all_depth_polygons_gdf = gpd.GeoDataFrame(
        pd.concat(
            all_depth_polygon_layers,
            ignore_index=True,
            sort=False,
        ),
        geometry="geometry",
        crs=crs,
    )

    front_cols = [
        "flood_type",
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
        col for col in front_cols if col in all_depth_polygons_gdf.columns
    ]
    remaining_cols = [
        col for col in all_depth_polygons_gdf.columns
        if col not in existing_front_cols
    ]
    all_depth_polygons_gdf = all_depth_polygons_gdf[
        existing_front_cols + remaining_cols
    ]

    stage_summary_df = pd.DataFrame(stage_summary_rows)

    final_footprint_geometry = current_footprint_geometry
    final_footprint_gdf = make_geometry_gdf(
        rows=[
            {
                "flood_type": flood_type,
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
        "flood_type": flood_type,
        "event_severity": severity,
        "event_water_budget": float(event_water_budget),
        "severity_max_depth_m": float(severity_max_depth_m),
        "n_growth_stages_requested": int(n_growth_stages),
        "n_growth_stages_completed": int(len(stage_outputs)),
        "total_depth_polygons": int(len(all_depth_polygons_gdf)),
        "final_footprint_area_sqkm": float(
            final_footprint_geometry.area / 1_000_000
        ),
        "final_water_zones": int(len(current_stage_water_zones_gdf)),
        "final_water_total": float(
            current_stage_water_zones_gdf["current_water_budget"].sum()
            if not current_stage_water_zones_gdf.empty
            else 0.0
        ),
        "max_depth_m": float(all_depth_polygons_gdf["max_depth_m"].max()),
        "mean_depth_m": float(all_depth_polygons_gdf["max_depth_m"].mean()),
    }

    return {
        "flood_type": flood_type,
        "severity": severity,
        "event_water_budget": event_water_budget,
        "severity_max_depth_m": severity_max_depth_m,
        "n_growth_stages": n_growth_stages,
        "source_point": source_point,
        "initial_setup": initial_setup,
        "stage_1_points_gdf": stage_1_points_gdf,
        "stage_1_candidate_points_gdf": stage_1_candidate_points_gdf,
        "stage_1_depth_segments_gdf": stage_1_depth_segments_gdf,
        "initial_current_water_zones_gdf": current_water_zones_gdf,
        "initial_stage_1_summary": initial_stage_1_summary,
        "stage_outputs": stage_outputs,
        "stage_summary_df": stage_summary_df,
        "final_summary": final_summary,
        "final_footprint_geometry": final_footprint_geometry,
        "final_footprint_gdf": final_footprint_gdf,
        "all_depth_polygons_gdf": all_depth_polygons_gdf,
        "final_current_water_zones_gdf": current_stage_water_zones_gdf,
    }


def build_riverine_flood_footprint(
    selected_segment,
    dem_path,
    expansion_boundary,
    severity,
    event_water_budget,
    severity_max_depth_m,
    **kwargs,
):
    """
    Build a riverine flood footprint using the generic line-source V2 flow.
    """

    return _build_generic_line_source_flood_footprint(
        selected_segment=selected_segment,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        flood_type="riverine",
        severity=severity,
        event_water_budget=event_water_budget,
        severity_max_depth_m=severity_max_depth_m,
        **kwargs,
    )


def build_interior_drainage_flood_footprint(
    selected_segment,
    dem_path,
    expansion_boundary,
    severity,
    event_water_budget,
    severity_max_depth_m,
    **kwargs,
):
    """
    Build an interior-drainage flood footprint using the same line-source V2 flow.
    """

    return _build_generic_line_source_flood_footprint(
        selected_segment=selected_segment,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        flood_type="interior_drainage",
        severity=severity,
        event_water_budget=event_water_budget,
        severity_max_depth_m=severity_max_depth_m,
        **kwargs,
    )
