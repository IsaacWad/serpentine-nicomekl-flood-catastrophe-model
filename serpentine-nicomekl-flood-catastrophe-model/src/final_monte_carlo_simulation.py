# ============================================================
# monte_carlo_simulation.py
#
# Purpose:
# Run flood catastrophe Monte Carlo simulations using the V2 footprint
# builders and HAZUS property damage pipeline.
# ============================================================

import copy

import geopandas as gpd
import numpy as np
import pandas as pd

import compute_event_lossV3
import compute_property_damage
import event_severity_distribution
import select_segment

try:
    from flood_footprint import coastalV2
    from flood_footprint import riverine_interior_drainage
except ImportError:
    import coastalV2
    import riverine_interior_drainage


DEFAULT_SEVERITY_PARAMETERS = {
    "minor": {
        "event_water_budget": 2.0,
        "severity_max_depth_m": 1.0,
        "n_growth_stages": 3,
    },
    "moderate": {
        "event_water_budget": 4.0,
        "severity_max_depth_m": 2.0,
        "n_growth_stages": 6,
    },
    "severe": {
        "event_water_budget": 8.0,
        "severity_max_depth_m": 3.0,
        "n_growth_stages": 9,
    },
    "extreme": {
        "event_water_budget": 12.0,
        "severity_max_depth_m": 4.0,
        "n_growth_stages": 12,
    },
}


DEFAULT_FOOTPRINT_PARAMETERS = {
    "coastal": {
        "initial_buffer_radius_m": 250,
        "n_stage_1_reference_points": 16,
        "initial_storage_overlap_threshold": 0.99,
        "stage_storage_overlap_threshold": 0.50,
        "elevation_scale_m": 1.0,
        "depth_scale_factor": 50_000,
        "n_candidate_points_per_zone": 8,
        "outer_distance_share": 0.97,
        "boundary_sample_spacing_m": 5,
        "boundary_match_tolerance_m": 0.5,
        "n_active_points": 16,
        "parent_penalty_strength": 0.35,
        "base_distance_m": 225,
        "min_distance_m": 40,
        "max_distance_m": 425,
        "score_power": 0.50,
        "depth_power": 0.75,
        "water_power": 0.35,
        "scan_left_degrees": 11.25,
        "scan_right_degrees": 11.25,
        "n_scan_angles": 9,
        "outflow_share": 0.75,
        "point_influence_radius_m": 75,
        "connection_buffer_m": 35,
        "n_radial_wedges": 96,
        "min_outward_gain_m": 5,
        "min_polygon_area_sqm": 1_000,
        "min_water_budget": 0.001,
        "min_depth_m": 0.05,
        "random_low": 0.85,
        "random_high": 1.15,
    },
    "riverine": {
        "initial_buffer_radius_m": 225,
        "n_stage_1_reference_points": 16,
        "n_candidate_points": 128,
        "n_active_points": 16,
        "base_distance_m": 225,
        "min_distance_m": 40,
        "max_distance_m": 425,
        "outflow_share": 0.75,
        "point_influence_radius_m": 50,
        "connection_buffer_m": 20,
        "n_radial_wedges": 96,
        "min_outward_gain_m": 5,
        "depth_scale_factor": 50_000,
        "min_polygon_area_sqm": 1_000,
        "min_water_budget": 0.001,
        "min_depth_m": 0.05,
    },
    "interior_drainage": {
        "initial_buffer_radius_m": 125,
        "n_stage_1_reference_points": 16,
        "n_candidate_points": 128,
        "n_active_points": 16,
        "base_distance_m": 225,
        "min_distance_m": 40,
        "max_distance_m": 425,
        "outflow_share": 0.75,
        "point_influence_radius_m": 50,
        "connection_buffer_m": 20,
        "n_radial_wedges": 96,
        "min_outward_gain_m": 5,
        "depth_scale_factor": 50_000,
        "min_polygon_area_sqm": 1_000,
        "min_water_budget": 0.001,
        "min_depth_m": 0.05,
    },
}


EVENT_SOURCE_ALIASES = {
    "coast": "coastal",
    "coastal": "coastal",
    "river": "riverine",
    "riverine": "riverine",
    "interior": "interior_drainage",
    "interior_drainage": "interior_drainage",
    "interior drainage": "interior_drainage",
}


def _normalize_event_source(event_source):
    source = str(event_source).strip().lower()

    if source not in EVENT_SOURCE_ALIASES:
        raise ValueError(
            "Unsupported event_source. "
            f"Received {event_source!r}; expected one of "
            f"{sorted(EVENT_SOURCE_ALIASES)}."
        )

    return EVENT_SOURCE_ALIASES[source]


def _normalize_severity(severity):
    severity = str(severity).strip().lower()

    if severity not in DEFAULT_SEVERITY_PARAMETERS:
        raise ValueError(
            "Unsupported severity. "
            f"Received {severity!r}; expected one of "
            f"{sorted(DEFAULT_SEVERITY_PARAMETERS)}."
        )

    return severity


def _segment_value(selected_segment, column, default=None):
    if isinstance(selected_segment, gpd.GeoDataFrame):
        if selected_segment.empty:
            return default
        selected_segment = selected_segment.iloc[0]

    if column == "geometry":
        return selected_segment.geometry

    if hasattr(selected_segment, "index") and column in selected_segment.index:
        return selected_segment[column]

    return default


def _merge_parameter_overrides(event_source, severity, footprint_parameter_overrides):
    params = copy.deepcopy(DEFAULT_FOOTPRINT_PARAMETERS[event_source])
    params.update(DEFAULT_SEVERITY_PARAMETERS[severity])

    if footprint_parameter_overrides is None:
        return params

    if event_source in footprint_parameter_overrides:
        params.update(footprint_parameter_overrides[event_source])

    if severity in footprint_parameter_overrides:
        params.update(footprint_parameter_overrides[severity])

    shared = {
        key: value
        for key, value in footprint_parameter_overrides.items()
        if key not in DEFAULT_FOOTPRINT_PARAMETERS
        and key not in DEFAULT_SEVERITY_PARAMETERS
    }
    params.update(shared)

    return params


def _footprint_summary_metrics(footprint_result):
    depth_polygons = footprint_result.get("all_depth_polygons_gdf")

    if depth_polygons is None or depth_polygons.empty:
        return {
            "footprint_area_sqkm": 0.0,
            "depth_polygon_count": 0,
            "max_footprint_depth_m": 0.0,
        }

    total_geom = depth_polygons.geometry.union_all()

    return {
        "footprint_area_sqkm": float(total_geom.area / 1_000_000),
        "depth_polygon_count": int(len(depth_polygons)),
        "max_footprint_depth_m": float(depth_polygons["max_depth_m"].max())
        if "max_depth_m" in depth_polygons.columns
        else 0.0,
    }


def _extract_stage_summary(footprint_result):
    for key in [
        "stage_summary_df",
        "stage_summaries_df",
        "stage_summary_gdf",
        "stage_summaries",
    ]:
        if key in footprint_result:
            value = footprint_result[key]
            if isinstance(value, list):
                return pd.DataFrame(value)
            return value

    return None


def draw_event_severity(selected_segment, random_seed=None):
    severity_distribution = (
        event_severity_distribution.build_event_severity_distribution(
            segment_score=_segment_value(selected_segment, "segment_score"),
            event_source=_segment_value(selected_segment, "event_source"),
        )
    )

    return event_severity_distribution.draw_event_severity(
        severity_distribution=severity_distribution,
        random_seed=random_seed,
    )


def build_footprint_for_selected_event(
    selected_segment,
    event_severity,
    dem_path,
    expansion_boundary,
    coastal_storage_gdf=None,
    coastal_storage_side_gdf=None,
    footprint_parameter_overrides=None,
    crs="EPSG:3005",
    random_seed=None,
):
    """
    Dispatch one selected event to the correct V2 footprint builder.
    """

    event_source = _normalize_event_source(
        _segment_value(selected_segment, "event_source")
    )
    event_severity = _normalize_severity(event_severity)

    params = _merge_parameter_overrides(
        event_source=event_source,
        severity=event_severity,
        footprint_parameter_overrides=footprint_parameter_overrides,
    )
    params["random_seed"] = random_seed
    params["crs"] = crs

    selected_geometry = _segment_value(selected_segment, "geometry")

    if event_source == "coastal":
        if coastal_storage_gdf is None and coastal_storage_side_gdf is None:
            raise ValueError(
                "Coastal simulations require coastal_storage_gdf or "
                "coastal_storage_side_gdf."
            )

        return coastalV2.build_coastal_v2_flood_footprint(
            selected_segment=selected_geometry,
            dem_path=dem_path,
            expansion_boundary=expansion_boundary,
            coastal_storage_gdf=coastal_storage_gdf,
            coastal_storage_side_gdf=coastal_storage_side_gdf,
            severity=event_severity,
            **params,
        )

    if event_source == "riverine":
        return riverine_interior_drainage.build_riverine_flood_footprint(
            selected_segment=selected_geometry,
            dem_path=dem_path,
            expansion_boundary=expansion_boundary,
            severity=event_severity,
            **params,
        )

    return riverine_interior_drainage.build_interior_drainage_flood_footprint(
        selected_segment=selected_geometry,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        severity=event_severity,
        **params,
    )


def simulate_one_flood_event(
    sim_id,
    scored_segments_gdf,
    exposure_gdf,
    hazus_curves_df,
    dem_path,
    expansion_boundary,
    coastal_storage_gdf=None,
    coastal_storage_side_gdf=None,
    event_source_weights=select_segment.DEFAULT_EVENT_SOURCE_WEIGHTS,
    selection_slope=0.15,
    tiv_col="building_tiv",
    exposure_id_col="exposure_id",
    hazus_damage_function_col="hazus_a_zone_building_dmg_fn_id",
    occupancy_col="hazus_occupancy",
    footprint_parameter_overrides=None,
    crs="EPSG:3005",
    random_seed=None,
):
    """
    Simulate one event from source selection through event loss.
    """

    rng = np.random.default_rng(random_seed)
    selection_seed = int(rng.integers(0, 1_000_000_000))
    severity_seed = int(rng.integers(0, 1_000_000_000))
    footprint_seed = int(rng.integers(0, 1_000_000_000))

    selected_segment = select_segment.select_event_segment(
        scored_segments_gdf=scored_segments_gdf,
        event_source_weights=event_source_weights,
        selection_slope=selection_slope,
        random_seed=selection_seed,
    )

    event_source = _normalize_event_source(
        _segment_value(selected_segment, "event_source")
    )
    event_severity = _normalize_severity(
        draw_event_severity(
            selected_segment=selected_segment,
            random_seed=severity_seed,
        )
    )

    source_segment_id = _segment_value(
        selected_segment,
        "segment_id",
        default=_segment_value(selected_segment, "event_segment_id", sim_id),
    )
    event_id = f"sim_{sim_id:06d}_{source_segment_id}_{event_severity}"

    footprint_result = build_footprint_for_selected_event(
        selected_segment=selected_segment,
        event_severity=event_severity,
        dem_path=dem_path,
        expansion_boundary=expansion_boundary,
        coastal_storage_gdf=coastal_storage_gdf,
        coastal_storage_side_gdf=coastal_storage_side_gdf,
        footprint_parameter_overrides=footprint_parameter_overrides,
        crs=crs,
        random_seed=footprint_seed,
    )

    damage_result = compute_property_damage.compute_damage_v2(
        footprint_result=footprint_result,
        exposure_gdf=exposure_gdf,
        hazus_curves_df=hazus_curves_df,
        tiv_col=tiv_col,
        exposure_id_col=exposure_id_col,
        hazus_damage_function_col=hazus_damage_function_col,
        occupancy_col=occupancy_col,
        crs=crs,
    )

    event_loss_result = compute_event_lossV3.compute_event_loss(
        properties_with_damage_gdf=damage_result["damaged_exposure_gdf"],
        event_id=event_id,
        source_segment_id=source_segment_id,
        event_source=event_source,
        severity=event_severity,
        tiv_col="building_tiv_used",
        damage_ratio_col="building_damage_ratio",
        loss_col="building_loss",
        crs=crs,
    )

    properties_with_loss_gdf = event_loss_result["properties_with_loss_gdf"]
    event_summary = dict(event_loss_result["event_summary"])
    event_summary.update(
        {
            "simulation_status": "succeeded",
            "error_type": None,
            "error_message": None,
            "sim_id": sim_id,
            "event_id": event_id,
            "source_segment_id": source_segment_id,
            "event_source": event_source,
            "event_severity": event_severity,
            "segment_score": _segment_value(selected_segment, "segment_score"),
            "segment_risk_class": _segment_value(
                selected_segment,
                "segment_risk_class",
            ),
            "selection_slope": selection_slope,
            "random_seed": random_seed,
            "selection_seed": selection_seed,
            "severity_seed": severity_seed,
            "footprint_seed": footprint_seed,
            "affected_property_count": event_summary["flooded_properties"],
            "affected_tiv": event_summary["flooded_tiv"],
            "event_loss": event_summary["total_gross_loss"],
        }
    )
    event_summary.update(_footprint_summary_metrics(footprint_result))

    damage_summary = damage_result.get("damage_summary", {})
    for key, value in damage_summary.items():
        event_summary[f"damage_{key}"] = value

    stage_summary = _extract_stage_summary(footprint_result)
    if isinstance(stage_summary, pd.DataFrame):
        event_summary["growth_stage_rows"] = int(len(stage_summary))

    affected_properties_gdf = properties_with_loss_gdf[
        (properties_with_loss_gdf.get("flooded", False))
        | (properties_with_loss_gdf["building_loss"] > 0)
    ].copy()
    affected_properties_gdf["sim_id"] = sim_id
    affected_properties_gdf["event_id"] = event_id
    affected_properties_gdf["source_segment_id"] = source_segment_id
    affected_properties_gdf["event_source"] = event_source
    affected_properties_gdf["event_severity"] = event_severity

    return {
        "event_summary": event_summary,
        "affected_properties_gdf": affected_properties_gdf,
        "footprint_result": footprint_result,
        "damage_result": damage_result,
        "selected_segment": selected_segment,
    }


def run_monte_carlo_simulation(
    n_sims,
    scored_segments_gdf,
    exposure_gdf,
    hazus_curves_df,
    dem_path,
    expansion_boundary,
    coastal_storage_gdf=None,
    coastal_storage_side_gdf=None,
    event_source_weights=select_segment.DEFAULT_EVENT_SOURCE_WEIGHTS,
    selection_slope=0.15,
    tiv_col="building_tiv",
    exposure_id_col="exposure_id",
    hazus_damage_function_col="hazus_a_zone_building_dmg_fn_id",
    occupancy_col="hazus_occupancy",
    footprint_parameter_overrides=None,
    crs="EPSG:3005",
    random_seed=None,
    print_progress=True,
    continue_on_error=True,
):
    """
    Run many simulated flood events.

    Returns
    -------
    tuple
        event_results_df, affected_properties_all_gdf, footprint_results,
        damage_results
    """

    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")

    if not isinstance(scored_segments_gdf, gpd.GeoDataFrame):
        raise TypeError("scored_segments_gdf must be a GeoDataFrame.")

    if not isinstance(exposure_gdf, gpd.GeoDataFrame):
        raise TypeError("exposure_gdf must be a GeoDataFrame.")

    rng = np.random.default_rng(random_seed)

    event_summary_rows = []
    affected_properties_list = []
    footprint_results = []
    damage_results = []

    for sim_id in range(n_sims):
        sim_seed = int(rng.integers(0, 1_000_000_000))

        try:
            result = simulate_one_flood_event(
                sim_id=sim_id,
                scored_segments_gdf=scored_segments_gdf,
                exposure_gdf=exposure_gdf,
                hazus_curves_df=hazus_curves_df,
                dem_path=dem_path,
                expansion_boundary=expansion_boundary,
                coastal_storage_gdf=coastal_storage_gdf,
                coastal_storage_side_gdf=coastal_storage_side_gdf,
                event_source_weights=event_source_weights,
                selection_slope=selection_slope,
                tiv_col=tiv_col,
                exposure_id_col=exposure_id_col,
                hazus_damage_function_col=hazus_damage_function_col,
                occupancy_col=occupancy_col,
                footprint_parameter_overrides=footprint_parameter_overrides,
                crs=crs,
                random_seed=sim_seed,
            )
        except Exception as exc:
            if not continue_on_error:
                raise

            failed_summary = {
                "simulation_status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "sim_id": sim_id,
                "event_id": f"sim_{sim_id:06d}_failed",
                "source_segment_id": None,
                "event_source": None,
                "event_severity": None,
                "segment_score": None,
                "segment_risk_class": None,
                "selection_slope": selection_slope,
                "random_seed": sim_seed,
                "selection_seed": None,
                "severity_seed": None,
                "footprint_seed": None,
                "total_properties": len(exposure_gdf),
                "flooded_properties": 0,
                "properties_with_loss": 0,
                "total_tiv": 0.0,
                "flooded_tiv": 0.0,
                "total_gross_loss": 0.0,
                "affected_property_count": 0,
                "affected_tiv": 0.0,
                "event_loss": 0.0,
                "footprint_area_sqkm": 0.0,
                "depth_polygon_count": 0,
                "max_footprint_depth_m": 0.0,
            }

            event_summary_rows.append(failed_summary)
            footprint_results.append(None)
            damage_results.append(None)

            if print_progress:
                print(
                    f"sim_id={sim_id} | failed | "
                    f"{type(exc).__name__}: {exc}"
                )

            continue

        event_summary = result["event_summary"]
        event_summary_rows.append(event_summary)
        footprint_results.append(result["footprint_result"])
        damage_results.append(result["damage_result"])

        if not result["affected_properties_gdf"].empty:
            affected_properties_list.append(result["affected_properties_gdf"])

        if print_progress:
            print(
                f"sim_id={sim_id} | "
                f"event_source={event_summary['event_source']} | "
                f"severity={event_summary['event_severity']} | "
                f"flooded_properties={event_summary['flooded_properties']} | "
                f"properties_with_loss={event_summary['properties_with_loss']} | "
                f"event_loss={event_summary['total_gross_loss']:,.0f}"
            )

    event_results_df = pd.DataFrame(event_summary_rows)

    if len(affected_properties_list) > 0:
        affected_properties_all_gdf = gpd.GeoDataFrame(
            pd.concat(affected_properties_list, ignore_index=True),
            geometry="geometry",
            crs=exposure_gdf.crs,
        )
    else:
        affected_properties_all_gdf = gpd.GeoDataFrame(
            columns=list(exposure_gdf.columns)
            + [
                "flooded",
                "flood_depth_m",
                "building_damage_ratio",
                "building_loss",
                "sim_id",
                "event_id",
                "source_segment_id",
                "event_source",
                "event_severity",
            ],
            geometry="geometry",
            crs=exposure_gdf.crs,
        )

    return (
        event_results_df,
        affected_properties_all_gdf,
        footprint_results,
        damage_results,
    )
