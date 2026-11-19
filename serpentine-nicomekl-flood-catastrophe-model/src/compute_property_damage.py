# ============================================================
# damageV2.py
# HAZUS depth-damage loss calculation
# ============================================================

import re

import geopandas as gpd
import numpy as np
import pandas as pd


def _first_existing_column(df, candidates, required=True, label="column"):
    """
    Return the first candidate column that exists in df.
    """

    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Could not find {label}. Tried: {', '.join(candidates)}"
        )

    return None


def _parse_hazus_depth_column(column_name):
    """
    Convert HAZUS wide depth columns to depth in feet.

    Examples:
        ft04m  -> -4
        ft00   -> 0
        ft03   -> 3
        ft02_5 -> 2.5
    """

    match = re.fullmatch(r"ft(\d{2})(m)?(?:_(\d+))?", column_name)

    if match is None:
        return None

    whole_feet = int(match.group(1))
    is_negative = match.group(2) == "m"
    decimal_part = match.group(3)

    depth_ft = float(whole_feet)

    if decimal_part is not None:
        depth_ft += float(f"0.{decimal_part}")

    if is_negative:
        depth_ft = -depth_ft

    return depth_ft


def prepare_hazus_depth_damage_curves(
    hazus_curves_df,
    damage_function_id_col="DmgFnId",
    occupancy_col="Occupancy",
):
    """
    Convert the HAZUS wide curve table to a long interpolation-ready table.

    HAZUS damage values are commonly stored as percentages. If values are above
    1, they are converted from percent to ratio.
    """

    if damage_function_id_col not in hazus_curves_df.columns:
        raise ValueError(
            f"hazus_curves_df is missing column: {damage_function_id_col}"
        )

    if occupancy_col not in hazus_curves_df.columns:
        raise ValueError(f"hazus_curves_df is missing column: {occupancy_col}")

    depth_cols = [
        col
        for col in hazus_curves_df.columns
        if _parse_hazus_depth_column(col) is not None
    ]

    if len(depth_cols) == 0:
        raise ValueError("No HAZUS depth columns were found.")

    id_cols = [
        col
        for col in [
            occupancy_col,
            damage_function_id_col,
            "Source",
            "Description",
            "Source_Table",
            "Occupy_Class",
            "Cover_Class",
        ]
        if col in hazus_curves_df.columns
    ]

    curves_long = hazus_curves_df[id_cols + depth_cols].melt(
        id_vars=id_cols,
        value_vars=depth_cols,
        var_name="depth_column",
        value_name="damage_ratio",
    )

    curves_long["depth_ft"] = curves_long["depth_column"].map(
        _parse_hazus_depth_column
    )

    curves_long["damage_ratio"] = pd.to_numeric(
        curves_long["damage_ratio"],
        errors="coerce",
    )

    curves_long = curves_long.dropna(
        subset=[damage_function_id_col, "depth_ft", "damage_ratio"]
    ).copy()

    curves_long[damage_function_id_col] = (
        curves_long[damage_function_id_col].astype(str)
    )

    max_by_curve = curves_long.groupby(damage_function_id_col)[
        "damage_ratio"
    ].transform("max")

    curves_long.loc[max_by_curve > 1.0, "damage_ratio"] = (
        curves_long.loc[max_by_curve > 1.0, "damage_ratio"] / 100.0
    )

    curves_long["damage_ratio"] = curves_long["damage_ratio"].clip(
        lower=0.0,
        upper=1.0,
    )

    curves_long = curves_long.sort_values(
        [damage_function_id_col, "depth_ft"]
    ).reset_index(drop=True)

    return curves_long


def attach_max_flood_depth_to_exposure(
    exposure_gdf,
    depth_polygons_gdf,
    exposure_id_col=None,
    depth_col="max_depth_m",
    crs="EPSG:3005",
):
    """
    Spatially attach the maximum intersecting flood depth to each exposure.

    If an exposure intersects multiple depth polygons, the maximum depth is used.
    This matches the current catastrophe-loss assumption.
    """

    if exposure_gdf is None or exposure_gdf.empty:
        raise ValueError("exposure_gdf is empty.")

    if depth_polygons_gdf is None or depth_polygons_gdf.empty:
        raise ValueError("depth_polygons_gdf is empty.")

    if depth_col not in depth_polygons_gdf.columns:
        raise ValueError(f"depth_polygons_gdf is missing column: {depth_col}")

    exposure_id_col = exposure_id_col or _first_existing_column(
        exposure_gdf,
        ["exposure_id", "PROPERTY_NUMBER", "FOLIO", "PID", "OBJECTID"],
        required=False,
        label="exposure id column",
    )

    exposures = exposure_gdf.to_crs(crs).copy()
    depth_polygons = depth_polygons_gdf.to_crs(crs).copy()

    exposures = exposures.drop(columns=["index_right"], errors="ignore")
    depth_polygons = depth_polygons.drop(columns=["index_right"], errors="ignore")

    exposures["_exposure_row_id"] = np.arange(len(exposures))

    keep_depth_cols = [
        col
        for col in [
            depth_col,
            "depth_stage",
            "depth_polygon_id",
            "segment_water_budget",
            "flood_type",
            "coastal_v2_zone_type",
        ]
        if col in depth_polygons.columns
    ]

    depth_subset = depth_polygons[keep_depth_cols + ["geometry"]].copy()

    joined = gpd.sjoin(
        exposures,
        depth_subset,
        how="left",
        predicate="intersects",
    )

    joined[depth_col] = pd.to_numeric(joined[depth_col], errors="coerce")

    flooded_matches = joined[joined[depth_col].notna()].copy()

    if flooded_matches.empty:
        output = exposures.copy()
        output["flooded"] = False
        output["flood_depth_m"] = 0.0
        output["flood_depth_ft"] = 0.0
        return output.drop(columns=["_exposure_row_id"], errors="ignore")

    idx = (
        flooded_matches
        .sort_values(["_exposure_row_id", depth_col], ascending=[True, False])
        .groupby("_exposure_row_id")
        .head(1)
        .set_index("_exposure_row_id")
    )

    exposures["flooded"] = exposures["_exposure_row_id"].isin(idx.index)
    exposures["flood_depth_m"] = (
        exposures["_exposure_row_id"]
        .map(idx[depth_col])
        .fillna(0.0)
    )
    exposures["flood_depth_ft"] = exposures["flood_depth_m"] * 3.28084

    for col in keep_depth_cols:
        if col == depth_col:
            continue
        exposures[f"matched_{col}"] = exposures["_exposure_row_id"].map(
            idx[col]
        )

    return exposures.drop(columns=["_exposure_row_id"], errors="ignore")


def interpolate_hazus_damage_ratios(
    exposure_with_depth_gdf,
    curves_long_df,
    damage_function_id_col="hazus_a_zone_building_dmg_fn_id",
    curve_damage_function_id_col="DmgFnId",
    depth_ft_col="flood_depth_ft",
):
    """
    Interpolate HAZUS damage ratios for each exposure row.
    """

    if damage_function_id_col not in exposure_with_depth_gdf.columns:
        raise ValueError(
            "exposure_with_depth_gdf is missing column: "
            f"{damage_function_id_col}"
        )

    if curve_damage_function_id_col not in curves_long_df.columns:
        raise ValueError(
            f"curves_long_df is missing column: {curve_damage_function_id_col}"
        )

    if depth_ft_col not in exposure_with_depth_gdf.columns:
        raise ValueError(
            f"exposure_with_depth_gdf is missing column: {depth_ft_col}"
        )

    output = exposure_with_depth_gdf.copy()
    output["_damage_fn_id"] = output[damage_function_id_col].astype(str)

    curve_lookup = {
        str(dmg_fn_id): group.sort_values("depth_ft")
        for dmg_fn_id, group in curves_long_df.groupby(
            curve_damage_function_id_col
        )
    }

    damage_ratios = np.zeros(len(output), dtype=float)
    missing_curve_mask = np.zeros(len(output), dtype=bool)

    for row_position, (_, row) in enumerate(output.iterrows()):

        if not bool(row.get("flooded", False)):
            damage_ratios[row_position] = 0.0
            continue

        curve = curve_lookup.get(str(row["_damage_fn_id"]))

        if curve is None or curve.empty:
            damage_ratios[row_position] = 0.0
            missing_curve_mask[row_position] = True
            continue

        depth_ft = float(row[depth_ft_col])

        damage_ratios[row_position] = np.interp(
            depth_ft,
            curve["depth_ft"].to_numpy(dtype=float),
            curve["damage_ratio"].to_numpy(dtype=float),
            left=float(curve["damage_ratio"].iloc[0]),
            right=float(curve["damage_ratio"].iloc[-1]),
        )

    output["building_damage_ratio"] = np.clip(damage_ratios, 0.0, 1.0)
    output["missing_hazus_curve"] = missing_curve_mask

    return output.drop(columns=["_damage_fn_id"], errors="ignore")


def compute_building_losses(
    exposure_with_damage_ratio_gdf,
    tiv_col=None,
):
    """
    Compute building losses from damage ratio and building TIV.
    """

    tiv_col = tiv_col or _first_existing_column(
        exposure_with_damage_ratio_gdf,
        [
            "building_tiv",
            "exposed_building_tiv",
            "estimated_building_tiv",
            "GROSS_IMPROVEMENTS",
            "Total_Gross_Building_Assessment",
        ],
        required=True,
        label="building TIV column",
    )

    if "building_damage_ratio" not in exposure_with_damage_ratio_gdf.columns:
        raise ValueError("Missing building_damage_ratio column.")

    output = exposure_with_damage_ratio_gdf.copy()
    output[tiv_col] = pd.to_numeric(output[tiv_col], errors="coerce").fillna(0.0)
    output["building_tiv_source_col"] = tiv_col
    output["building_tiv_used"] = output[tiv_col]
    output["building_loss"] = (
        output["building_tiv_used"] * output["building_damage_ratio"]
    )

    return output


def summarize_damage_results(
    damaged_exposure_gdf,
    occupancy_col="hazus_occupancy",
):
    """
    Build overall and occupancy-level loss summaries.
    """

    total_exposures = len(damaged_exposure_gdf)
    flooded_exposures = int(damaged_exposure_gdf["flooded"].sum())

    total_tiv = float(damaged_exposure_gdf["building_tiv_used"].sum())
    flooded_tiv = float(
        damaged_exposure_gdf.loc[
            damaged_exposure_gdf["flooded"],
            "building_tiv_used",
        ].sum()
    )
    total_loss = float(damaged_exposure_gdf["building_loss"].sum())

    summary = {
        "total_exposures": int(total_exposures),
        "flooded_exposures": flooded_exposures,
        "unflooded_exposures": int(total_exposures - flooded_exposures),
        "total_building_tiv": total_tiv,
        "flooded_building_tiv": flooded_tiv,
        "total_building_loss": total_loss,
        "portfolio_loss_ratio": (
            total_loss / total_tiv
            if total_tiv > 0
            else 0.0
        ),
        "flooded_loss_ratio": (
            total_loss / flooded_tiv
            if flooded_tiv > 0
            else 0.0
        ),
        "max_flood_depth_m": float(damaged_exposure_gdf["flood_depth_m"].max()),
        "max_damage_ratio": float(
            damaged_exposure_gdf["building_damage_ratio"].max()
        ),
        "missing_hazus_curve_count": int(
            damaged_exposure_gdf["missing_hazus_curve"].sum()
        ),
    }

    if occupancy_col in damaged_exposure_gdf.columns:
        by_occupancy = (
            damaged_exposure_gdf
            .groupby(occupancy_col, dropna=False)
            .agg(
                exposures=("geometry", "count"),
                flooded_exposures=("flooded", "sum"),
                building_tiv=("building_tiv_used", "sum"),
                building_loss=("building_loss", "sum"),
                mean_depth_m=("flood_depth_m", "mean"),
                max_depth_m=("flood_depth_m", "max"),
                mean_damage_ratio=("building_damage_ratio", "mean"),
                max_damage_ratio=("building_damage_ratio", "max"),
            )
            .reset_index()
        )

        by_occupancy["loss_ratio"] = np.where(
            by_occupancy["building_tiv"] > 0,
            by_occupancy["building_loss"] / by_occupancy["building_tiv"],
            0.0,
        )

        by_occupancy = by_occupancy.sort_values(
            "building_loss",
            ascending=False,
        )
    else:
        by_occupancy = pd.DataFrame()

    return {
        "summary": summary,
        "by_occupancy": by_occupancy,
    }


def compute_damage_v2(
    footprint_result,
    exposure_gdf,
    hazus_curves_df,
    tiv_col=None,
    exposure_id_col=None,
    hazus_damage_function_col="hazus_a_zone_building_dmg_fn_id",
    occupancy_col="hazus_occupancy",
    depth_col="max_depth_m",
    crs="EPSG:3005",
):
    """
    Compute property-level flood damage from a footprint result.

    Main modelling choices:
        - use all depth polygons from the footprint result
        - assign each exposure the maximum intersecting depth
        - use the A-zone building HAZUS damage function by default
        - calculate building loss against the selected building TIV column
    """

    if "all_depth_polygons_gdf" not in footprint_result:
        raise ValueError("footprint_result is missing all_depth_polygons_gdf.")

    depth_polygons_gdf = footprint_result["all_depth_polygons_gdf"]

    curves_long_df = prepare_hazus_depth_damage_curves(hazus_curves_df)

    exposure_with_depth_gdf = attach_max_flood_depth_to_exposure(
        exposure_gdf=exposure_gdf,
        depth_polygons_gdf=depth_polygons_gdf,
        exposure_id_col=exposure_id_col,
        depth_col=depth_col,
        crs=crs,
    )

    exposure_with_damage_ratio_gdf = interpolate_hazus_damage_ratios(
        exposure_with_depth_gdf=exposure_with_depth_gdf,
        curves_long_df=curves_long_df,
        damage_function_id_col=hazus_damage_function_col,
        curve_damage_function_id_col="DmgFnId",
        depth_ft_col="flood_depth_ft",
    )

    damaged_exposure_gdf = compute_building_losses(
        exposure_with_damage_ratio_gdf=exposure_with_damage_ratio_gdf,
        tiv_col=tiv_col,
    )

    summaries = summarize_damage_results(
        damaged_exposure_gdf=damaged_exposure_gdf,
        occupancy_col=occupancy_col,
    )

    return {
        "damaged_exposure_gdf": damaged_exposure_gdf,
        "damage_summary": summaries["summary"],
        "damage_by_occupancy_df": summaries["by_occupancy"],
        "hazus_curves_long_df": curves_long_df,
        "depth_polygons_gdf": depth_polygons_gdf,
    }
