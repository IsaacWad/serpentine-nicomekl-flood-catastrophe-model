# ============================================================
# compute_event_loss.py
#
# Purpose:
# Summarize property-level and event-level flood losses.
#
# This module is intentionally small. It is useful when properties already
# have a damage ratio attached, such as the output from compute_property_damage.
# ============================================================

import geopandas as gpd
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


def compute_property_event_losses(
    properties_with_damage_gdf,
    tiv_col=None,
    damage_ratio_col=None,
    loss_col="building_loss",
    crs="EPSG:3005",
):
    """
    Compute property-level gross event losses.

    Loss is applied to building/improvement TIV, not land value:

        gross_loss = building_tiv * damage_ratio

    Defaults match compute_property_damage.py output.
    """

    if not isinstance(properties_with_damage_gdf, gpd.GeoDataFrame):
        raise TypeError("properties_with_damage_gdf must be a GeoDataFrame.")

    if properties_with_damage_gdf.empty:
        raise ValueError("properties_with_damage_gdf is empty.")

    tiv_col = tiv_col or _first_existing_column(
        properties_with_damage_gdf,
        [
            "building_tiv_used",
            "building_tiv",
            "exposed_building_tiv",
            "estimated_building_tiv",
            "GROSS_IMPROVEMENTS",
            "Total_Gross_Building_Assessment",
            "total_tiv",
        ],
        label="building TIV column",
    )

    damage_ratio_col = damage_ratio_col or _first_existing_column(
        properties_with_damage_gdf,
        [
            "building_damage_ratio",
            "damage_ratio",
        ],
        label="damage ratio column",
    )

    output_gdf = properties_with_damage_gdf.copy().to_crs(crs)

    output_gdf[tiv_col] = pd.to_numeric(
        output_gdf[tiv_col],
        errors="coerce",
    ).fillna(0.0)

    output_gdf[damage_ratio_col] = pd.to_numeric(
        output_gdf[damage_ratio_col],
        errors="coerce",
    ).fillna(0.0)

    output_gdf["event_tiv_used"] = output_gdf[tiv_col]
    output_gdf["event_tiv_source_col"] = tiv_col
    output_gdf["event_damage_ratio_used"] = output_gdf[
        damage_ratio_col
    ].clip(
        lower=0.0,
        upper=1.0,
    )
    output_gdf["event_damage_ratio_source_col"] = damage_ratio_col

    output_gdf[loss_col] = (
        output_gdf["event_tiv_used"]
        * output_gdf["event_damage_ratio_used"]
    ).clip(lower=0.0)

    if loss_col != "building_loss":
        output_gdf["building_loss"] = output_gdf[loss_col]

    return gpd.GeoDataFrame(
        output_gdf,
        geometry="geometry",
        crs=crs,
    )


def summarize_event_loss(
    properties_with_loss_gdf,
    event_id=None,
    source_segment_id=None,
    event_source=None,
    severity=None,
    tiv_col=None,
    damage_ratio_col=None,
    loss_col="building_loss",
    flooded_col=None,
):
    """
    Summarize one simulated event loss.
    """

    if not isinstance(properties_with_loss_gdf, gpd.GeoDataFrame):
        raise TypeError("properties_with_loss_gdf must be a GeoDataFrame.")

    if properties_with_loss_gdf.empty:
        raise ValueError("properties_with_loss_gdf is empty.")

    tiv_col = tiv_col or _first_existing_column(
        properties_with_loss_gdf,
        ["event_tiv_used", "building_tiv_used", "building_tiv", "total_tiv"],
        label="building TIV column",
    )

    damage_ratio_col = damage_ratio_col or _first_existing_column(
        properties_with_loss_gdf,
        [
            "event_damage_ratio_used",
            "building_damage_ratio",
            "damage_ratio",
        ],
        label="damage ratio column",
    )

    if loss_col not in properties_with_loss_gdf.columns:
        raise ValueError(f"Missing loss column: {loss_col}")

    flooded_col = flooded_col or _first_existing_column(
        properties_with_loss_gdf,
        ["flooded", "is_flooded"],
        required=False,
        label="flooded indicator column",
    )

    gdf = properties_with_loss_gdf.copy()

    if flooded_col is None:
        flooded_col = "is_flooded"
        gdf[flooded_col] = gdf[loss_col] > 0

    gdf[tiv_col] = pd.to_numeric(gdf[tiv_col], errors="coerce").fillna(0.0)
    gdf[damage_ratio_col] = pd.to_numeric(
        gdf[damage_ratio_col],
        errors="coerce",
    ).fillna(0.0)
    gdf[loss_col] = pd.to_numeric(gdf[loss_col], errors="coerce").fillna(0.0)

    flooded_gdf = gdf[gdf[flooded_col]].copy()
    loss_gdf = gdf[gdf[loss_col] > 0].copy()

    total_tiv = float(gdf[tiv_col].sum())
    flooded_tiv = float(flooded_gdf[tiv_col].sum())
    total_loss = float(gdf[loss_col].sum())

    event_summary = {
        "event_id": event_id,
        "source_segment_id": source_segment_id,
        "event_source": event_source,
        "severity": severity,
        "total_properties": int(len(gdf)),
        "flooded_properties": int(len(flooded_gdf)),
        "properties_with_loss": int(len(loss_gdf)),
        "total_tiv": total_tiv,
        "flooded_tiv": flooded_tiv,
        "total_gross_loss": total_loss,
        "mean_loss_all_properties": (
            total_loss / len(gdf)
            if len(gdf) > 0
            else 0.0
        ),
        "mean_loss_flooded_properties": (
            total_loss / len(flooded_gdf)
            if len(flooded_gdf) > 0
            else 0.0
        ),
        "loss_ratio_total_tiv": (
            total_loss / total_tiv
            if total_tiv > 0
            else 0.0
        ),
        "loss_ratio_flooded_tiv": (
            total_loss / flooded_tiv
            if flooded_tiv > 0
            else 0.0
        ),
        "mean_damage_ratio_flooded": (
            float(flooded_gdf[damage_ratio_col].mean())
            if len(flooded_gdf) > 0
            else 0.0
        ),
        "max_damage_ratio": float(gdf[damage_ratio_col].max()),
        "max_property_loss": float(gdf[loss_col].max()),
        "tiv_col": tiv_col,
        "damage_ratio_col": damage_ratio_col,
        "loss_col": loss_col,
        "flooded_col": flooded_col,
    }

    return event_summary


def compute_event_loss(
    properties_with_damage_gdf,
    event_id=None,
    source_segment_id=None,
    event_source=None,
    severity=None,
    tiv_col=None,
    damage_ratio_col=None,
    loss_col="building_loss",
    crs="EPSG:3005",
):
    """
    Compute property-level event losses and one event-level summary.

    If properties already have building_loss from compute_property_damage.py,
    this recomputes it using the selected TIV and damage ratio so the summary
    is internally consistent.
    """

    properties_with_loss_gdf = compute_property_event_losses(
        properties_with_damage_gdf=properties_with_damage_gdf,
        tiv_col=tiv_col,
        damage_ratio_col=damage_ratio_col,
        loss_col=loss_col,
        crs=crs,
    )

    event_summary = summarize_event_loss(
        properties_with_loss_gdf=properties_with_loss_gdf,
        event_id=event_id,
        source_segment_id=source_segment_id,
        event_source=event_source,
        severity=severity,
        tiv_col="event_tiv_used",
        damage_ratio_col="event_damage_ratio_used",
        loss_col=loss_col,
        flooded_col="flooded" if "flooded" in properties_with_loss_gdf else None,
    )

    return {
        "properties_with_loss_gdf": properties_with_loss_gdf,
        "event_summary": event_summary,
    }