# ============================================================
# select_segment.py
#
# Purpose:
# Select an event source and physical source segment for one
# Monte Carlo flood event.
#
# This file runs inside the Monte Carlo loop.
# It assumes segment scores were already created before the loop
# by assign_segment_score.py.
# ============================================================

import numpy as np
import pandas as pd
import geopandas as gpd


def validate_scored_segments_input(
    scored_segments_gdf,
    segment_id_col="segment_id",
    event_source_col="event_source",
    segment_score_col="segment_score",
):
    """
    Validate scored segment GeoDataFrame before segment selection.

    Required:
    - GeoDataFrame
    - CRS
    - geometry
    - segment_id
    - event_source
    - segment_score

    Parameters
    ----------
    scored_segments_gdf : geopandas.GeoDataFrame
        Segment data after assign_segment_score.py has been applied.

    segment_id_col : str
        Column containing segment IDs.

    event_source_col : str
        Column identifying event source type.

    segment_score_col : str
        Column containing physical segment risk score.

    Returns
    -------
    geopandas.GeoDataFrame
        Validated copy of scored_segments_gdf.
    """

    if not isinstance(scored_segments_gdf, gpd.GeoDataFrame):
        raise TypeError("scored_segments_gdf must be a GeoDataFrame.")

    if scored_segments_gdf.empty:
        raise ValueError("scored_segments_gdf is empty.")

    if scored_segments_gdf.crs is None:
        raise ValueError("scored_segments_gdf must have a CRS.")

    required_cols = [
        segment_id_col,
        event_source_col,
        segment_score_col,
        "geometry",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in scored_segments_gdf.columns
    ]

    if missing_cols:
        raise ValueError(f"scored_segments_gdf is missing columns: {missing_cols}")

    out = scored_segments_gdf.copy()

    out = out[
        out.geometry.notna()
        & (~out.geometry.is_empty)
    ].copy()

    if out.empty:
        raise ValueError("No valid geometries after removing null/empty geometries.")

    if out[segment_id_col].isna().any():
        raise ValueError(f"{segment_id_col} contains missing values.")

    if out[event_source_col].isna().any():
        raise ValueError(f"{event_source_col} contains missing values.")

    if out[segment_score_col].isna().any():
        raise ValueError(f"{segment_score_col} contains missing values.")

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

DEFAULT_EVENT_SOURCE_WEIGHTS = {
    "coastal": 0.20,
    "riverine": 0.40,
    "interior_drainage": 0.40,
}


def select_event_source(
    event_source_weights=DEFAULT_EVENT_SOURCE_WEIGHTS,
    random_seed=None,
):
    """
    Randomly select the event source type for one simulation iteration.

    Parameters
    ----------
    event_source_weights : dict
        Dictionary of event source probabilities.
        Example:
        {
            "coastal": 0.20,
            "riverine": 0.40,
            "interior_drainage": 0.40,
        }

    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    str
        Selected event source.
    """

    if not isinstance(event_source_weights, dict):
        raise TypeError("event_source_weights must be a dictionary.")

    if len(event_source_weights) == 0:
        raise ValueError("event_source_weights cannot be empty.")

    event_sources = list(event_source_weights.keys())
    probabilities = np.array(list(event_source_weights.values()), dtype=float)

    if np.any(probabilities < 0):
        raise ValueError("Event source probabilities cannot be negative.")

    if probabilities.sum() <= 0:
        raise ValueError("Event source probabilities must sum to a positive value.")

    # Normalize just in case weights do not sum exactly to 1
    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(random_seed)

    selected_event_source = rng.choice(
        event_sources,
        p=probabilities,
    )

    return selected_event_source


def select_segment_from_event_source(
    scored_segments_gdf,
    event_source,
    segment_score_col="segment_score",
    event_source_col="event_source",
    selection_slope=0.15,
    random_seed=None,
):
    """
    Select one line segment from a selected event source.

    Segments with higher segment_score receive higher selection weight,
    but selection_slope controls how strong that weighting is.

    Parameters
    ----------
    scored_segments_gdf : geopandas.GeoDataFrame
        Segment data with segment_score already assigned.

    event_source : str
        Selected event source, e.g. "riverine".

    segment_score_col : str
        Column containing segment risk score.

    event_source_col : str
        Column containing event source labels.

    selection_slope : float
        Controls how strongly segment_score affects selection probability.
        Small values avoid over-weighting high-risk segments.

    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pandas.Series
        Selected segment row.
    """

    if event_source_col not in scored_segments_gdf.columns:
        raise ValueError(f"Missing column: {event_source_col}")

    if segment_score_col not in scored_segments_gdf.columns:
        raise ValueError(f"Missing column: {segment_score_col}")

    candidate_segments = scored_segments_gdf[
        scored_segments_gdf[event_source_col] == event_source
    ].copy()

    if candidate_segments.empty:
        raise ValueError(f"No segments found for event_source={event_source}")

    if candidate_segments[segment_score_col].isna().any():
        raise ValueError(f"{segment_score_col} contains missing values for {event_source}")

    # --------------------------------------------------------
    # Score-weighted selection
    # --------------------------------------------------------
    # Higher segment score = higher selection weight
    # Small slope keeps this from becoming too aggressive.

    candidate_segments["segment_selection_weight"] = np.exp(
        selection_slope * candidate_segments[segment_score_col]
    )

    total_weight = candidate_segments["segment_selection_weight"].sum()

    if total_weight <= 0:
        raise ValueError("Segment selection weights must sum to a positive value.")

    candidate_segments["segment_selection_probability"] = (
        candidate_segments["segment_selection_weight"] / total_weight
    )

    rng = np.random.default_rng(random_seed)

    selected_index = rng.choice(
        candidate_segments.index.to_numpy(),
        p=candidate_segments["segment_selection_probability"].to_numpy(),
    )

    selected_segment = candidate_segments.loc[selected_index].copy()

    return selected_segment


def select_event_segment(
    scored_segments_gdf,
    event_source_weights=DEFAULT_EVENT_SOURCE_WEIGHTS,
    segment_id_col="segment_id",
    event_source_col="event_source",
    segment_score_col="segment_score",
    selection_slope=0.15,
    random_seed=None,
):
    """
    Select one event source and one physical source segment.

    This is the main wrapper function used inside the Monte Carlo loop.

    Steps:
    1. Validate scored segment input
    2. Select event source
    3. Select one segment from that event source using score-weighted probabilities

    Parameters
    ----------
    scored_segments_gdf : geopandas.GeoDataFrame
        Segment data with segment_score already assigned.

    event_source_weights : dict
        Event source probability weights.

    segment_id_col : str
        Segment ID column.

    event_source_col : str
        Event source column.

    segment_score_col : str
        Segment score column.

    selection_slope : float
        Controls how strongly segment_score affects segment selection.

    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pandas.Series
        Selected segment row, including:
        - selected event source
        - segment_id
        - segment_score
        - segment_selection_weight
        - segment_selection_probability
    """

    scored_segments_validated = validate_scored_segments_input(
        scored_segments_gdf=scored_segments_gdf,
        segment_id_col=segment_id_col,
        event_source_col=event_source_col,
        segment_score_col=segment_score_col,
    )

    rng = np.random.default_rng(random_seed)

    event_source_seed = rng.integers(0, 1_000_000_000)
    segment_seed = rng.integers(0, 1_000_000_000)

    selected_event_source = select_event_source(
        event_source_weights=event_source_weights,
        random_seed=event_source_seed,
    )

    selected_segment = select_segment_from_event_source(
        scored_segments_gdf=scored_segments_validated,
        event_source=selected_event_source,
        segment_score_col=segment_score_col,
        event_source_col=event_source_col,
        selection_slope=selection_slope,
        random_seed=segment_seed,
    )

    return selected_segment