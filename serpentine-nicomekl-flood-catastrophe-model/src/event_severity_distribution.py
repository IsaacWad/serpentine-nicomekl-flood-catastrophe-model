# ============================================================
# event_severity_distribution.py
#
# Purpose:
# Build an event severity probability distribution conditional on:
#   1. event source type
#   2. selected segment physical score
#
# The event source controls the baseline severity mix.
# The segment score applies an exponential tilt toward higher severities.
# ============================================================

import numpy as np
import pandas as pd


DEFAULT_SEVERITY_VALUES = {
    "minor": 0,
    "moderate": 1,
    "severe": 2,
    "extreme": 3,
}


DEFAULT_BASE_SEVERITY_WEIGHTS = {
    "minor": 0.70,
    "moderate": 0.23,
    "severe": 0.06,
    "extreme": 0.01,
}


DEFAULT_BASE_SEVERITY_WEIGHTS_BY_EVENT_SOURCE = {
    "default": DEFAULT_BASE_SEVERITY_WEIGHTS,
    "riverine": {
        "minor": 0.70,
        "moderate": 0.23,
        "severe": 0.06,
        "extreme": 0.01,
    },
    "interior_drainage": {
        "minor": 0.70,
        "moderate": 0.23,
        "severe": 0.06,
        "extreme": 0.01,
    },
    "coastal": {
        "minor": 0.78,
        "moderate": 0.18,
        "severe": 0.035,
        "extreme": 0.005,
    },
}


DEFAULT_SEVERITY_TILT_SLOPE = 0.50


def normalize_event_source(event_source):
    """
    Normalize event-source labels used by the simulation.
    """

    if event_source is None:
        return None

    event_source = str(event_source).strip().lower()

    aliases = {
        "coast": "coastal",
        "coastal": "coastal",
        "river": "riverine",
        "riverine": "riverine",
        "interior": "interior_drainage",
        "interior drainage": "interior_drainage",
        "interior_drainage": "interior_drainage",
    }

    return aliases.get(event_source, event_source)


def get_base_severity_weights_for_event_source(
    event_source=None,
    base_severity_weights_by_event_source=None,
):
    """
    Return baseline severity weights for an event source.

    If event_source is missing or unknown, the default baseline is used.
    """

    if base_severity_weights_by_event_source is None:
        base_severity_weights_by_event_source = (
            DEFAULT_BASE_SEVERITY_WEIGHTS_BY_EVENT_SOURCE
        )

    event_source = normalize_event_source(event_source)

    if event_source is None:
        return base_severity_weights_by_event_source["default"]

    return base_severity_weights_by_event_source.get(
        event_source,
        base_severity_weights_by_event_source["default"],
    )


def build_event_severity_distribution(
    segment_score,
    event_source=None,
    base_severity_weights=None,
    base_severity_weights_by_event_source=None,
    severity_values=DEFAULT_SEVERITY_VALUES,
    severity_tilt_slope=DEFAULT_SEVERITY_TILT_SLOPE,
):
    """
    Build event severity probabilities conditional on segment score.

    The baseline severity mix can differ by event source. Higher segment_score
    then shifts probability mass toward more severe outcomes using an
    exponential tilt:

        tilted_weight = base_weight * exp(slope * score * severity_value)

    Parameters
    ----------
    segment_score : float
        Physical risk score assigned to the selected segment.

    event_source : str or None
        Event source type: coastal, riverine, interior_drainage, etc.

    base_severity_weights : dict or None
        Optional explicit baseline severity weights. If provided, this
        overrides event-source baseline weights.

    base_severity_weights_by_event_source : dict or None
        Mapping from event source to baseline severity weights.

    severity_values : dict
        Numeric ordering for event severities.

    severity_tilt_slope : float
        Controls how strongly segment_score shifts probability toward
        higher severity outcomes.

    Returns
    -------
    dict
        Severity probability distribution.
    """

    if segment_score is None:
        raise ValueError("segment_score cannot be None.")

    if pd.isna(segment_score):
        raise ValueError("segment_score cannot be NaN.")

    if not isinstance(severity_values, dict):
        raise TypeError("severity_values must be a dictionary.")

    if base_severity_weights is None:
        base_severity_weights = get_base_severity_weights_for_event_source(
            event_source=event_source,
            base_severity_weights_by_event_source=(
                base_severity_weights_by_event_source
            ),
        )

    if not isinstance(base_severity_weights, dict):
        raise TypeError("base_severity_weights must be a dictionary.")

    if set(base_severity_weights.keys()) != set(severity_values.keys()):
        raise ValueError(
            "base_severity_weights and severity_values must have the same "
            "severity keys."
        )

    tilted_weights = {}

    for severity, base_weight in base_severity_weights.items():
        if base_weight < 0:
            raise ValueError("Base severity weights cannot be negative.")

        severity_value = severity_values[severity]

        tilted_weights[severity] = (
            float(base_weight)
            * np.exp(severity_tilt_slope * segment_score * severity_value)
        )

    total_weight = sum(tilted_weights.values())

    if total_weight <= 0:
        raise ValueError("Tilted severity weights must sum to a positive value.")

    return {
        severity: weight / total_weight
        for severity, weight in tilted_weights.items()
    }


def draw_event_severity(
    severity_distribution,
    random_seed=None,
):
    """
    Draw one event severity from a severity probability distribution.
    """

    if not isinstance(severity_distribution, dict):
        raise TypeError("severity_distribution must be a dictionary.")

    if len(severity_distribution) == 0:
        raise ValueError("severity_distribution cannot be empty.")

    severities = list(severity_distribution.keys())
    probabilities = np.array(list(severity_distribution.values()), dtype=float)

    if np.any(probabilities < 0):
        raise ValueError("Severity probabilities cannot be negative.")

    if probabilities.sum() <= 0:
        raise ValueError("Severity probabilities must sum to a positive value.")

    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(random_seed)

    return rng.choice(
        severities,
        p=probabilities,
    )


def draw_event_severity_for_segment(
    selected_segment_row,
    segment_score_col="segment_score",
    event_source_col="event_source",
    base_severity_weights=None,
    base_severity_weights_by_event_source=None,
    severity_values=DEFAULT_SEVERITY_VALUES,
    severity_tilt_slope=DEFAULT_SEVERITY_TILT_SLOPE,
    random_seed=None,
):
    """
    Full wrapper used after a segment has already been selected.

    Takes one selected scored segment row, picks the baseline severity weights
    from its event source, applies the segment-score tilt, then draws one
    event severity.
    """

    if segment_score_col not in selected_segment_row.index:
        raise ValueError(
            f"selected_segment_row is missing column: {segment_score_col}"
        )

    segment_score = selected_segment_row[segment_score_col]

    event_source = None
    if event_source_col in selected_segment_row.index:
        event_source = selected_segment_row[event_source_col]

    severity_distribution = build_event_severity_distribution(
        segment_score=segment_score,
        event_source=event_source,
        base_severity_weights=base_severity_weights,
        base_severity_weights_by_event_source=(
            base_severity_weights_by_event_source
        ),
        severity_values=severity_values,
        severity_tilt_slope=severity_tilt_slope,
    )

    event_severity = draw_event_severity(
        severity_distribution=severity_distribution,
        random_seed=random_seed,
    )

    return {
        "event_source": normalize_event_source(event_source),
        "segment_score": segment_score,
        "severity_distribution": severity_distribution,
        "event_severity": event_severity,
    }