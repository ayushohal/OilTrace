"""
Trajectory analysis.

For each candidate vessel, measure how well its movement around the
estimated release lines up with the reconstructed drift of the oil.

A vessel that discharged while under way leaves oil along its own track,
so the slick's drift-corrected axis should follow the vessel's course.
Alignment is a cosine similarity between two compass bearings, rescaled
to 0-100.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from modules.geo import bearing_deg, angular_difference


@dataclass
class TrajectoryMetrics:
    distance_km: float
    time_delta_hours: float
    course_bearing: float
    drift_bearing: float
    angular_error_deg: float
    alignment_score: float      # 0-100
    proximity_score: float      # 0-100
    temporal_score: float       # 0-100
    mean_speed_kn: float


def _course_over_window(track, centre: datetime, window_hours: float = 1.5) -> float:
    """Net course made good over a window centred on the closest approach."""
    df = track.records
    centre_ts = pd.Timestamp(centre)
    if centre_ts.tzinfo is None:
        centre_ts = centre_ts.tz_localize("UTC")
    lo = centre_ts - timedelta(hours=window_hours)
    hi = centre_ts + timedelta(hours=window_hours)
    seg = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    if len(seg) < 2:
        seg = df
    if len(seg) < 2:
        return float(df["heading"].mean())

    first, last = seg.iloc[0], seg.iloc[-1]
    if abs(first["latitude"] - last["latitude"]) < 1e-9 and \
       abs(first["longitude"] - last["longitude"]) < 1e-9:
        return float(seg["heading"].mean())
    return bearing_deg(
        float(first["latitude"]), float(first["longitude"]),
        float(last["latitude"]), float(last["longitude"]),
    )


def analyse_trajectory(track, drift_bearing: float, radius_km: float,
                       window_hours: float) -> TrajectoryMetrics:
    """Compute the proximity, temporal and alignment components for a vessel."""
    course = _course_over_window(track, track.closest_time)
    angular_error = angular_difference(course, drift_bearing)

    # Cosine similarity mapped from [-1, 1] onto [0, 100].
    cosine = float(np.cos(np.radians(angular_error)))
    alignment = 50.0 * (1.0 + cosine)

    proximity = 100.0 * max(0.0, 1.0 - track.closest_distance_km / max(radius_km, 1e-6))
    temporal = 100.0 * max(0.0, 1.0 - abs(track.time_delta_hours) / max(window_hours, 1e-6))

    return TrajectoryMetrics(
        distance_km=round(track.closest_distance_km, 2),
        time_delta_hours=round(track.time_delta_hours, 2),
        course_bearing=round(course, 1),
        drift_bearing=round(drift_bearing, 1),
        angular_error_deg=round(angular_error, 1),
        alignment_score=round(alignment, 1),
        proximity_score=round(proximity, 1),
        temporal_score=round(temporal, 1),
        mean_speed_kn=round(float(track.records["speed"].mean()), 1),
    )
