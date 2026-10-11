"""
Behavioural analysis.

Five simple indicators computed from the AIS track. They describe motion
patterns that are *worth an investigator's attention* near a probable
release point - a speed drop, a course change, a stop, wandering off a
straight route, or a gap in reporting.

None of these prove an illegal discharge. Vessels slow down, turn and lose
AIS coverage for many lawful reasons. The output is deliberately named
"Behavioural anomaly score" and contributes only 15% of the final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from modules.geo import haversine_km, angular_difference, bearing_deg


@dataclass
class BehaviourMetrics:
    speed_change_score: float
    heading_change_score: float
    loitering_score: float
    route_deviation_score: float
    ais_gap_score: float
    anomaly_score: float          # 0-100, mean of the five indicators
    continuity_score: float       # 0-100, separate attribution component
    max_speed_drop_kn: float
    max_heading_change_deg: float
    min_speed_kn: float
    max_gap_minutes: float
    indicators: list              # human-readable strings for the UI


def analyse_behaviour(track, closest_time, expected_interval_minutes: int = 15,
                      window_hours: float = 2.0) -> BehaviourMetrics:
    df = track.records
    centre = pd.Timestamp(closest_time)
    if centre.tzinfo is None:
        centre = centre.tz_localize("UTC")

    seg = df[(df["timestamp"] >= centre - timedelta(hours=window_hours)) &
             (df["timestamp"] <= centre + timedelta(hours=window_hours))]
    if len(seg) < 3:
        seg = df

    speeds = seg["speed"].to_numpy(dtype=float)
    headings = seg["heading"].to_numpy(dtype=float)
    times = seg["timestamp"].to_numpy()
    lats = seg["latitude"].to_numpy(dtype=float)
    lons = seg["longitude"].to_numpy(dtype=float)

    indicators: list[str] = []

    # --- 1. Sudden speed change -------------------------------------
    if len(speeds) > 1:
        drops = np.abs(np.diff(speeds))
        max_drop = float(drops.max())
    else:
        max_drop = 0.0
    speed_change = float(np.clip(max_drop / 8.0, 0.0, 1.0))
    if max_drop >= 3.0:
        indicators.append(f"Speed changed by {max_drop:.1f} kn between consecutive reports")

    # --- 2. Sudden heading change -----------------------------------
    if len(headings) > 1:
        turns = [angular_difference(headings[i], headings[i - 1]) for i in range(1, len(headings))]
        max_turn = float(max(turns))
    else:
        max_turn = 0.0
    heading_change = float(np.clip(max_turn / 60.0, 0.0, 1.0))
    if max_turn >= 20.0:
        indicators.append(f"Course altered by {max_turn:.0f} deg near the probable origin")

    # --- 3. Loitering / stopping ------------------------------------
    min_speed = float(speeds.min()) if len(speeds) else 0.0
    loitering = float(np.clip((6.0 - min_speed) / 6.0, 0.0, 1.0))
    if min_speed <= 4.0:
        indicators.append(f"Slowed to {min_speed:.1f} kn within the release window")

    # --- 4. Route deviation -----------------------------------------
    if len(lats) >= 3:
        # Mean cross-track distance from the straight line joining the ends.
        brg = bearing_deg(lats[0], lons[0], lats[-1], lons[-1])
        devs = []
        for i in range(1, len(lats) - 1):
            d = haversine_km(lats[0], lons[0], lats[i], lons[i])
            b = bearing_deg(lats[0], lons[0], lats[i], lons[i])
            devs.append(abs(d * np.sin(np.radians(angular_difference(b, brg)))))
        mean_dev = float(np.mean(devs)) if devs else 0.0
    else:
        mean_dev = 0.0
    route_deviation = float(np.clip(mean_dev / 2.0, 0.0, 1.0))
    if mean_dev >= 0.4:
        indicators.append(f"Track deviates {mean_dev:.2f} km from a direct route")

    # --- 5. AIS reporting gaps (whole track, not just the window) -----
    all_times = df["timestamp"].to_numpy()
    if len(all_times) > 1:
        gaps_min = np.diff(all_times) / np.timedelta64(1, "m")
        max_gap = float(gaps_min.max())
        expected = float(expected_interval_minutes)
        # Continuity: fraction of the track covered at the expected rate.
        missing = float(np.sum(np.clip(gaps_min - expected * 1.5, 0.0, None)))
        span = float((all_times[-1] - all_times[0]) / np.timedelta64(1, "m"))
        continuity = 100.0 * float(np.clip(1.0 - missing / max(span, 1.0), 0.0, 1.0))
    else:
        max_gap, continuity = 0.0, 0.0

    gap_flag = float(np.clip((max_gap - expected_interval_minutes * 1.5) / 60.0, 0.0, 1.0))
    if gap_flag > 0:
        indicators.append(f"AIS reporting gap of {max_gap:.0f} minutes")

    if not indicators:
        indicators.append("No notable behavioural deviation detected")

    anomaly = 100.0 * float(
        np.mean([speed_change, heading_change, loitering, route_deviation, gap_flag])
    )

    return BehaviourMetrics(
        speed_change_score=round(speed_change * 100, 1),
        heading_change_score=round(heading_change * 100, 1),
        loitering_score=round(loitering * 100, 1),
        route_deviation_score=round(route_deviation * 100, 1),
        ais_gap_score=round(gap_flag * 100, 1),
        anomaly_score=round(anomaly, 1),
        continuity_score=round(continuity, 1),
        max_speed_drop_kn=round(max_drop, 1),
        max_heading_change_deg=round(max_turn, 1),
        min_speed_kn=round(min_speed, 1),
        max_gap_minutes=round(max_gap, 1),
        indicators=indicators,
    )
