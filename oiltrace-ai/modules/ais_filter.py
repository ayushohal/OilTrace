"""
AIS spatio-temporal search.

The satellite image is never joined directly to AIS records. The join runs
through the drift model:

    detected spill -> probable origin + release time -> AIS search

Two stages, reported separately so the funnel is visible in the UI:
  1. spatial  - closest approach within `candidate_radius_km` of the origin
  2. temporal - that closest approach within +/- `release_window_hours`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from modules.geo import haversine_km


@dataclass
class VesselTrack:
    mmsi: str
    name: str
    vessel_type: str
    records: pd.DataFrame          # sorted by timestamp
    closest_distance_km: float
    closest_time: datetime
    time_delta_hours: float        # signed: negative = before release

    @property
    def positions(self) -> list[tuple[float, float]]:
        return list(zip(self.records["latitude"], self.records["longitude"]))


@dataclass
class FilterResult:
    total: int
    after_spatial: int
    after_temporal: int
    candidates: list           # list[VesselTrack]
    rejected: list             # list[tuple[VesselTrack, str]]


def load_tracks(ais_df: pd.DataFrame, origin_lat: float, origin_lon: float,
                release_time: datetime) -> list[VesselTrack]:
    """Group AIS records per vessel and measure closest approach to the origin."""
    df = ais_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    release = pd.Timestamp(release_time)
    if release.tzinfo is None:
        release = release.tz_localize("UTC")

    tracks: list[VesselTrack] = []
    for mmsi, group in df.groupby("mmsi", sort=True):
        g = group.sort_values("timestamp").reset_index(drop=True)
        distances = [
            haversine_km(origin_lat, origin_lon, float(r.latitude), float(r.longitude))
            for r in g.itertuples()
        ]
        g = g.assign(distance_to_origin_km=distances)
        idx = int(g["distance_to_origin_km"].idxmin())
        closest_time = g.loc[idx, "timestamp"].to_pydatetime()
        delta_h = (g.loc[idx, "timestamp"] - release).total_seconds() / 3600.0
        tracks.append(
            VesselTrack(
                mmsi=str(mmsi),
                name=str(g.loc[0, "vessel_name"]),
                vessel_type=str(g.loc[0, "vessel_type"]),
                records=g,
                closest_distance_km=float(g.loc[idx, "distance_to_origin_km"]),
                closest_time=closest_time,
                time_delta_hours=float(delta_h),
            )
        )
    tracks.sort(key=lambda t: t.closest_distance_km)
    return tracks


def filter_vessels(tracks: list[VesselTrack], radius_km: float,
                   window_hours: float) -> FilterResult:
    """Apply the spatial filter, then the temporal filter."""
    rejected: list[tuple[VesselTrack, str]] = []

    spatial = []
    for t in tracks:
        if t.closest_distance_km <= radius_km:
            spatial.append(t)
        else:
            rejected.append(
                (t, f"Closest approach {t.closest_distance_km:.1f} km exceeds the "
                    f"{radius_km:.0f} km candidate radius")
            )

    temporal = []
    for t in spatial:
        if abs(t.time_delta_hours) <= window_hours:
            temporal.append(t)
        else:
            rejected.append(
                (t, f"Closest approach {abs(t.time_delta_hours):.1f} h from the estimated "
                    f"release falls outside the +/-{window_hours:.0f} h window")
            )

    return FilterResult(
        total=len(tracks),
        after_spatial=len(spatial),
        after_temporal=len(temporal),
        candidates=temporal,
        rejected=rejected,
    )
