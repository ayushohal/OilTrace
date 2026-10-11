"""
Simplified vector oil-drift model.

    oil_velocity = ocean_current
                 + wind_factor  * wind
                 + wave_factor  * wave

Forward simulation projects the observed slick ahead (+6/+12/+24 h).
Backward simulation (hindcasting) walks the slick centroid back in time to
estimate where the release occurred.

How long to hindcast is NOT assumed. It is derived from the observed slick
itself: a release spreads along the drift axis at a roughly constant rate,
so

    elapsed_hours = observed_slick_length_km / spreading_rate_km_per_h

This is a deliberately simple parameterisation. A production system would
use a numerical Lagrangian particle model (e.g. OpenDrift / GNOME) with
weathering, and would carry an uncertainty ellipse rather than a point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from modules.geo import KM_PER_DEG_LAT, km_per_deg_lon


@dataclass
class DriftResult:
    origin_lat: float
    origin_lon: float
    release_time: datetime
    elapsed_hours: float
    drift_bearing: float
    drift_speed_ms: float
    backward_path: list = field(default_factory=list)   # [(lat, lon, time), ...]
    forward_paths: dict = field(default_factory=dict)   # {hours: [(lat, lon, time)]}
    mean_conditions: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


class EnvironmentField:
    """Nearest-neighbour sampler over the environmental CSV.

    Real metocean data is gridded in space and time; this sampler is the
    seam where an xarray/NetCDF reader would be substituted in production.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        # Store as naive datetime64 so NumPy arithmetic stays vectorised.
        self._times = self.df["timestamp"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        self._lat = self.df["latitude"].to_numpy(dtype=float)
        self._lon = self.df["longitude"].to_numpy(dtype=float)

    def sample(self, lat: float, lon: float, when: datetime) -> dict:
        ts = pd.Timestamp(when)
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        target = np.datetime64(ts.tz_localize(None).to_pydatetime(), "ns")
        dt_hours = np.abs((self._times - target) / np.timedelta64(1, "h"))
        dist = np.hypot((self._lat - lat) * KM_PER_DEG_LAT,
                        (self._lon - lon) * km_per_deg_lon(lat))
        # Weight time and space comparably: 1 hour ~ 10 km of relevance.
        cost = dist + dt_hours * 10.0
        row = self.df.iloc[int(np.argmin(cost))]
        return {
            "current_u": float(row["current_u"]),
            "current_v": float(row["current_v"]),
            "wind_u": float(row["wind_u"]),
            "wind_v": float(row["wind_v"]),
            "wave_direction": float(row["wave_direction"]),
            "wave_speed": float(row["wave_speed"]),
        }


def oil_velocity(env: dict, wind_factor: float, wave_factor: float) -> tuple[float, float]:
    """Return (east, north) oil drift velocity in m/s."""
    wave_dir = math.radians(env["wave_direction"])   # direction waves travel toward
    wave_e = env["wave_speed"] * math.sin(wave_dir)
    wave_n = env["wave_speed"] * math.cos(wave_dir)
    u = env["current_u"] + wind_factor * env["wind_u"] + wave_factor * wave_e
    v = env["current_v"] + wind_factor * env["wind_v"] + wave_factor * wave_n
    return u, v


def _integrate(
    field: EnvironmentField,
    lat: float,
    lon: float,
    start: datetime,
    hours: float,
    step_minutes: int,
    wind_factor: float,
    wave_factor: float,
    backward: bool,
) -> list[tuple[float, float, datetime]]:
    """Step the parcel through the velocity field. Returns the full path."""
    sign = -1.0 if backward else 1.0
    dt_s = step_minutes * 60.0
    n_steps = max(1, int(round(abs(hours) * 60.0 / step_minutes)))

    path = [(lat, lon, start)]
    cur_lat, cur_lon, cur_t = lat, lon, start
    for _ in range(n_steps):
        env = field.sample(cur_lat, cur_lon, cur_t)
        u, v = oil_velocity(env, wind_factor, wave_factor)
        # metres -> kilometres -> degrees
        east_km = sign * u * dt_s / 1000.0
        north_km = sign * v * dt_s / 1000.0
        cur_lat += north_km / KM_PER_DEG_LAT
        cur_lon += east_km / km_per_deg_lon(cur_lat)
        cur_t = cur_t + timedelta(seconds=sign * dt_s)
        path.append((cur_lat, cur_lon, cur_t))
    return path


def _round_time(t: datetime, minutes: int) -> datetime:
    if minutes <= 0:
        return t
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    total = (t - epoch).total_seconds() / 60.0
    return epoch + timedelta(minutes=round(total / minutes) * minutes)


def run_drift(
    spill,
    env_df: pd.DataFrame,
    cfg: dict,
) -> DriftResult:
    """Hindcast the probable origin, then project the forward trajectory."""
    field = EnvironmentField(env_df)
    wind_factor = float(cfg["wind_factor"])
    wave_factor = float(cfg["wave_factor"])
    step = int(cfg["timestep_minutes"])
    notes: list[str] = []

    obs_time = spill.observation_time
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    # --- 1. How far back to hindcast, derived from the slick's own size ---
    rate = float(cfg["spreading_rate_km_per_h"])
    elapsed = spill.length_km / max(rate, 1e-6)
    elapsed = float(min(elapsed, float(cfg["max_hindcast_hours"])))
    notes.append(
        f"Hindcast window derived from observed slick length: "
        f"{spill.length_km:.2f} km / {rate:.1f} km/h = {elapsed:.2f} h."
    )

    # --- 2. Round to the reporting resolution of the estimate ---
    raw_release = obs_time - timedelta(hours=elapsed)
    release = _round_time(raw_release, int(cfg["release_time_rounding_minutes"]))
    elapsed_rounded = (obs_time - release).total_seconds() / 3600.0
    notes.append(
        f"Release time reported at {cfg['release_time_rounding_minutes']}-minute "
        f"resolution (raw estimate {raw_release.strftime('%H:%M')} UTC)."
    )

    # --- 3. Backward integration to the probable origin ---
    backward = _integrate(
        field, spill.latitude, spill.longitude, obs_time, elapsed_rounded,
        step, wind_factor, wave_factor, backward=True,
    )
    origin_lat, origin_lon, _ = backward[-1]

    # --- 4. Mean conditions and net drift vector ---
    samples = [field.sample(la, lo, t) for (la, lo, t) in backward[:: max(1, len(backward) // 8)]]
    mean_env = {k: float(np.mean([s[k] for s in samples])) for k in samples[0]}
    u, v = oil_velocity(mean_env, wind_factor, wave_factor)
    speed = float(math.hypot(u, v))
    bearing = (math.degrees(math.atan2(u, v)) + 360.0) % 360.0

    # --- 5. Forward projection from the observed slick ---
    forward: dict[int, list] = {}
    for hrs in cfg.get("forward_hours", [6, 12, 24]):
        forward[int(hrs)] = _integrate(
            field, spill.latitude, spill.longitude, obs_time, float(hrs),
            step, wind_factor, wave_factor, backward=False,
        )

    return DriftResult(
        origin_lat=round(origin_lat, 4),
        origin_lon=round(origin_lon, 4),
        release_time=release,
        elapsed_hours=round(elapsed_rounded, 2),
        drift_bearing=round(bearing, 1),
        drift_speed_ms=round(speed, 3),
        backward_path=backward,
        forward_paths=forward,
        mean_conditions=mean_env,
        notes=notes,
    )
