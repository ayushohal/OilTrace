"""
Deterministic demo-data generation.

Builds, for each scene in the library, three SEPARATE datasets that
describe ONE coherent scenario:

  1. a synthetic SAR scene + ground-truth mask
  2. environment.csv  (metocean field)
  3. vessels.csv      (AIS extract)

Design principle: the *inputs* are engineered so the pipeline naturally
rediscovers the intended story. No output value is written anywhere.

The currents in environment.csv are solved from the designed geometry:

    required mean oil velocity = (observed centroid - intended origin) / elapsed
    current = required - wind_factor*wind - wave_factor*wave

so when drift.py integrates the field backwards it lands on the intended
origin because the physics is consistent, not because it was told to.

Vessel tracks are defined by ROLE, relative to the scene's drift bearing
and release time. The same role table therefore produces a coherent
scenario in any scene, at any location, on any heading.

Everything uses a fixed per-scene seed. Each demo runs identically every
time.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modules.geo import BBox, KM_PER_DEG_LAT, km_per_deg_lon, offset_km, bearing_deg
from utils.scene_builder import build_scene, write_scene

KNOTS_TO_MS = 0.514444


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
def build_environment(
    bbox: BBox,
    centroid: tuple[float, float],
    origin: tuple[float, float],
    observation_time: datetime,
    release_time: datetime,
    wind_factor: float,
    wave_factor: float,
    seed: int,
) -> pd.DataFrame:
    """Solve for the current field that carries the origin to the centroid."""
    elapsed_h = (observation_time - release_time).total_seconds() / 3600.0
    clat, clon = centroid
    olat, olon = origin

    east_km = (clon - olon) * km_per_deg_lon(clat)
    north_km = (clat - olat) * KM_PER_DEG_LAT

    # Required mean oil-drift velocity, m/s
    req_u = east_km * 1000.0 / (elapsed_h * 3600.0)
    req_v = north_km * 1000.0 / (elapsed_h * 3600.0)

    # Wind and wave forcing roughly aligned with the net drift, so the
    # scene reads as physically plausible rather than arbitrary.
    drift_brg = math.degrees(math.atan2(req_u, req_v)) % 360.0
    wind_speed, wave_speed = 4.0, 1.8
    wind_u = wind_speed * math.sin(math.radians(drift_brg))
    wind_v = wind_speed * math.cos(math.radians(drift_brg))
    wave_dir = drift_brg
    wave_u = wave_speed * math.sin(math.radians(wave_dir))
    wave_v = wave_speed * math.cos(math.radians(wave_dir))

    base_cu = req_u - wind_factor * wind_u - wave_factor * wave_u
    base_cv = req_v - wind_factor * wind_v - wave_factor * wave_v

    lats = np.linspace(bbox.south + 0.05, bbox.north - 0.05, 3)
    lons = np.linspace(bbox.west + 0.06, bbox.east - 0.06, 3)
    hours = list(range(0, 25))
    day = observation_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # Zero-mean hourly modulation across the hindcast window, so the mean
    # velocity over the integration stays exactly as solved above.
    window = [h for h in hours if release_time.hour <= h <= observation_time.hour]
    modulation = {h: 0.03 * math.sin(2 * math.pi * (h - 6) / 9.0) for h in hours}
    if window:
        offset = float(np.mean([modulation[h] for h in window]))
        modulation = {h: m - offset for h, m in modulation.items()}

    rows = []
    for h in hours:
        ts = day + timedelta(hours=h)
        m = modulation[h]
        for la in lats:
            for lo in lons:
                # Zero-mean spatial texture: the field is not perfectly
                # uniform, but introduces no net bias.
                sx = 0.006 * math.sin(2 * math.pi * (lo - bbox.west) / max(bbox.lon_span, 1e-9))
                sy = 0.006 * math.sin(2 * math.pi * (la - bbox.south) / max(bbox.lat_span, 1e-9))
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "latitude": round(float(la), 4),
                    "longitude": round(float(lo), 4),
                    "current_u": round(base_cu * (1.0 + m) + sx, 4),
                    "current_v": round(base_cv * (1.0 + m) + sy, 4),
                    "wind_u": round(wind_u + 0.35 * m * 10, 3),
                    "wind_v": round(wind_v + 0.25 * m * 10, 3),
                    "wave_direction": round(wave_dir + 2.0 * m * 10, 1),
                    "wave_speed": round(wave_speed + 0.15 * m * 10, 3),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# AIS - role-based fleet
# ---------------------------------------------------------------------
# Each role is expressed RELATIVE to the scene: `cpa_km` is the closest
# approach to the probable origin, `dt_h` the offset from the estimated
# release, `d_course` the heading offset from the reconstructed drift
# bearing. The same table therefore yields a coherent scenario anywhere.
#
#        role                 cpa    side  dt_h   d_course  kn   flags
FLEET_ROLES = [
    ("culprit",               None,   -1,   0.00,     0.0, 12.0,
     {"loiter": True, "course_change": 28.0, "loiter_speed_kn": 1.8}),
    ("moderate correlation",   7.2,   +1,  +0.33,    27.0, 11.0, {"course_change": 10.0}),
    ("against the drift",      4.3,   +1,  +0.08,   175.0,  9.0, {}),
    ("distant, early, AIS gap",16.8,  +1,  -1.33,     3.0, 11.0, {"gap_minutes": 50}),
    ("close, wrong time",      3.1,   -1,  -4.50,     3.0, 10.0, {}),
    ("mid-range, wrong time",  12.5,  +1,  -3.17,    73.0, 13.0, {"gap_minutes": 40}),
    ("beyond radius",          24.0,  +1,  +1.50,    43.0, 12.0, {}),
    ("beyond radius",          33.0,  -1,  -1.00,   -42.0, 10.0, {}),
    ("beyond radius",          46.0,  +1,   0.00,   133.0, 14.0, {}),
    ("beyond radius",          61.0,  +1,  +1.00,   218.0, 16.0, {}),
]


def build_fleet_plan(fleet, drift_bearing: float, release_hour: float,
                     culprit_cpa_km: float):
    """Bind the role table to a scene's geometry and vessel names."""
    plan = []
    for (name, mmsi, vtype), role in zip(fleet, FLEET_ROLES):
        label, cpa, side, dt_h, d_course, kn, flags = role
        plan.append({
            "name": name, "mmsi": mmsi, "type": vtype, "role": label,
            "cpa_km": culprit_cpa_km if cpa is None else cpa,
            "side": side,
            "t0_h": release_hour + dt_h,
            "course": (drift_bearing + d_course) % 360.0,
            "speed_kn": kn,
            "flags": flags,
        })
    return plan


def _track_for(spec, origin, day, release, interval_min: int, rng) -> list[dict]:
    olat, olon = origin
    course = spec["course"]
    flags = spec["flags"]
    speed_kn = spec["speed_kn"]

    # CPA point: offset perpendicular to the vessel's course, so the
    # designed distance really is the closest approach.
    off_brg = (course + 90.0 * spec["side"]) % 360.0
    pe = spec["cpa_km"] * math.sin(math.radians(off_brg))
    pn = spec["cpa_km"] * math.cos(math.radians(off_brg))
    cpa_lat, cpa_lon = offset_km(olat, olon, pe, pn)
    t0 = day + timedelta(hours=spec["t0_h"])

    # Track window centred on the estimated release rather than a fixed
    # calendar day, so a release near midnight still yields full tracks.
    span_min = 8 * 60
    t_start = release - timedelta(minutes=span_min)
    times = [t_start + timedelta(minutes=m)
             for m in range(0, 2 * span_min + 1, interval_min)]
    cruise_ms = speed_kn * KNOTS_TO_MS
    loiter_kn = float(flags.get("loiter_speed_kn", 2.9))

    rows, prev = [], None
    for t in times:
        dt_h = (t - t0).total_seconds() / 3600.0

        if flags.get("loiter") and abs(dt_h) <= 0.55:
            speed = loiter_kn + 1.4 * abs(dt_h) / 0.55
        else:
            speed = speed_kn + float(rng.normal(0.0, 0.12))
        speed = max(0.4, speed)

        cc = flags.get("course_change", 0.0)
        leg_course = course + (cc if dt_h > 0.25 else 0.0)

        if flags.get("loiter"):
            if dt_h <= -0.55:
                along = cruise_ms * (dt_h + 0.55) * 3.6
            elif dt_h < 0.55:
                along = (loiter_kn + 0.5) * KNOTS_TO_MS * dt_h * 3.6
            else:
                along = (cruise_ms * (dt_h - 0.55) * 3.6
                         + (loiter_kn + 0.5) * KNOTS_TO_MS * 0.55 * 3.6)
        else:
            along = cruise_ms * dt_h * 3.6      # km  (m/s * h * 3.6)

        e = along * math.sin(math.radians(leg_course))
        n = along * math.cos(math.radians(leg_course))
        wander = 0.04 * math.sin(dt_h * 1.7) + float(rng.normal(0.0, 0.012))
        e += wander * math.cos(math.radians(leg_course))
        n -= wander * math.sin(math.radians(leg_course))

        lat, lon = offset_km(cpa_lat, cpa_lon, e, n)

        heading = leg_course
        if prev is not None:
            d = math.hypot((lat - prev[0]) * KM_PER_DEG_LAT,
                           (lon - prev[1]) * km_per_deg_lon(lat))
            if d > 0.02:
                heading = bearing_deg(prev[0], prev[1], lat, lon)
        prev = (lat, lon)

        rows.append({
            "mmsi": spec["mmsi"],
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latitude": round(lat, 5), "longitude": round(lon, 5),
            "speed": round(speed, 1), "heading": round(heading % 360.0, 1),
            "vessel_name": spec["name"], "vessel_type": spec["type"],
        })

    gap = flags.get("gap_minutes")
    if gap:
        lo = t0 - timedelta(minutes=gap + 20)
        hi = t0 - timedelta(minutes=20)
        rows = [r for r in rows
                if not (lo <= datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                        .replace(tzinfo=timezone.utc) <= hi)]
    return rows


def build_ais(fleet, origin, observation_time, release_time, drift_bearing,
              culprit_cpa_km, interval_min: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 23)
    day = observation_time.replace(hour=0, minute=0, second=0, microsecond=0)
    release_hour = (release_time - day).total_seconds() / 3600.0
    plan = build_fleet_plan(fleet, drift_bearing, release_hour, culprit_cpa_km)
    rows: list[dict] = []
    for spec in plan:
        rows.extend(_track_for(spec, origin, day, release_time, interval_min, rng))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------
def scene_paths(root: Path, scene_id: str) -> dict:
    base = root / "data" / "scenes" / scene_id
    return {
        "dir": base,
        "image": base / "scene.png",
        "mask": base / "scene_mask.png",
        "environment": base / "environment.csv",
        "ais": base / "vessels.csv",
    }


def ensure_scene(cfg: dict, scene: dict, root: Path, force: bool = False) -> dict:
    """Create every asset for one scene that does not already exist."""
    p = scene_paths(root, scene["id"])
    p["dir"].mkdir(parents=True, exist_ok=True)
    bbox = BBox.from_config(scene)

    obs = datetime.fromisoformat(scene["observation_time_utc"].replace("Z", "+00:00"))
    rel = datetime.fromisoformat(scene["truth_release_time_utc"].replace("Z", "+00:00"))
    centroid = tuple(scene["truth_spill_centroid"])
    origin = tuple(scene["truth_origin"])
    drift_bearing = bearing_deg(origin[0], origin[1], centroid[0], centroid[1])

    created = []

    if force or not p["image"].exists() or not p["mask"].exists():
        image, mask = build_scene(
            bbox=bbox,
            width=int(scene["image_width"]), height=int(scene["image_height"]),
            centroid_latlon=centroid,
            slick_length_km=float(scene["slick_length_km"]),
            slick_area_km2=float(scene["slick_area_km2"]),
            drift_bearing_deg=drift_bearing,
            seed=int(scene["random_seed"]),
        )
        write_scene(image, mask, p["image"], p["mask"])
        created.append("SAR scene + reference mask")

    if force or not p["environment"].exists():
        build_environment(
            bbox=bbox, centroid=centroid, origin=origin,
            observation_time=obs, release_time=rel,
            wind_factor=float(cfg["drift"]["wind_factor"]),
            wave_factor=float(cfg["drift"]["wave_factor"]),
            seed=int(scene["random_seed"]),
        ).to_csv(p["environment"], index=False)
        created.append("environment.csv")

    if force or not p["ais"].exists():
        build_ais(
            fleet=scene["fleet"], origin=origin,
            observation_time=obs, release_time=rel,
            drift_bearing=drift_bearing,
            culprit_cpa_km=float(scene["culprit_cpa_km"]),
            interval_min=int(cfg["ais"]["ais_reporting_interval_minutes"]),
            seed=int(scene["random_seed"]),
        ).to_csv(p["ais"], index=False)
        created.append("vessels.csv")

    p["created"] = created
    p["bbox"] = bbox
    p["observation_time"] = obs
    p["drift_bearing"] = drift_bearing
    return p


def ensure_all_scenes(cfg: dict, root: Path, force: bool = False) -> dict:
    return {s["id"]: ensure_scene(cfg, s, root, force) for s in cfg["scenes"]}


def get_scene(cfg: dict, scene_id: str) -> dict:
    for s in cfg["scenes"]:
        if s["id"] == scene_id:
            return s
    return cfg["scenes"][0]


# ---------------------------------------------------------------------
# Adaptive context for images with no matching scenario
# ---------------------------------------------------------------------
def synthesise_context(cfg: dict, spill, bbox: BBox) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build a metocean field and AIS extract around a DETECTED spill.

    Used for uploaded images and external-dataset images, where no real
    environmental or AIS data exists. The spill drives the scenario, so
    the downstream pipeline stays coherent and the demonstration still
    shows a meaningful ranking.

    This demonstrates the mechanism. It is not a real investigation.
    """
    ac = cfg["adaptive_context"]
    drift_bearing = float(ac["drift_bearing_deg"])

    obs = spill.observation_time
    elapsed = float(spill.length_km) / float(cfg["drift"]["spreading_rate_km_per_h"])
    elapsed = min(elapsed, float(cfg["drift"]["max_hindcast_hours"]))
    # Snap to the same resolution the drift module reports at.
    step = float(cfg["drift"]["release_time_rounding_minutes"]) / 60.0
    elapsed = max(step, round(elapsed / step) * step)
    release = obs - timedelta(hours=elapsed)

    # Place the notional origin up-drift of the detected centroid.
    travel_km = 0.0
    # Distance implied by a typical prototype drift speed over the window.
    typical_speed_ms = 0.67
    travel_km = typical_speed_ms * elapsed * 3.6
    back = (drift_bearing + 180.0) % 360.0
    origin = offset_km(
        spill.latitude, spill.longitude,
        travel_km * math.sin(math.radians(back)),
        travel_km * math.cos(math.radians(back)),
    )

    env = build_environment(
        bbox=bbox, centroid=(spill.latitude, spill.longitude), origin=origin,
        observation_time=obs, release_time=release,
        wind_factor=float(cfg["drift"]["wind_factor"]),
        wave_factor=float(cfg["drift"]["wave_factor"]),
        seed=int(ac["random_seed"]),
    )

    prefix = ac.get("fleet_prefix", "DEMO VESSEL")
    types = ["Oil/Chemical Tanker", "Bulk Carrier", "Container Ship", "General Cargo",
             "Crude Oil Tanker", "Bulk Carrier", "Tug/Supply", "Container Ship",
             "General Cargo", "Vehicle Carrier"]
    fleet = [(f"{prefix} {chr(65 + i)}", f"9{ac['random_seed']:03d}{i:04d}", types[i])
             for i in range(10)]

    ais = build_ais(
        fleet=fleet, origin=origin, observation_time=obs, release_time=release,
        drift_bearing=drift_bearing, culprit_cpa_km=float(ac["culprit_cpa_km"]),
        interval_min=int(cfg["ais"]["ais_reporting_interval_minutes"]),
        seed=int(ac["random_seed"]),
    )

    info = {
        "notional_origin": origin,
        "notional_release": release,
        "assumed_drift_bearing": drift_bearing,
        "elapsed_hours": elapsed,
    }
    return env, ais, info
