"""
Geospatial helpers for OILTRACE AI.

The prototype imagery has no georeferencing, so `pixel_to_geo` maps pixel
coordinates proportionally into a configurable synthetic bounding box.
In production this whole module is replaced by the GeoTIFF affine
transform carried in Sentinel-1 metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KM_PER_DEG_LAT = 110.574


def km_per_deg_lon(latitude: float) -> float:
    """Length of one degree of longitude, in km, at the given latitude."""
    return 111.320 * math.cos(math.radians(latitude))


@dataclass(frozen=True)
class BBox:
    """Synthetic demonstration bounding box (degrees)."""

    north: float
    south: float
    west: float
    east: float

    @property
    def lat_span(self) -> float:
        return self.north - self.south

    @property
    def lon_span(self) -> float:
        return self.east - self.west

    @property
    def center(self) -> tuple[float, float]:
        return ((self.north + self.south) / 2.0, (self.west + self.east) / 2.0)

    @property
    def width_km(self) -> float:
        return self.lon_span * km_per_deg_lon(self.center[0])

    @property
    def height_km(self) -> float:
        return self.lat_span * KM_PER_DEG_LAT

    def folium_bounds(self) -> list[list[float]]:
        return [[self.south, self.west], [self.north, self.east]]

    @classmethod
    def from_config(cls, cfg: dict) -> "BBox":
        return cls(
            north=float(cfg["north"]),
            south=float(cfg["south"]),
            west=float(cfg["west"]),
            east=float(cfg["east"]),
        )


# ---------------------------------------------------------------------
# Pixel <-> geographic transforms
# ---------------------------------------------------------------------
def pixel_to_geo(
    x: float, y: float, image_width: int, image_height: int, bbox: BBox
) -> tuple[float, float]:
    """Map a pixel (x right, y down) to (latitude, longitude).

    Image dimensions are always taken from the actual image; nothing is
    assumed about resolution.
    """
    lon = bbox.west + (float(x) / float(image_width)) * bbox.lon_span
    lat = bbox.north - (float(y) / float(image_height)) * bbox.lat_span
    return lat, lon


def geo_to_pixel(
    lat: float, lon: float, image_width: int, image_height: int, bbox: BBox
) -> tuple[float, float]:
    """Inverse of :func:`pixel_to_geo`."""
    x = (lon - bbox.west) / bbox.lon_span * image_width
    y = (bbox.north - lat) / bbox.lat_span * image_height
    return x, y


def pixel_area_km2(image_width: int, image_height: int, bbox: BBox) -> float:
    """Ground area represented by a single pixel under the synthetic bbox."""
    return (bbox.width_km / image_width) * (bbox.height_km / image_height)


# ---------------------------------------------------------------------
# Distance / bearing
# ---------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2, degrees from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def offset_km(lat: float, lon: float, east_km: float, north_km: float) -> tuple[float, float]:
    """Displace a position by an east/north offset in kilometres."""
    new_lat = lat + north_km / KM_PER_DEG_LAT
    new_lon = lon + east_km / km_per_deg_lon(lat)
    return new_lat, new_lon


def destination(lat: float, lon: float, bearing: float, distance_km: float) -> tuple[float, float]:
    """Position reached travelling `distance_km` along `bearing` from a point."""
    east = distance_km * math.sin(math.radians(bearing))
    north = distance_km * math.cos(math.radians(bearing))
    return offset_km(lat, lon, east, north)


def angular_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings (0-180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def cosine_alignment(bearing_a: float, bearing_b: float) -> float:
    """Cosine similarity of two compass directions, in [-1, 1]."""
    return math.cos(math.radians(angular_difference(bearing_a, bearing_b)))
