"""
Spill geometry.

Turns a binary mask into the intermediate "Spill Object": polygon,
centroid, area, perimeter, length, width and shape descriptors, expressed
first in pixels and then converted into the synthetic demonstration
geography.

All geographic values are labelled "Prototype geographic estimation".
Nothing here is survey grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime

import cv2
import numpy as np

from modules.geo import BBox, pixel_area_km2, pixel_to_geo, KM_PER_DEG_LAT, km_per_deg_lon


@dataclass
class SpillObject:
    spill_id: str
    observation_time: datetime
    latitude: float
    longitude: float
    area_km2: float
    perimeter_km: float
    length_km: float
    width_km: float
    elongation: float
    orientation_bearing: float
    pixel_area: int
    confidence: float
    detection_mode: str
    bounding_box: dict = field(default_factory=dict)
    polygon: list = field(default_factory=list)      # [(lat, lon), ...]
    polygon_pixels: list = field(default_factory=list)
    geographic_basis: str = "Prototype geographic estimation - synthetic demo bounding box"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observation_time"] = self.observation_time.isoformat().replace("+00:00", "Z")
        return d


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def analyse_spill(
    mask: np.ndarray,
    bbox: BBox,
    observation_time: datetime,
    confidence: float,
    detection_mode: str,
    spill_id: str = "SPILL-001",
) -> SpillObject | None:
    """Extract the Spill Object from a detection mask."""
    if mask is None or mask.sum() == 0:
        return None

    h, w = mask.shape
    px_area = pixel_area_km2(w, h, bbox)

    pixel_count = int(mask.sum())
    area_km2 = pixel_count * px_area

    # --- centroid (pixel -> geo) ---
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    lat, lon = pixel_to_geo(cx, cy, w, h, bbox)

    # --- orientation and extent via principal component analysis ---
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)])
    cov = np.cov(pts)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    major = evecs[:, 0]

    centred = np.stack([xs - cx, ys - cy])
    proj_major = major @ centred
    proj_minor = evecs[:, 1] @ centred

    # Scale factors differ slightly between axes under the synthetic bbox.
    km_x = bbox.width_km / w
    km_y = bbox.height_km / h
    # Effective km-per-pixel along each principal direction.
    km_major = float(np.hypot(major[0] * km_x, major[1] * km_y))
    km_minor = float(np.hypot(evecs[0, 1] * km_x, evecs[1, 1] * km_y))

    length_km = float(proj_major.max() - proj_major.min()) * km_major
    width_km = float(proj_minor.max() - proj_minor.min()) * km_minor
    elongation = length_km / max(width_km, 1e-6)

    # Compass bearing of the major axis (image y grows southward).
    east = major[0] * km_x
    north = -major[1] * km_y
    bearing = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    if bearing > 180.0:
        bearing -= 180.0

    # --- perimeter and polygon ---
    contour = _largest_contour(mask)
    perimeter_km = 0.0
    polygon: list[tuple[float, float]] = []
    polygon_pixels: list[tuple[float, float]] = []
    if contour is not None:
        eps = 0.004 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        prev = None
        for (px, py) in approx:
            plat, plon = pixel_to_geo(float(px), float(py), w, h, bbox)
            polygon.append((plat, plon))
            polygon_pixels.append((float(px), float(py)))
            if prev is not None:
                dlat = (plat - prev[0]) * KM_PER_DEG_LAT
                dlon = (plon - prev[1]) * km_per_deg_lon(plat)
                perimeter_km += float(np.hypot(dlat, dlon))
            prev = (plat, plon)
        if len(polygon) > 2:
            dlat = (polygon[0][0] - polygon[-1][0]) * KM_PER_DEG_LAT
            dlon = (polygon[0][1] - polygon[-1][1]) * km_per_deg_lon(polygon[0][0])
            perimeter_km += float(np.hypot(dlat, dlon))

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    nw = pixel_to_geo(x0, y0, w, h, bbox)
    se = pixel_to_geo(x1, y1, w, h, bbox)

    return SpillObject(
        spill_id=spill_id,
        observation_time=observation_time,
        latitude=round(lat, 4),
        longitude=round(lon, 4),
        area_km2=round(area_km2, 2),
        perimeter_km=round(perimeter_km, 2),
        length_km=round(length_km, 2),
        width_km=round(width_km, 2),
        elongation=round(elongation, 2),
        orientation_bearing=round(float(bearing), 1),
        pixel_area=pixel_count,
        confidence=round(float(confidence), 4),
        detection_mode=detection_mode,
        bounding_box={
            "north": round(nw[0], 4),
            "west": round(nw[1], 4),
            "south": round(se[0], 4),
            "east": round(se[1], 4),
            "pixels": [x0, y0, x1, y1],
        },
        polygon=polygon,
        polygon_pixels=polygon_pixels,
    )
