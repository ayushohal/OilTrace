"""
Deterministic synthetic SAR scene builder.

Produces a SAR-like grayscale image containing a dark, elongated oil slick
plus a matching ground-truth mask. Used only when no external dataset is
present. Everything is driven by a fixed seed so a live demo is repeatable.

The slick is drawn in GEOGRAPHIC space (an ellipse of known semi-axes,
oriented along the designed drift bearing) and then rasterised through the
same pixel->geo transform the pipeline uses. The measured area therefore
lands on the intended value because the *geometry* was designed for it,
not because any number was written into the output.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from modules.geo import BBox, KM_PER_DEG_LAT, geo_to_pixel, km_per_deg_lon


def _speckle(rng: np.random.Generator, shape: tuple[int, int], looks: int = 4) -> np.ndarray:
    """Multiplicative speckle: mean-1 Gamma noise, as in multi-look SAR."""
    return rng.gamma(shape=looks, scale=1.0 / looks, size=shape).astype(np.float32)


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(a, sigma=sigma)


def build_scene(
    bbox: BBox,
    width: int,
    height: int,
    centroid_latlon: tuple[float, float],
    slick_length_km: float,
    slick_area_km2: float,
    drift_bearing_deg: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sar_image_float01, ground_truth_mask_bool)."""
    rng = np.random.default_rng(seed)

    # ---- 1. Slick geometry in kilometres -----------------------------
    semi_major = slick_length_km / 2.0
    # Analytic starting point; calibrated below so the *rasterised* mask
    # (which gains area from the irregular edge and trailing patches)
    # matches the designed footprint rather than the ideal ellipse.
    semi_minor = slick_area_km2 / (math.pi * semi_major)

    clat, clon = centroid_latlon
    kx = km_per_deg_lon(clat)      # km per degree longitude
    ky = KM_PER_DEG_LAT            # km per degree latitude

    # Pixel grid -> geographic -> local km offsets from the centroid
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    lons = bbox.west + xs * bbox.lon_span
    lats = bbox.north - ys * bbox.lat_span
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    b = math.radians(drift_bearing_deg)

    # Irregular boundary from low-frequency noise (normalised to unit
    # variance so the perturbation does not bias the footprint).
    edge = _smooth(rng.normal(0.0, 1.0, (height, width)).astype(np.float32), 9.0)
    edge = edge / (np.std(edge) + 1e-9)

    px_km2 = (bbox.width_km / width) * (bbox.height_km / height)

    # Centre offset (km, east/north) applied when re-centring the slick so
    # that its MEASURED centroid lands on the designed point. The trailing
    # patches are asymmetric, so the geometric centre and the centroid of
    # the rasterised mask do not coincide.
    shift = [0.0, 0.0]

    def rasterise(minor: float) -> np.ndarray:
        east_km = (lon_grid - clon) * kx - shift[0]
        north_km = (lat_grid - clat) * ky - shift[1]

        # Rotate into the slick frame (major axis along the drift bearing)
        along = east_km * math.sin(b) + north_km * math.cos(b)
        across = east_km * math.cos(b) - north_km * math.sin(b)
        # Gentle curvature so the slick is not a textbook ellipse
        across = across - 0.055 * (along ** 2) / max(semi_major, 1e-6)

        radial = (along / semi_major) ** 2 + (across / minor) ** 2
        m = radial <= (1.0 + 0.12 * edge)
        # Detached patches trailing the main slick (weathered oil breaks up)
        for frac, scale in ((0.66, 0.15), (0.84, 0.10), (-0.74, 0.12)):
            pl = clat + (semi_major * frac * math.cos(b)) / ky
            po = clon + (semi_major * frac * math.sin(b)) / kx
            pe = (lon_grid - po) * kx
            pn = (lat_grid - pl) * ky
            pa = pe * math.sin(b) + pn * math.cos(b)
            pc = pe * math.cos(b) - pn * math.sin(b)
            patch = (pa / (semi_major * scale)) ** 2 + (pc / (minor * 0.8)) ** 2 <= 1.0
            m |= patch & (radial > 1.05) & (radial < 3.0)
        return m

    def calibrate_area() -> np.ndarray:
        """Bisect the minor axis until the rasterised area hits the target."""
        lo, hi = semi_minor * 0.35, semi_minor * 1.30
        m = rasterise(semi_minor)
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            m = rasterise(mid)
            area = float(m.sum()) * px_km2
            if abs(area - slick_area_km2) < 0.015:
                break
            if area > slick_area_km2:
                hi = mid
            else:
                lo = mid
        return m

    # Alternate area calibration and centroid re-centring until both the
    # footprint and the centre of mass match the design.
    mask = calibrate_area()
    for _ in range(4):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            break
        mx = (xs.mean() + 0.5) / width
        my = (ys.mean() + 0.5) / height
        m_lon = bbox.west + mx * bbox.lon_span
        m_lat = bbox.north - my * bbox.lat_span
        de = (m_lon - clon) * kx
        dn = (m_lat - clat) * ky
        if math.hypot(de, dn) < 0.01:      # within 10 m
            break
        shift[0] -= de
        shift[1] -= dn
        mask = calibrate_area()

    # ---- 2. SAR backscatter ------------------------------------------
    sea_sigma0 = 0.46          # wind-roughened sea surface
    slick_sigma0 = 0.055       # damped capillary waves -> strongly dark
    base = np.full((height, width), sea_sigma0, dtype=np.float32)

    # Large-scale wind streaks across the scene
    wind_field = _smooth(rng.normal(0.0, 1.0, (height, width)).astype(np.float32), 28.0)
    base += 0.055 * wind_field / (np.std(wind_field) + 1e-9)

    base[mask] = slick_sigma0
    # Soft transition zone at the slick edge
    from scipy.ndimage import gaussian_filter

    base = gaussian_filter(base, sigma=1.1)

    img = base * _speckle(rng, (height, width), looks=5)

    # ---- 3. Bright targets: vessels and their wakes -------------------
    for (vlat, vlon, wake_bearing, wake_len_km) in [
        (18.517, 72.612, 62.0, 9.0),
        (18.268, 72.930, 118.0, 7.0),
        (18.470, 72.960, 25.0, 6.0),
        (18.330, 72.560, 300.0, 5.5),
    ]:
        x0, y0 = geo_to_pixel(vlat, vlon, width, height, bbox)
        steps = int(wake_len_km * 30)
        for s in range(steps):
            d = wake_len_km * s / max(steps - 1, 1)
            e = d * math.sin(math.radians(wake_bearing))
            n = d * math.cos(math.radians(wake_bearing))
            lat = vlat + n / ky
            lon = vlon + e / kx
            x, y = geo_to_pixel(lat, lon, width, height, bbox)
            xi, yi = int(round(x)), int(round(y))
            if 1 <= xi < width - 1 and 1 <= yi < height - 1:
                img[yi, xi] = min(1.0, img[yi, xi] + 0.30)
        xi, yi = int(round(x0)), int(round(y0))
        if 2 <= xi < width - 2 and 2 <= yi < height - 2:
            img[yi - 1:yi + 2, xi - 1:xi + 2] = 1.0

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img, mask


def write_scene(
    image: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L").save(image_path)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
