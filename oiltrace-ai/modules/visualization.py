"""
GIS visualisation and image overlays.

Builds the Folium map (spill polygon, centroid, probable origin, drift
paths, vessel tracks) and the raster panels shown in the satellite
analysis section.
"""

from __future__ import annotations

import base64
import io

import folium
import numpy as np
from PIL import Image

from modules.geo import BBox

# Layer palette. The single hot colour is reserved for the top candidate;
# everything else stays cool so the eye goes straight to the ranking.
C_SPILL = "#12E0C8"
C_ORIGIN = "#FFB020"
C_BACK = "#FFB020"
C_FORWARD = "#5AA9E6"
C_VESSEL = "#8FA3BF"
C_TOP = "#FF4D5E"


# ---------------------------------------------------------------------
# Raster panels
# ---------------------------------------------------------------------
def to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a float [0,1] or bool array as PNG bytes."""
    if arr.dtype == bool:
        a = (arr.astype(np.uint8)) * 255
    else:
        a = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def overlay_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint the detected region over the grayscale scene."""
    rgb = np.stack([image, image, image], axis=-1).astype(np.float32)
    tint = np.array([0.07, 0.88, 0.78], dtype=np.float32)   # matches C_SPILL
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * tint

    # Outline the mask so the boundary reads clearly at small sizes.
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cv2.drawContours(out, contours, -1, (255, 210, 60), 2)
    return out


def comparison_image(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Green = agreement, red = false positive, blue = missed."""
    h, w = predicted.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[np.logical_and(predicted, truth)] = (40, 220, 140)
    out[np.logical_and(predicted, ~truth)] = (255, 90, 100)
    out[np.logical_and(~predicted, truth)] = (90, 150, 255)
    return out


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ---------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------
def build_map(
    result,
    bbox: BBox,
    show_sar: bool = True,
    show_forward: bool = True,
    show_tracks: bool = True,
) -> folium.Map:
    spill, dr = result.spill, result.drift
    centre = [spill.latitude, spill.longitude]

    fmap = folium.Map(
        location=centre,
        zoom_start=11,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # --- SAR scene as a georeferenced overlay -------------------------
    if show_sar and result.image is not None:
        folium.raster_layers.ImageOverlay(
            image=_data_uri(to_png_bytes(result.image)),
            bounds=bbox.folium_bounds(),
            opacity=0.55,
            name="SAR scene (synthetic georeference)",
        ).add_to(fmap)

    # --- Spill polygon ------------------------------------------------
    if spill.polygon and len(spill.polygon) > 2:
        folium.Polygon(
            locations=[[la, lo] for la, lo in spill.polygon],
            color=C_SPILL, weight=2, fill=True, fill_color=C_SPILL, fill_opacity=0.35,
            tooltip=(f"{spill.spill_id} - {spill.area_km2:.2f} km2 "
                     f"(estimated prototype area)"),
            name="Detected slick",
        ).add_to(fmap)

    # --- Spill centroid ----------------------------------------------
    folium.CircleMarker(
        location=centre, radius=7, color=C_SPILL, fill=True, fill_opacity=1.0, weight=2,
        tooltip=(f"Observed slick centroid<br>{spill.latitude:.4f} N, "
                 f"{spill.longitude:.4f} E<br>"
                 f"{spill.observation_time.strftime('%H:%M')} UTC"),
    ).add_to(fmap)

    # --- Backward drift path -----------------------------------------
    back = [[la, lo] for (la, lo, _) in dr.backward_path]
    folium.PolyLine(back, color=C_BACK, weight=3, opacity=0.95, dash_array="6,6",
                    tooltip="Backward drift (hindcast to probable origin)").add_to(fmap)

    # --- Probable origin ---------------------------------------------
    folium.Marker(
        location=[dr.origin_lat, dr.origin_lon],
        icon=folium.Icon(color="orange", icon="crosshairs", prefix="fa"),
        tooltip=(f"Probable origin<br>{dr.origin_lat:.4f} N, {dr.origin_lon:.4f} E<br>"
                 f"Estimated release {dr.release_time.strftime('%H:%M')} UTC"),
    ).add_to(fmap)
    folium.Circle(
        location=[dr.origin_lat, dr.origin_lon],
        radius=float(result.search_radius_km) * 1000.0 if hasattr(result, "search_radius_km") else 20000.0,
        color=C_ORIGIN, weight=1, opacity=0.6, fill=True, fill_opacity=0.05,
        tooltip="AIS candidate search radius",
    ).add_to(fmap)

    # --- Forward drift projection ------------------------------------
    if show_forward and dr.forward_paths:
        longest = max(dr.forward_paths)
        path = [[la, lo] for (la, lo, _) in dr.forward_paths[longest]]
        folium.PolyLine(path, color=C_FORWARD, weight=3, opacity=0.85,
                        tooltip=f"Forward drift projection (+{longest} h)").add_to(fmap)
        for hrs in sorted(dr.forward_paths):
            la, lo, t = dr.forward_paths[hrs][-1]
            folium.CircleMarker(
                [la, lo], radius=4, color=C_FORWARD, fill=True, fill_opacity=0.9, weight=1,
                tooltip=f"+{hrs} h  {t.strftime('%d %b %H:%M')} UTC",
            ).add_to(fmap)

    # --- Vessel tracks ------------------------------------------------
    if show_tracks:
        top_mmsi = result.top.mmsi if result.top else None
        ranked = {v.mmsi: v for v in result.ranking}

        for track, reason in result.filtering.rejected:
            folium.PolyLine(
                track.positions, color=C_VESSEL, weight=1.5, opacity=0.35,
                tooltip=f"{track.name} ({track.mmsi}) - excluded: {reason}",
            ).add_to(fmap)

        for track in result.filtering.candidates:
            v = ranked.get(track.mmsi)
            is_top = track.mmsi == top_mmsi
            colour = C_TOP if is_top else C_VESSEL
            popup = _vessel_popup(track, v)
            folium.PolyLine(
                track.positions, color=colour,
                weight=5 if is_top else 2.5,
                opacity=0.95 if is_top else 0.6,
                tooltip=f"#{v.rank} {track.name} - {v.total_score:.1f}/100" if v else track.name,
                popup=folium.Popup(popup, max_width=340),
            ).add_to(fmap)

            la, lo = track.positions[0]
            folium.CircleMarker([la, lo], radius=3, color=colour, fill=True,
                                weight=1, tooltip=f"{track.name} track start").add_to(fmap)

            row = track.records.loc[track.records["distance_to_origin_km"].idxmin()]
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=9 if is_top else 5,
                color=colour, fill=True, fill_opacity=0.95, weight=3 if is_top else 1,
                tooltip=(f"{track.name} closest approach - "
                         f"{track.closest_distance_km:.2f} km at "
                         f"{track.closest_time.strftime('%H:%M')} UTC"),
                popup=folium.Popup(popup, max_width=340),
            ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def _vessel_popup(track, v) -> str:
    if v is None:
        return f"<b>{track.name}</b><br>MMSI {track.mmsi}"
    t, b = v.trajectory, v.behaviour
    reasons = "".join(f"<li>{r}</li>" for r in v.reasons[:-1])
    return f"""
    <div style="font-family:system-ui,sans-serif;font-size:12px;line-height:1.45">
      <div style="font-weight:700;font-size:14px">#{v.rank} {v.name}</div>
      <div style="color:#667;margin-bottom:6px">MMSI {v.mmsi} &middot; {v.vessel_type}</div>
      <table style="border-collapse:collapse">
        <tr><td>Score</td><td style="text-align:right"><b>{v.total_score:.1f}/100</b></td></tr>
        <tr><td>Correlation</td><td style="text-align:right">{v.confidence_band}</td></tr>
        <tr><td>Distance</td><td style="text-align:right">{t.distance_km:.2f} km</td></tr>
        <tr><td>Time difference</td><td style="text-align:right">{t.time_delta_hours:+.2f} h</td></tr>
        <tr><td>Trajectory alignment</td><td style="text-align:right">{t.alignment_score:.0f}/100</td></tr>
        <tr><td>Behavioural anomaly</td><td style="text-align:right">{b.anomaly_score:.0f}/100</td></tr>
        <tr><td>AIS continuity</td><td style="text-align:right">{b.continuity_score:.0f}/100</td></tr>
      </table>
      <div style="margin-top:6px;font-weight:600">Reason for ranking</div>
      <ul style="margin:4px 0 0 16px;padding:0">{reasons}</ul>
    </div>
    """
