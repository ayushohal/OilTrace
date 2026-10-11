"""
OILTRACE AI - dashboard.

    streamlit run app.py

This file is presentation only. Every number shown is computed by the
modules under ``modules/``; nothing is hard-coded here.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules import detection, report as report_mod, visualization  # noqa: E402
from modules.attribution import weight_table                        # noqa: E402
from modules.dataset_adapter import inspect_dataset                 # noqa: E402
from modules.geo import BBox                                        # noqa: E402
from modules.pipeline import load_config, run_pipeline              # noqa: E402
from utils.demo_data import (ensure_all_scenes, get_scene,           # noqa: E402
                             synthesise_context)

st.set_page_config(page_title="OILTRACE AI", page_icon="🛰", layout="wide")

# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
CSS = """
<style>
:root{
  --deep:#060d18; --hull:#0d1826; --panel:#101f33; --line:#1d3149;
  --ice:#eaf2fb; --steel:#8fa8c4; --signal:#12e0c8; --amber:#ffb020; --alert:#ff4d5e;
}
.stApp{background:linear-gradient(180deg,#060d18 0%,#0b1524 55%,#081120 100%);color:var(--ice)}
section[data-testid="stSidebar"]{background:var(--hull);border-right:1px solid var(--line)}
h1,h2,h3,h4{color:var(--ice);letter-spacing:-.01em}
.masthead{border:1px solid var(--line);border-left:4px solid var(--signal);
  background:linear-gradient(100deg,rgba(18,224,200,.10),rgba(13,24,38,0) 55%);
  padding:18px 22px;border-radius:4px;margin-bottom:6px}
.masthead h1{margin:0;font-size:30px;font-weight:700}
.masthead p{margin:5px 0 0;color:var(--steel);font-size:14px}
.chain{display:flex;gap:10px;align-items:center;margin:14px 0 20px;flex-wrap:wrap}
.chain span{border:1px solid var(--line);background:var(--panel);color:var(--steel);
  padding:6px 16px;border-radius:2px;font-size:13px;font-weight:600}
.chain span.on{border-color:var(--signal);color:var(--signal)}
.chain b{color:var(--line)}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:4px;
  padding:14px 16px;height:100%}
.kpi .lab{color:var(--steel);font-size:11px;margin-bottom:6px}
.kpi .val{font-size:25px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.15}
.kpi .sub{color:var(--steel);font-size:11px;margin-top:4px}
.kpi.hot{border-color:var(--alert)} .kpi.hot .val{color:var(--alert)}
.kpi.sig .val{color:var(--signal)} .kpi.amb .val{color:var(--amber)}
.rank{background:var(--panel);border:1px solid var(--line);border-radius:4px;
  padding:12px 16px;margin-bottom:8px;display:flex;justify-content:space-between;
  align-items:center;gap:12px}
.rank.first{border-color:var(--alert);
  background:linear-gradient(95deg,rgba(255,77,94,.14),var(--panel) 60%)}
.rank .no{color:var(--steel);font-size:12px;font-weight:700;min-width:26px}
.rank .nm{font-weight:700;font-size:15px}
.rank .mm{color:var(--steel);font-size:11px}
.rank .sc{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums}
.rank.first .sc{color:var(--alert)}
.band{font-size:10px;border:1px solid var(--line);color:var(--steel);
  padding:2px 8px;border-radius:2px}
.stage{font-size:13px;padding:3px 0;color:var(--ice)}
.stage .d{color:var(--steel);font-size:11px}
.why{background:var(--panel);border:1px solid var(--alert);border-left-width:4px;
  border-radius:4px;padding:16px 20px}
.why li{margin:5px 0;font-size:14px}
.caveat{border:1px solid var(--line);background:rgba(255,176,32,.07);
  border-left:3px solid var(--amber);padding:11px 15px;border-radius:3px;
  color:var(--steel);font-size:12px;margin:10px 0}
div[data-testid="stMetricValue"]{font-size:22px}
.stTabs [data-baseweb="tab"]{color:var(--steel)}
.stTabs [aria-selected="true"]{color:var(--signal)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cfg():
    return load_config(ROOT / "config.yaml")


@st.cache_data(show_spinner=False)
def _scenes(_cfg_dict):
    return ensure_all_scenes(_cfg_dict, ROOT)


@st.cache_data(show_spinner=False)
def _dataset(_path: str, _stamp: float):
    return inspect_dataset(_path)


cfg = _cfg()
scene_assets = _scenes(cfg)
kaggle_root = ROOT / "data" / "kaggle_oilspill"
try:
    stamp = max((p.stat().st_mtime for p in kaggle_root.rglob("*") if p.is_file()), default=0.0)
except OSError:
    stamp = 0.0
ds = _dataset(str(kaggle_root), stamp)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Mission control")

    presentation = st.toggle(
        "Presentation demo", value=True,
        help="Runs one of the built-in scenarios deterministically, using its "
             "own imagery, metocean field and AIS extract.",
    )

    scene_ids = [s["id"] for s in cfg["scenes"]]
    scene_names = [s["name"] for s in cfg["scenes"]]

    uploaded = None
    selected_item = None
    scene_id = scene_ids[0]

    if presentation:
        scene_id = scene_ids[scene_names.index(
            st.selectbox("Scenario", scene_names, index=0,
                         help="Each scenario is an independent case with its own "
                              "imagery, metocean data and AIS extract.")
        )]
        st.caption(get_scene(cfg, scene_id)["blurb"])
    else:
        uploaded = st.file_uploader(
            "Upload SAR / oil-spill image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
        )
        if ds.present and ds.items:
            labels = ["(use built-in demo scene)"] + [
                f"{'[mask] ' if i.has_mask else ''}{i.name}" for i in ds.items[:200]
            ]
            choice = st.selectbox("Or pick from the local dataset", labels, index=0)
            if choice != labels[0]:
                selected_item = ds.items[labels.index(choice) - 1]

    st.divider()
    st.markdown("**Detection mode**")
    mode_label = st.radio(
        "Detection mode", ["Automatic (best available)", "Reference mask (demo)",
                           "Image processing only"],
        index=0, label_visibility="collapsed",
    )
    force = {"Automatic (best available)": None,
             "Reference mask (demo)": "reference_mask",
             "Image processing only": "image_processing"}[mode_label]

    st.divider()
    st.markdown("**Synthetic demo georeference**")
    st.caption("Prototype imagery carries no geographic metadata. Pixels are "
               "mapped into this box so the geospatial stages can be demonstrated.")
    sc = get_scene(cfg, scene_id)
    c1, c2 = st.columns(2)
    north = c1.number_input("North", value=float(sc["north"]), step=0.01,
                            format="%.2f", key=f"n_{scene_id}")
    south = c2.number_input("South", value=float(sc["south"]), step=0.01,
                            format="%.2f", key=f"s_{scene_id}")
    west = c1.number_input("West", value=float(sc["west"]), step=0.01,
                           format="%.2f", key=f"w_{scene_id}")
    east = c2.number_input("East", value=float(sc["east"]), step=0.01,
                           format="%.2f", key=f"e_{scene_id}")

    st.divider()
    st.markdown("**Search parameters**")
    radius = st.slider("Candidate radius (km)", 5, 60,
                       int(cfg["ais"]["candidate_radius_km"]), 1)
    window = st.slider("Release window (± h)", 0.5, 6.0,
                       float(cfg["ais"]["release_window_hours"]), 0.5)

    st.divider()
    st.markdown("**Map layers**")
    show_sar = st.checkbox("SAR scene overlay", value=True)
    show_fwd = st.checkbox("Forward drift projection", value=True)
    show_trk = st.checkbox("Vessel trajectories", value=True)

    st.divider()
    st.markdown("**Data sources**")
    adaptive = (not presentation) and (uploaded is not None or selected_item is not None)
    src_sat = ("Local dataset: " + selected_item.name) if selected_item else (
        "Uploaded image" if uploaded else f"Built-in scene: {sc['name']}")
    ctx_src = ("Synthesised around the detected spill" if adaptive
               else "Scenario demo data")
    for k, v in [("Satellite", src_sat),
                 ("Georeferencing", "Synthetic demo bounding box"),
                 ("Environmental", ctx_src),
                 ("AIS", ctx_src)]:
        st.caption(f"**{k}** — {v}")

    if not (north > south and east > west):
        st.error("Bounding box invalid: north must exceed south, east must exceed west.")


# ---------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------
st.markdown(
    '<div class="masthead"><h1>OILTRACE AI</h1>'
    '<p>Satellite oil-spill detection, drift hindcasting and AIS vessel '
    'attribution &mdash; investigation decision support</p></div>',
    unsafe_allow_html=True,
)

run = st.button("Run OILTRACE analysis", type="primary", use_container_width=True)
if run:
    st.session_state["ran"] = True

if not st.session_state.get("ran"):
    st.markdown(
        '<div class="chain"><span>Detect</span><b>&rarr;</b><span>Trace</span>'
        '<b>&rarr;</b><span>Attribute</span><b>&rarr;</b><span>Visualise</span></div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.markdown('<div class="kpi"><div class="lab">Where is the spill?</div>'
               '<div class="val" style="font-size:15px">Segmentation of the SAR scene, '
               'then geometry &mdash; area, centroid, extent.</div></div>',
               unsafe_allow_html=True)
    b.markdown('<div class="kpi"><div class="lab">Where did it come from?</div>'
               '<div class="val" style="font-size:15px">Currents, wind and waves run '
               'backwards to a probable origin and release time.</div></div>',
               unsafe_allow_html=True)
    c.markdown('<div class="kpi"><div class="lab">Which vessel, and why?</div>'
               '<div class="val" style="font-size:15px">AIS tracks are filtered, scored '
               'on five components and ranked, with the reasoning shown.</div></div>',
               unsafe_allow_html=True)
    st.info("Press **Run OILTRACE analysis** to execute the full pipeline.")
    st.stop()


# ---------------------------------------------------------------------
# Resolve inputs and run
# ---------------------------------------------------------------------
assets = scene_assets[scene_id]
image_path = assets["image"]
mask_path = assets["mask"]
env_src, ais_src = assets["environment"], assets["ais"]
obs_time = assets["observation_time"]

source_info = {
    "Scenario": sc["name"],
    "Satellite imagery": f"Built-in synthetic SAR scene ({sc['id']})",
    "Georeferencing": f"Synthetic demo bounding box {south}-{north} N, {west}-{east} E",
    "Environmental data": "Scenario metocean CSV (synthetic)",
    "AIS": "Scenario AIS extract (synthetic, 10 vessels)",
    "Detection mode requested": mode_label,
}

if adaptive:
    # No scenario data accompanies this image, so the metocean field and
    # AIS extract are synthesised around whatever spill is detected.
    env_src, ais_src, obs_time = None, None, None
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / f"oiltrace_{uploaded.name}"
        tmp.write_bytes(uploaded.getbuffer())
        image_path, mask_path = tmp, None
        source_info["Satellite imagery"] = f"User upload: {uploaded.name}"
    else:
        image_path = selected_item.image_path
        mask_path = selected_item.mask_path
        source_info["Satellite imagery"] = f"Local dataset: {selected_item.image_path.name}"
    source_info["Environmental data"] = "Synthesised around the detected spill"
    source_info["AIS"] = "Synthesised around the detected spill"

bbox = BBox(north=north, south=south, west=west, east=east)
run_cfg = {**cfg, "ais": {**cfg["ais"], "candidate_radius_km": radius,
                          "release_window_hours": window}}

with st.spinner("Running detection, drift hindcast and attribution…"):
    res = run_pipeline(
        run_cfg, image_path, mask_path, env_src, ais_src, bbox,
        observation_time=obs_time, force_detection_mode=force,
        source_info=source_info,
        context_builder=(lambda spill: synthesise_context(run_cfg, spill, bbox))
        if adaptive else None,
    )
    res.search_radius_km = radius

if res.error and res.spill is None:
    st.error(f"Analysis could not complete: {res.error}")
    st.markdown("**Pipeline status**")
    for s in res.stages:
        st.markdown(f'<div class="stage">{"✓" if s.ok else "✗"} {s.label} '
                    f'<span class="d">{s.detail}</span></div>', unsafe_allow_html=True)
    st.stop()

sp, dr, filt = res.spill, res.drift, res.filtering
top = res.top

st.caption(f"Scenario: **{source_info.get('Scenario', 'Adaptive')}** · "
           f"observation {sp.observation_time.strftime('%d %b %Y %H:%M')} UTC"
           if not res.context_info else
           f"Adaptive run · observation {sp.observation_time.strftime('%d %b %Y %H:%M')} UTC")
st.markdown(
    '<div class="chain"><span class="on">Detect</span><b>&rarr;</b>'
    '<span class="on">Trace</span><b>&rarr;</b><span class="on">Attribute</span>'
    '<b>&rarr;</b><span class="on">Visualise</span></div>',
    unsafe_allow_html=True,
)

if res.context_info:
    ci = res.context_info
    st.markdown(
        f'<div class="caveat"><b>Adaptive context.</b> This image carries no '
        f'accompanying environmental or AIS data, so both were synthesised around '
        f'the spill the system detected — assuming a {ci["assumed_drift_bearing"]:.0f}° '
        f'drift set over a {ci["elapsed_hours"]:.2f} h window. This demonstrates the '
        f'full pipeline mechanism; it is not a real investigation, and the vessels '
        f'shown are generated, not observed.</div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------
k = st.columns(5)
k[0].markdown(f'<div class="kpi sig"><div class="lab">Estimated prototype area</div>'
              f'<div class="val">{sp.area_km2:.1f} km²</div>'
              f'<div class="sub">{sp.length_km:.1f} × {sp.width_km:.2f} km</div></div>',
              unsafe_allow_html=True)
k[1].markdown(f'<div class="kpi sig"><div class="lab">Detection confidence</div>'
              f'<div class="val">{sp.confidence * 100:.0f}%</div>'
              f'<div class="sub">{sp.detection_mode}</div></div>', unsafe_allow_html=True)
k[2].markdown(f'<div class="kpi amb"><div class="lab">Probable origin</div>'
              f'<div class="val" style="font-size:19px">{dr.origin_lat:.4f} N<br>'
              f'{dr.origin_lon:.4f} E</div>'
              f'<div class="sub">hindcast {dr.elapsed_hours:.2f} h</div></div>',
              unsafe_allow_html=True)
k[3].markdown(f'<div class="kpi amb"><div class="lab">Estimated release</div>'
              f'<div class="val">{dr.release_time.strftime("%H:%M")}</div>'
              f'<div class="sub">UTC, {dr.release_time.strftime("%d %b %Y")}</div></div>',
              unsafe_allow_html=True)
if top:
    k[4].markdown(f'<div class="kpi hot"><div class="lab">Highest correlated candidate</div>'
                  f'<div class="val" style="font-size:19px">{top.name}</div>'
                  f'<div class="sub">{top.total_score:.1f}/100 · {top.confidence_band} '
                  f'correlation</div></div>', unsafe_allow_html=True)
else:
    k[4].markdown('<div class="kpi"><div class="lab">Candidates</div>'
                  '<div class="val" style="font-size:19px">None</div>'
                  '<div class="sub">No vessel passed the filters</div></div>',
                  unsafe_allow_html=True)

st.markdown('<div class="caveat">Scores express analytical correlation and '
            'investigation priority. They are not evidence that any vessel caused '
            'the discharge. Geographic values derive from a synthetic demonstration '
            'bounding box and are not survey grade.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Map + ranking
# ---------------------------------------------------------------------
left, right = st.columns([2.05, 1], gap="medium")

with left:
    st.subheader("Geospatial picture")
    try:
        from streamlit_folium import st_folium

        fmap = visualization.build_map(res, bbox, show_sar, show_fwd, show_trk)
        st_folium(fmap, height=560, use_container_width=True, returned_objects=[])
    except Exception as exc:
        st.warning(f"Interactive map unavailable ({exc}). Showing the static overlay.")
        st.image(visualization.overlay_image(res.image, res.detection.mask),
                 use_container_width=True)
    st.caption("Teal — detected slick and centroid · Amber — hindcast path, probable "
               "origin and search radius · Blue — forward projection · Red — highest "
               "correlated candidate · Grey — other and excluded vessels. Click a track "
               "for its full scoring breakdown.")

with right:
    st.subheader("Candidate vessel ranking")
    if not res.ranking:
        st.info("No vessels satisfied both the spatial and temporal filters. "
                "Widen the search parameters in the sidebar.")
    for v in res.ranking:
        st.markdown(
            f'<div class="rank {"first" if v.rank == 1 else ""}">'
            f'<div class="no">#{v.rank}</div>'
            f'<div style="flex:1"><div class="nm">{v.name}</div>'
            f'<div class="mm">MMSI {v.mmsi} · {v.vessel_type}</div></div>'
            f'<div style="text-align:right"><div class="sc">{v.total_score:.1f}</div>'
            f'<div class="band">{v.confidence_band}</div></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Filtering funnel**")
    st.markdown(
        f"- AIS extract: **{filt.total}** vessels\n"
        f"- After spatial filter ({radius} km): **{filt.after_spatial}**\n"
        f"- After temporal filter (±{window:g} h): **{filt.after_temporal}**\n"
        f"- Final candidates: **{len(res.ranking)}**"
    )

# ---------------------------------------------------------------------
# Why #1
# ---------------------------------------------------------------------
if top:
    st.subheader(f"Why was {top.name} ranked #1?")
    bullets = "".join(f"<li>{r}</li>" for r in top.reasons[:-1])
    st.markdown(f'<div class="why"><ul>{bullets}</ul>'
                f'<p style="margin:12px 0 0;color:#8fa8c4;font-size:12.5px">'
                f'{top.reasons[-1]}</p></div>', unsafe_allow_html=True)

    cols = st.columns(5)
    labels = {"proximity": "Proximity", "temporal": "Temporal",
              "trajectory": "Trajectory", "behaviour": "Behaviour",
              "ais_continuity": "AIS continuity"}
    for col, (key, lab) in zip(cols, labels.items()):
        col.markdown(
            f'<div class="kpi"><div class="lab">{lab} '
            f'({float(cfg["attribution"]["weights"][key]) * 100:.0f}%)</div>'
            f'<div class="val">{top.components[key]:.0f}</div>'
            f'<div class="sub">contributes {top.contributions[key]:.1f} pts</div></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# Detail tabs
# ---------------------------------------------------------------------
st.subheader("Technical detail")
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Satellite analysis", "Pipeline status", "Spill & drift", "Vessels",
     "Dataset & provenance", "Report"]
)

with t1:
    c = st.columns(3)
    c[0].image(res.image, caption="Source image", use_container_width=True, clamp=True)
    c[1].image(res.detection.mask.astype(np.uint8) * 255,
               caption=f"Detected mask — {res.detection.mode}", use_container_width=True)
    c[2].image(visualization.overlay_image(res.image, res.detection.mask),
               caption="Detection overlay", use_container_width=True)

    d = res.detection
    st.markdown(f"**Detection mode:** {d.mode}")
    if not d.ai_generated:
        st.warning("Demo segmentation using the provided reference mask. "
                   "This mask was **not** generated by a model.")
    for n in d.notes:
        st.caption(f"· {n}")

    st.markdown("**Confidence components** — computed from the pixels, not assigned")
    cc = st.columns(3)
    for col, (key, lab, why) in zip(cc, [
        ("contrast", "Contrast", "Oil damps capillary waves, so it returns less energy"),
        ("homogeneity", "Homogeneity", "A slick is smoother than wind-roughened sea"),
        ("shape", "Elongation", "Slicks are elongated; compact blobs are look-alikes"),
    ]):
        col.markdown(f'<div class="kpi"><div class="lab">{lab}</div>'
                     f'<div class="val">{d.diagnostics[key] * 100:.0f}</div>'
                     f'<div class="sub">{why}</div></div>', unsafe_allow_html=True)

    if res.metrics:
        st.markdown("**Dataset reference evaluation** — predicted mask against the "
                    "dataset's ground truth")
        m = res.metrics
        mc = st.columns(4)
        for col, key in zip(mc, ["iou", "dice", "precision", "recall"]):
            col.metric(key.upper() if key == "iou" else key.title(), f"{m[key]:.3f}")
        if res.reference_mask is not None:
            st.image(
                visualization.comparison_image(res.detection.mask, res.reference_mask),
                caption="Green — agreement · Red — false positive · Blue — missed",
                use_container_width=True,
            )
    else:
        st.caption("Segmentation metrics are only computed when an independent "
                   "ground-truth mask exists. None are shown here, rather than "
                   "reporting a meaningless self-comparison.")

with t2:
    for s in res.stages:
        st.markdown(f'<div class="stage">{"✓" if s.ok else "✗"} {s.label} '
                    f'<span class="d">— {s.detail}</span></div>', unsafe_allow_html=True)

with t3:
    a, b = st.columns(2)
    with a:
        st.markdown("**Spill object**")
        st.json({k: v for k, v in sp.to_dict().items()
                 if k not in ("polygon", "polygon_pixels")})
    with b:
        st.markdown("**Drift reconstruction**")
        st.json({
            "probable_origin": [dr.origin_lat, dr.origin_lon],
            "estimated_release_utc": dr.release_time.isoformat().replace("+00:00", "Z"),
            "hindcast_hours": dr.elapsed_hours,
            "mean_drift_bearing_deg": dr.drift_bearing,
            "mean_drift_speed_ms": dr.drift_speed_ms,
            "mean_conditions": {k: round(v, 4) for k, v in dr.mean_conditions.items()},
        })
        for n in dr.notes:
            st.caption(f"· {n}")
    st.markdown("**Forward projection**")
    st.dataframe(pd.DataFrame([
        {"Horizon": f"+{h} h",
         "Time (UTC)": dr.forward_paths[h][-1][2].strftime("%d %b %H:%M"),
         "Latitude": round(dr.forward_paths[h][-1][0], 4),
         "Longitude": round(dr.forward_paths[h][-1][1], 4)}
        for h in sorted(dr.forward_paths)
    ]), hide_index=True, use_container_width=True)
    st.markdown("**Environmental input (head)**")
    st.dataframe(res.env_df.head(12), hide_index=True, use_container_width=True)

with t4:
    st.markdown("**Scoring weights**")
    st.dataframe(pd.DataFrame(weight_table(cfg["attribution"])),
                 hide_index=True, use_container_width=True)
    if res.ranking:
        st.markdown("**Candidate scores**")
        st.dataframe(pd.DataFrame([{
            "#": v.rank, "Vessel": v.name, "MMSI": v.mmsi, "Type": v.vessel_type,
            "Distance (km)": v.trajectory.distance_km,
            "Δt (h)": v.trajectory.time_delta_hours,
            "Course (°)": v.trajectory.course_bearing,
            "Proximity": v.components["proximity"],
            "Temporal": v.components["temporal"],
            "Trajectory": v.components["trajectory"],
            "Behaviour": v.components["behaviour"],
            "AIS continuity": v.components["ais_continuity"],
            "Score": v.total_score, "Correlation": v.confidence_band,
        } for v in res.ranking]), hide_index=True, use_container_width=True)

        for v in res.ranking:
            with st.expander(f"#{v.rank} {v.name} — behavioural indicators"):
                b = v.behaviour
                bc = st.columns(5)
                for col, (lab, val) in zip(bc, [
                    ("Speed change", b.speed_change_score),
                    ("Heading change", b.heading_change_score),
                    ("Loitering", b.loitering_score),
                    ("Route deviation", b.route_deviation_score),
                    ("AIS gap", b.ais_gap_score),
                ]):
                    col.metric(lab, f"{val:.0f}")
                st.caption(f"Max speed change {b.max_speed_drop_kn:.1f} kn · "
                           f"max turn {b.max_heading_change_deg:.0f}° · "
                           f"min speed {b.min_speed_kn:.1f} kn · "
                           f"longest AIS gap {b.max_gap_minutes:.0f} min")
                for i in b.indicators:
                    st.markdown(f"- {i}")
                st.caption("Behavioural indicators describe motion patterns only. "
                           "Vessels slow, turn and lose AIS coverage for many lawful "
                           "reasons; an anomaly is not evidence of illegal discharge.")

    if filt.rejected:
        st.markdown("**Excluded vessels**")
        st.dataframe(pd.DataFrame([{
            "Vessel": t.name, "MMSI": t.mmsi,
            "Closest approach (km)": round(t.closest_distance_km, 2),
            "At (UTC)": t.closest_time.strftime("%H:%M"),
            "Δt (h)": round(t.time_delta_hours, 2), "Excluded because": r,
        } for t, r in filt.rejected]), hide_index=True, use_container_width=True)

with t5:
    st.markdown("**Local dataset inspection** — `data/kaggle_oilspill/`")
    if ds.present:
        i = st.columns(4)
        i[0].metric("Images", ds.image_count)
        i[1].metric("Masks", ds.mask_count)
        i[2].metric("Matched pairs", ds.paired_count)
        i[3].metric("Files scanned", ds.total_files)
        st.markdown(f"**Detected type:** {ds.dataset_type_label}")
        dd = st.columns(3)
        dd[0].markdown("**Formats**"); dd[0].write(ds.formats or "—")
        dd[1].markdown("**Pixel modes**"); dd[1].write(ds.mode_summary or "—")
        dd[2].markdown("**Dimensions**"); dd[2].write(ds.dimensions or "—")
        if ds.class_labels:
            st.markdown("**Folder labels found:** " + ", ".join(ds.class_labels))
        st.markdown("**Directories:** " + ", ".join(ds.directories[:12]))
    else:
        st.info("No external dataset detected. The prototype is running on its "
                "built-in synthetic scene.")
    for n in ds.notes:
        st.caption(f"· {n}")

    st.markdown("**Prototype vs production**")
    st.dataframe(pd.DataFrame([
        {"Stage": "Satellite imagery", "Prototype": "Local image dataset / synthetic scene",
         "Production": "Georeferenced Sentinel-1 SAR ingestion"},
        {"Stage": "Georeferencing", "Prototype": "Synthetic demo bounding box",
         "Production": "Satellite geospatial metadata (GeoTIFF affine)"},
        {"Stage": "Segmentation", "Prototype": "Reference mask / deterministic image processing",
         "Production": "Trained U-Net on VV+VH polarisation"},
        {"Stage": "Environmental", "Prototype": "Synthetic metocean CSV",
         "Production": "Operational current, wind and wave reanalysis"},
        {"Stage": "AIS", "Prototype": "Synthetic 10-vessel extract",
         "Production": "Historical AIS database (terrestrial + satellite)"},
        {"Stage": "Drift", "Prototype": "Simplified vector advection",
         "Production": "Lagrangian particle model with weathering and uncertainty"},
        {"Stage": "Attribution", "Prototype": "Interpretable weighted scoring",
         "Production": "ML ranking model trained on confirmed cases"},
    ]), hide_index=True, use_container_width=True)

with t6:
    st.markdown("Generates a self-contained HTML report. It opens offline in any "
                "browser and prints to PDF.")
    html_doc = report_mod.build_report(res, run_cfg, {**source_info, **res.source_info})
    st.download_button(
        "Generate investigation report",
        data=html_doc.encode("utf-8"),
        file_name=f"OILTRACE_{sp.spill_id}_{dr.release_time.strftime('%Y%m%d')}.html",
        mime="text/html", type="primary", use_container_width=True,
    )
    st.caption(report_mod.DISCLAIMER)

st.divider()
st.caption("OILTRACE AI — prototype for Smart India Hackathon. Detection, drift and "
           "attribution outputs are analytical decision support requiring human "
           "verification.")
