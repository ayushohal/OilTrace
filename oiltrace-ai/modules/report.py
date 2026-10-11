"""
Investigation report.

Produces a single self-contained HTML file (no external assets, opens
offline, prints to PDF from any browser). Every section states which
inputs were simulated.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone

from modules.visualization import to_png_bytes, overlay_image

DISCLAIMER = (
    "Prototype analytical result - requires human investigation and "
    "verification. Correlation scores indicate investigation priority only "
    "and are not evidence of responsibility for the discharge."
)


def _img_tag(png: bytes, alt: str) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return (f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" '
            f'style="width:100%;border-radius:6px;border:1px solid #24354d">')


def build_report(result, cfg: dict, source_info: dict) -> str:
    sp, dr, f = result.spill, result.drift, result.filtering
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    env = dr.mean_conditions

    rows = []
    for v in result.ranking:
        t, b = v.trajectory, v.behaviour
        rows.append(f"""
        <tr class="{'top' if v.rank == 1 else ''}">
          <td>{v.rank}</td>
          <td><b>{html.escape(v.name)}</b><br><span class="sub">{html.escape(v.mmsi)}</span></td>
          <td>{html.escape(v.vessel_type)}</td>
          <td class="num">{t.distance_km:.2f}</td>
          <td class="num">{t.time_delta_hours:+.2f}</td>
          <td class="num">{t.alignment_score:.0f}</td>
          <td class="num">{b.anomaly_score:.0f}</td>
          <td class="num">{b.continuity_score:.0f}</td>
          <td class="num"><b>{v.total_score:.1f}</b></td>
          <td>{v.confidence_band}</td>
        </tr>""")

    excluded = "".join(
        f"<li><b>{html.escape(t.name)}</b> ({html.escape(t.mmsi)}) &mdash; {html.escape(r)}</li>"
        for t, r in f.rejected
    ) or "<li>No vessels were excluded.</li>"

    top = result.top
    why = "".join(f"<li>{html.escape(r)}</li>" for r in top.reasons) if top else ""

    panels = ""
    if result.image is not None and result.detection is not None:
        panels = f"""
        <div class="grid3">
          <div>{_img_tag(to_png_bytes(result.image), 'Source scene')}
               <div class="cap">Source image</div></div>
          <div>{_img_tag(to_png_bytes(result.detection.mask), 'Detected mask')}
               <div class="cap">Detected mask</div></div>
          <div>{_img_tag(to_png_bytes(overlay_image(result.image, result.detection.mask) / 255.0),
                         'Detection overlay')}
               <div class="cap">Overlay</div></div>
        </div>"""

    metrics_block = ""
    if result.metrics:
        m = result.metrics
        metrics_block = f"""
        <h3>Dataset reference evaluation</h3>
        <p class="sub">Predicted mask compared against the dataset's ground-truth mask.</p>
        <table class="kv">
          <tr><td>IoU</td><td class="num">{m['iou']:.3f}</td></tr>
          <tr><td>Dice</td><td class="num">{m['dice']:.3f}</td></tr>
          <tr><td>Precision</td><td class="num">{m['precision']:.3f}</td></tr>
          <tr><td>Recall</td><td class="num">{m['recall']:.3f}</td></tr>
        </table>"""

    sources = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in source_info.items()
    )

    weights = "".join(
        f"<tr><td>{html.escape(k.replace('_', ' ').title())}</td>"
        f"<td class='num'>{float(w) * 100:.0f}%</td></tr>"
        for k, w in cfg["attribution"]["weights"].items()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>OILTRACE AI - Investigation Report {html.escape(sp.spill_id)}</title>
<style>
 :root {{ --ink:#0d1826; --line:#d8e0ea; --muted:#5b6b80; --hot:#c02f3d; }}
 * {{ box-sizing:border-box }}
 body {{ font-family:"Segoe UI",system-ui,-apple-system,sans-serif; color:var(--ink);
        margin:0; padding:36px; max-width:1000px; margin-inline:auto; line-height:1.5 }}
 header {{ border-bottom:3px solid var(--ink); padding-bottom:14px; margin-bottom:22px }}
 h1 {{ font-size:26px; margin:0 0 4px }}
 h2 {{ font-size:17px; margin:28px 0 10px; border-bottom:1px solid var(--line);
       padding-bottom:6px }}
 h3 {{ font-size:14px; margin:18px 0 6px }}
 .sub {{ color:var(--muted); font-size:12px }}
 table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:6px }}
 th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left }}
 th {{ background:#f2f5f9; font-size:12px; letter-spacing:.02em }}
 .num {{ text-align:right; font-variant-numeric:tabular-nums }}
 tr.top {{ background:#fff2f3 }}
 tr.top td {{ border-bottom-color:#f0c9cd }}
 .kv td:first-child {{ color:var(--muted); width:60% }}
 .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px }}
 .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:10px }}
 .cap {{ font-size:11px; color:var(--muted); margin-top:4px; text-align:center }}
 .callout {{ border-left:4px solid var(--hot); background:#fff5f6; padding:12px 14px;
             margin:14px 0; font-size:13px }}
 .note {{ border:1px solid var(--line); background:#f8fafc; padding:12px 14px;
          font-size:12px; color:var(--muted); margin-top:26px }}
 ul {{ margin:6px 0 0 18px; padding:0 }} li {{ margin:3px 0; font-size:13px }}
 footer {{ margin-top:30px; padding-top:12px; border-top:1px solid var(--line);
           font-size:11px; color:var(--muted) }}
</style></head><body>

<header>
  <h1>OILTRACE AI &mdash; Investigation Report</h1>
  <div class="sub">Spill {html.escape(sp.spill_id)} &nbsp;|&nbsp; generated {now}</div>
</header>

<div class="callout"><b>{html.escape(DISCLAIMER)}</b></div>

<h2>1. Detected spill</h2>
<div class="grid2">
 <table class="kv">
  <tr><td>Spill identifier</td><td>{html.escape(sp.spill_id)}</td></tr>
  <tr><td>Observation time</td>
      <td>{sp.observation_time.strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
  <tr><td>Centroid</td>
      <td>{sp.latitude:.4f} N, {sp.longitude:.4f} E</td></tr>
  <tr><td>Estimated prototype area</td><td class="num">{sp.area_km2:.2f} km&sup2;</td></tr>
  <tr><td>Perimeter</td><td class="num">{sp.perimeter_km:.2f} km</td></tr>
 </table>
 <table class="kv">
  <tr><td>Length &times; width</td>
      <td class="num">{sp.length_km:.2f} &times; {sp.width_km:.2f} km</td></tr>
  <tr><td>Elongation</td><td class="num">{sp.elongation:.1f} : 1</td></tr>
  <tr><td>Major-axis bearing</td><td class="num">{sp.orientation_bearing:.0f}&deg;</td></tr>
  <tr><td>Detection mode</td><td>{html.escape(sp.detection_mode)}</td></tr>
  <tr><td>Detection confidence</td><td class="num">{sp.confidence * 100:.1f}%</td></tr>
 </table>
</div>
<p class="sub">{html.escape(sp.geographic_basis)}. Not survey-grade.</p>
{panels}
{metrics_block}

<h2>2. Drift reconstruction and probable origin</h2>
<div class="grid2">
 <table class="kv">
  <tr><td>Probable origin</td>
      <td>{dr.origin_lat:.4f} N, {dr.origin_lon:.4f} E</td></tr>
  <tr><td>Estimated release time</td>
      <td>{dr.release_time.strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
  <tr><td>Hindcast duration</td><td class="num">{dr.elapsed_hours:.2f} h</td></tr>
 </table>
 <table class="kv">
  <tr><td>Mean drift bearing</td><td class="num">{dr.drift_bearing:.1f}&deg;</td></tr>
  <tr><td>Mean drift speed</td><td class="num">{dr.drift_speed_ms:.3f} m/s</td></tr>
  <tr><td>Surface current</td>
      <td class="num">{env.get('current_u', 0):.3f} / {env.get('current_v', 0):.3f} m/s</td></tr>
  <tr><td>Wind</td>
      <td class="num">{env.get('wind_u', 0):.2f} / {env.get('wind_v', 0):.2f} m/s</td></tr>
  <tr><td>Wave</td>
      <td class="num">{env.get('wave_speed', 0):.2f} m/s @ {env.get('wave_direction', 0):.0f}&deg;</td></tr>
 </table>
</div>
<ul>{''.join(f'<li>{html.escape(n)}</li>' for n in dr.notes)}</ul>

<h2>3. AIS candidate search</h2>
<p class="sub">Vessels are matched to the probable origin and release time, never to the
satellite image directly.</p>
<table class="kv">
 <tr><td>Vessels in the AIS extract</td><td class="num">{f.total}</td></tr>
 <tr><td>After spatial filter
     ({cfg['ais']['candidate_radius_km']:.0f} km)</td><td class="num">{f.after_spatial}</td></tr>
 <tr><td>After temporal filter
     (&plusmn;{cfg['ais']['release_window_hours']:.0f} h)</td><td class="num">{f.after_temporal}</td></tr>
</table>
<h3>Excluded vessels</h3>
<ul>{excluded}</ul>

<h2>4. Attribution scoring</h2>
<table><tr><th>Component</th><th class="num">Weight</th></tr>{weights}</table>
<table>
 <tr><th>#</th><th>Vessel</th><th>Type</th><th class="num">Dist (km)</th>
     <th class="num">&Delta;t (h)</th><th class="num">Traj</th><th class="num">Behav</th>
     <th class="num">AIS</th><th class="num">Score</th><th>Correlation</th></tr>
 {''.join(rows)}
</table>

<h2>5. Highest correlated candidate</h2>
<p><b>{html.escape(top.name) if top else 'None'}</b>
   {f'(MMSI {html.escape(top.mmsi)}) &mdash; {top.total_score:.1f}/100' if top else ''}</p>
<ul>{why}</ul>

<h2>6. Data provenance</h2>
<table><tr><th>Input</th><th>Source</th></tr>{sources}</table>

<div class="note">
 <b>Prototype limitations.</b> Geographic coordinates derive from a synthetic
 demonstration bounding box, not satellite georeferencing. The drift model is a
 simplified vector advection scheme without weathering, turbulent diffusion or an
 uncertainty envelope. Attribution uses transparent fixed weights rather than a
 model trained on confirmed cases. Behavioural indicators describe motion patterns
 only; vessels slow, turn and lose AIS coverage for many lawful reasons.
</div>

<footer>OILTRACE AI prototype &mdash; Smart India Hackathon submission.
 {html.escape(DISCLAIMER)}</footer>
</body></html>"""
