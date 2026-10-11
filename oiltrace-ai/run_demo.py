"""
Headless end-to-end check across the whole scene library.

    python run_demo.py            # run every scene
    python run_demo.py --force    # regenerate all demo assets first
    python run_demo.py --scene gulf_of_kutch
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.geo import haversine_km                       # noqa: E402
from modules.pipeline import load_config, run_pipeline     # noqa: E402
from utils.demo_data import ensure_scene                   # noqa: E402


def run_scene(cfg, scene, force):
    p = ensure_scene(cfg, scene, ROOT, force=force)
    if p["created"]:
        print(f"  generated: {', '.join(p['created'])}")

    res = run_pipeline(cfg, p["image"], p["mask"], p["environment"], p["ais"],
                       p["bbox"], observation_time=p["observation_time"])

    for s in res.stages:
        print(f"  {'[ok]' if s.ok else '[!!]'} {s.label:<34} {s.detail}")
    if res.error:
        print("  ERROR:", res.error)
        return False

    sp, dr, f = res.spill, res.drift, res.filtering
    print(f"\n  spill      {sp.area_km2:.2f} km2  |  {sp.length_km:.1f} x {sp.width_km:.2f} km"
          f"  |  confidence {sp.confidence * 100:.1f}%")
    print(f"  centroid   {sp.latitude:.4f} N, {sp.longitude:.4f} E "
          f"at {sp.observation_time.strftime('%H:%M')} UTC")
    print(f"  origin     {dr.origin_lat:.4f} N, {dr.origin_lon:.4f} E  "
          f"release {dr.release_time.strftime('%H:%M')} UTC  "
          f"(hindcast {dr.elapsed_hours:.2f} h, bearing {dr.drift_bearing:.0f} deg)")
    print(f"  funnel     {f.total} -> {f.after_spatial} -> {f.after_temporal}")
    for v in res.ranking:
        print(f"    #{v.rank} {v.name:<16} {v.total_score:6.1f}/100  [{v.confidence_band}]")

    err = haversine_km(dr.origin_lat, dr.origin_lon, *scene["truth_origin"])
    ok_area = abs(sp.area_km2 - scene["slick_area_km2"]) <= max(
        0.6, 0.06 * scene["slick_area_km2"])
    ok_origin = err <= 0.8
    ok_release = (dr.release_time.strftime("%H:%M")
                  == scene["truth_release_time_utc"][11:16])
    ok_top = res.top and res.top.name == scene["fleet"][0][0]
    ok_funnel = (f.total, f.after_spatial, f.after_temporal) == (10, 6, 4)

    print(f"\n  check  area {'OK' if ok_area else 'FAIL'} ({sp.area_km2:.2f} vs "
          f"{scene['slick_area_km2']})  |  origin {'OK' if ok_origin else 'FAIL'} "
          f"({err * 1000:.0f} m)  |  release {'OK' if ok_release else 'FAIL'}  |  "
          f"funnel {'OK' if ok_funnel else 'FAIL'}  |  "
          f"top vessel {'OK' if ok_top else 'FAIL'}")
    return all([ok_area, ok_origin, ok_release, ok_top, ok_funnel])


def main(argv) -> int:
    cfg = load_config(ROOT / "config.yaml")
    force = "--force" in argv
    only = None
    if "--scene" in argv:
        only = argv[argv.index("--scene") + 1]

    scenes = [s for s in cfg["scenes"] if only is None or s["id"] == only]
    if not scenes:
        print(f"No scene named {only!r}. Available: "
              f"{', '.join(s['id'] for s in cfg['scenes'])}")
        return 1

    ok = True
    for scene in scenes:
        print("\n" + "=" * 78)
        print(f"SCENE: {scene['name']}  [{scene['id']}]")
        print("=" * 78)
        ok &= run_scene(cfg, scene, force)

    print("\n" + "=" * 78)
    print("ALL SCENES PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
