"""
Specification audit.

Checks the built prototype against the OILTRACE AI requirements and the
target demo-story values. Run before a presentation:

    python verify.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.geo import BBox, haversine_km, pixel_to_geo, geo_to_pixel
from modules.pipeline import load_config, run_pipeline
from modules import detection
from utils.demo_data import ensure_scene, synthesise_context

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    scene = cfg["scenes"][0]
    paths = ensure_scene(cfg, scene, ROOT)
    bbox = paths["bbox"]
    obs = paths["observation_time"]

    # --- structure -------------------------------------------------
    required = ["app.py", "config.yaml", "requirements.txt", "README.md",
                "run_demo.py", "modules/detection.py", "modules/spill_analysis.py",
                "modules/drift.py", "modules/ais_filter.py", "modules/trajectory.py",
                "modules/behaviour.py", "modules/attribution.py",
                "modules/visualization.py", "modules/report.py",
                "modules/dataset_adapter.py", "modules/geo.py", "modules/pipeline.py",
                "utils/demo_data.py", "utils/scene_builder.py"]
    missing = [f for f in required if not (ROOT / f).exists()]
    check("Project structure complete", not missing, f"missing: {missing}" if missing else "")

    for d in ["data/satellite", "data/masks", "data/ais", "data/environment",
              "data/kaggle_oilspill", "models"]:
        check(f"Directory {d}", (ROOT / d).is_dir())

    # --- pixel/geo transform round trip ----------------------------
    la, lo = pixel_to_geo(640, 336, 1000, 800, bbox)
    x, y = geo_to_pixel(la, lo, 1000, 800, bbox)
    check("pixel_to_geo / geo_to_pixel round trip",
          abs(x - 640) < 1e-6 and abs(y - 336) < 1e-6, f"({x:.4f},{y:.4f})")

    # --- run all three detection paths ------------------------------
    runs = {}
    for label, mode, mask in [("reference mask", "reference_mask", paths["mask"]),
                              ("image processing", "image_processing", paths["mask"]),
                              ("no mask supplied", None, None)]:
        r = run_pipeline(cfg, paths["image"], mask, paths["environment"], paths["ais"],
                         bbox, observation_time=obs, force_detection_mode=mode)
        runs[label] = r
        check(f"Pipeline runs: {label}", r.error is None and r.spill is not None,
              r.error or "")

    r = runs["reference mask"]
    sp, dr, f = r.spill, r.drift, r.filtering
    d = scene

    # --- demo story targets ----------------------------------------
    check("Spill area ~12.6 km2", abs(sp.area_km2 - 12.6) <= 0.5, f"{sp.area_km2:.2f} km2")
    err = haversine_km(dr.origin_lat, dr.origin_lon, *d["truth_origin"])
    check("Probable origin ~18.3901, 72.7542", err <= 0.5,
          f"{dr.origin_lat:.4f},{dr.origin_lon:.4f} ({err * 1000:.0f} m off)")
    check("Estimated release 07:00 UTC", dr.release_time.strftime("%H:%M") == "07:00",
          dr.release_time.strftime("%H:%M"))
    check("Funnel 10 -> 6 -> 4",
          (f.total, f.after_spatial, f.after_temporal) == (10, 6, 4),
          f"{f.total} -> {f.after_spatial} -> {f.after_temporal}")
    top = r.top
    check("Vessel A ranked #1", top is not None and top.name == "MARINA STAR",
          top.name if top else "none")
    check("Vessel A score ~94.4", top and abs(top.total_score - 94.4) <= 1.0,
          f"{top.total_score}" if top else "")
    check("Ranking is descending",
          all(r.ranking[i].total_score >= r.ranking[i + 1].total_score
              for i in range(len(r.ranking) - 1)))

    # --- geometry completeness --------------------------------------
    for attr in ["bounding_box", "polygon", "latitude", "longitude", "area_km2",
                 "perimeter_km", "length_km", "width_km", "elongation",
                 "orientation_bearing"]:
        check(f"Spill geometry: {attr}", getattr(sp, attr) not in (None, [], {}))

    # --- drift ------------------------------------------------------
    check("Forward drift +6/+12/+24 h", sorted(dr.forward_paths) == [6, 12, 24])
    check("Backward drift path generated", len(dr.backward_path) > 5,
          f"{len(dr.backward_path)} steps")

    # --- scoring ----------------------------------------------------
    w = cfg["attribution"]["weights"]
    check("Weights sum to 1.0", abs(sum(w.values()) - 1.0) < 1e-9, str(dict(w)))
    check("All five score components present",
          set(top.components) == {"proximity", "temporal", "trajectory",
                                  "behaviour", "ais_continuity"})
    check("Weighted contributions equal total",
          abs(sum(top.contributions.values()) - top.total_score) < 0.05)
    check("Scores normalised 0-100",
          all(0 <= v.total_score <= 100 for v in r.ranking))

    # --- scientific honesty -----------------------------------------
    check("Reference mask not claimed as AI output",
          r.detection.mode == detection.MODE_REFERENCE and not r.detection.ai_generated)
    check("No metrics fabricated in reference-mask mode", r.metrics is None)
    check("Metrics computed when prediction is independent",
          runs["image processing"].metrics is not None,
          str({k: round(v, 3) for k, v in (runs["image processing"].metrics or {}).items()}))
    check("Geographic basis labelled as prototype",
          "Prototype" in sp.geographic_basis, sp.geographic_basis)
    reasons = " ".join(top.reasons).lower()
    check("No guilt language in output",
          "responsible" not in reasons.replace("responsibility", ""))
    check("Investigation-priority wording present",
          "investigation priority" in reasons)

    # --- determinism -------------------------------------------------
    r2 = run_pipeline(cfg, paths["image"], paths["mask"], paths["environment"],
                      paths["ais"], bbox, observation_time=obs)
    check("Deterministic across runs",
          r2.spill.area_km2 == sp.area_km2
          and r2.top.total_score == top.total_score
          and (r2.drift.origin_lat, r2.drift.origin_lon) == (dr.origin_lat, dr.origin_lon))

    # --- report -------------------------------------------------------
    from modules import report as rp
    doc = rp.build_report(r, cfg, {"Satellite": "synthetic"})
    check("Report generated", len(doc) > 5000, f"{len(doc)} bytes")
    check("Report carries the disclaimer", "requires human investigation" in doc)
    check("Report avoids 'confirmed responsible'", "Confirmed Responsible" not in doc)

    # --- map ----------------------------------------------------------
    from modules import visualization as vz
    r.search_radius_km = 20
    html = vz.build_map(r, bbox).get_root().render()
    for layer in ["Polygon", "polyline", "Probable origin", "MARINA STAR"]:
        check(f"Map layer: {layer}", layer.lower() in html.lower())

    # --- scene library ---------------------------------------------------
    check("Scene library has multiple scenes", len(cfg["scenes"]) >= 3,
          f"{len(cfg['scenes'])} scenes")
    seen_bbox, seen_top = set(), set()
    for sc in cfg["scenes"]:
        pp = ensure_scene(cfg, sc, ROOT)
        rr = run_pipeline(cfg, pp["image"], pp["mask"], pp["environment"], pp["ais"],
                          pp["bbox"], observation_time=pp["observation_time"])
        ok = (rr.error is None and rr.top is not None
              and rr.top.name == sc["fleet"][0][0]
              and (rr.filtering.total, rr.filtering.after_spatial,
                   rr.filtering.after_temporal) == (10, 6, 4)
              and haversine_km(rr.drift.origin_lat, rr.drift.origin_lon,
                               *sc["truth_origin"]) <= 0.8
              and rr.drift.release_time.strftime("%H:%M")
                  == sc["truth_release_time_utc"][11:16])
        check(f"Scene '{sc['id']}' end to end", ok,
              f"{rr.top.name} {rr.top.total_score} | origin "
              f"{rr.drift.origin_lat:.4f},{rr.drift.origin_lon:.4f} | "
              f"release {rr.drift.release_time.strftime('%H:%M')}" if rr.top else rr.error)
        seen_bbox.add((sc["north"], sc["west"]))
        seen_top.add(rr.top.name if rr.top else None)
    check("Scenes are geographically distinct", len(seen_bbox) == len(cfg["scenes"]))
    check("Scenes have distinct top candidates", len(seen_top) == len(cfg["scenes"]),
          str(sorted(x for x in seen_top if x)))

    # --- adaptive context --------------------------------------------------
    other = ensure_scene(cfg, cfg["scenes"][1], ROOT)
    ra = run_pipeline(cfg, other["image"], other["mask"], None, None, bbox,
                      context_builder=lambda sp: synthesise_context(cfg, sp, bbox))
    check("Adaptive context runs on an unmatched image",
          ra.error is None and ra.top is not None,
          f"{ra.top.name} {ra.top.total_score}" if ra.top else ra.error)
    check("Adaptive context is disclosed", ra.context_info is not None)
    check("Adaptive run still produces a clear leader",
          ra.top and len(ra.ranking) > 1
          and ra.top.total_score - ra.ranking[1].total_score > 10)

    # --- summary --------------------------------------------------------
    width = max(len(n) for _, n, _ in results)
    print("\nOILTRACE AI - specification audit\n" + "=" * (width + 22))
    for status, name, detail in results:
        print(f"[{status}] {name:<{width}}  {detail}")
    failed = sum(1 for s, _, _ in results if s == FAIL)
    print("=" * (width + 22))
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
