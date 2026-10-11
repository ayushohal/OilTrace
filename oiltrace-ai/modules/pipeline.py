"""
Pipeline orchestration.

Runs DETECT -> TRACE -> ATTRIBUTE as one call and records the status of
every stage so the dashboard can show the progress checklist. The UI holds
no analysis logic; it only renders what this returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules import detection, spill_analysis, drift, ais_filter, trajectory, behaviour, attribution
from modules.dataset_adapter import load_image_gray, load_mask_binary
from modules.geo import BBox


@dataclass
class Stage:
    label: str
    ok: bool = False
    detail: str = ""


@dataclass
class PipelineResult:
    stages: list = field(default_factory=list)
    image: np.ndarray | None = None
    reference_mask: np.ndarray | None = None
    detection: object = None
    spill: object = None
    drift: object = None
    filtering: object = None
    ranking: list = field(default_factory=list)
    env_df: object = None
    ais_df: object = None
    metrics: dict | None = None
    bbox: BBox | None = None
    context_info: dict | None = None
    source_info: dict = field(default_factory=dict)
    error: str | None = None
    search_radius_km: float = 20.0

    @property
    def top(self):
        return self.ranking[0] if self.ranking else None


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _as_frame(source) -> pd.DataFrame:
    """Accept either a CSV path or an in-memory DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(source)


def run_pipeline(
    cfg: dict,
    image_path: str | Path,
    mask_path: str | Path | None,
    env_path: str | Path | None,
    ais_path: str | Path | None,
    bbox: BBox,
    observation_time: datetime | None = None,
    force_detection_mode: str | None = None,
    source_info: dict | None = None,
    context_builder=None,
) -> PipelineResult:
    """Run DETECT -> TRACE -> ATTRIBUTE.

    `env_path` / `ais_path` may be CSV paths or DataFrames. If they are
    None, `context_builder(spill)` is called after the spill geometry is
    known, to synthesise a metocean field and AIS extract around the
    detected slick.
    """
    result = PipelineResult(bbox=bbox, source_info=source_info or {})
    stages = result.stages

    try:
        # ---- 1. Satellite image ------------------------------------
        image = load_image_gray(image_path)
        result.image = image
        stages.append(Stage("Satellite image loaded", True,
                            f"{image.shape[1]} x {image.shape[0]} px, single channel"))

        reference = None
        if mask_path and Path(mask_path).exists():
            reference = load_mask_binary(mask_path)
            if reference.shape != image.shape:
                import cv2
                reference = cv2.resize(reference.astype(np.uint8),
                                       (image.shape[1], image.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
        result.reference_mask = reference

        # ---- 2. Detection ------------------------------------------
        det = detection.detect(image, reference, cfg["detection"],
                               force_mode=force_detection_mode)
        result.detection = det
        stages.append(Stage("Oil spill detected", det.mask.sum() > 0,
                            f"{det.mode} - confidence {det.confidence * 100:.1f}%"))

        # Only evaluate against ground truth when the prediction is
        # independent of it. Scoring a mask against itself is meaningless.
        if reference is not None and det.mode != detection.MODE_REFERENCE:
            result.metrics = detection.segmentation_metrics(det.mask, reference)

        # ---- 3. Spill geometry -------------------------------------
        obs = observation_time or datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0)
        if obs.tzinfo is None:
            obs = obs.replace(tzinfo=timezone.utc)

        spill = spill_analysis.analyse_spill(
            det.mask, bbox, obs, det.confidence, det.mode)
        if spill is None:
            result.error = "No oil-spill region could be segmented from this image."
            stages.append(Stage("Spill geometry calculated", False, result.error))
            return result
        result.spill = spill
        stages.append(Stage("Spill geometry calculated", True,
                            f"{spill.area_km2:.2f} km2, {spill.length_km:.1f} x "
                            f"{spill.width_km:.2f} km"))

        # ---- 4. Environmental data ---------------------------------
        # When no scenario data accompanies the image, synthesise a
        # coherent context around the spill that was actually detected.
        if env_path is None or ais_path is None:
            if context_builder is None:
                raise ValueError(
                    "No environmental/AIS data supplied and no context builder given.")
            env_df, ais_source, ctx = context_builder(spill)
            result.context_info = ctx
            stages.append(Stage(
                "Environmental data synthesised", True,
                f"{len(env_df)} records generated around the detected spill"))
        else:
            env_df = _as_frame(env_path)
            ais_source = ais_path
            stages.append(Stage("Environmental data loaded", True,
                                f"{len(env_df)} metocean records"))
        result.env_df = env_df

        # ---- 5-6. Backward drift and origin ------------------------
        dr = drift.run_drift(spill, env_df, cfg["drift"])
        result.drift = dr
        stages.append(Stage("Backward drift completed", True,
                            f"Hindcast {dr.elapsed_hours:.2f} h at {dr.drift_speed_ms:.2f} m/s"))
        stages.append(Stage("Probable origin estimated", True,
                            f"{dr.origin_lat:.4f} N, {dr.origin_lon:.4f} E at "
                            f"{dr.release_time.strftime('%H:%M')} UTC"))

        # ---- 7. AIS ------------------------------------------------
        ais_df = _as_frame(ais_source)
        result.ais_df = ais_df
        tracks = ais_filter.load_tracks(ais_df, dr.origin_lat, dr.origin_lon, dr.release_time)
        stages.append(Stage("AIS data loaded", True,
                            f"{len(tracks)} vessels, {len(ais_df)} position reports"))

        # ---- 8-9. Filtering ----------------------------------------
        filt = ais_filter.filter_vessels(
            tracks,
            float(cfg["ais"]["candidate_radius_km"]),
            float(cfg["ais"]["release_window_hours"]),
        )
        result.filtering = filt
        stages.append(Stage("Spatial filtering completed", True,
                            f"{filt.total} -> {filt.after_spatial} within "
                            f"{cfg['ais']['candidate_radius_km']:.0f} km"))
        stages.append(Stage("Temporal filtering completed", True,
                            f"{filt.after_spatial} -> {filt.after_temporal} within "
                            f"+/-{cfg['ais']['release_window_hours']:.0f} h"))

        # ---- 10-11. Scoring and ranking ----------------------------
        analysed = []
        for t in filt.candidates:
            tm = trajectory.analyse_trajectory(
                t, dr.drift_bearing,
                float(cfg["ais"]["candidate_radius_km"]),
                float(cfg["ais"]["release_window_hours"]),
            )
            bm = behaviour.analyse_behaviour(
                t, t.closest_time,
                int(cfg["ais"]["ais_reporting_interval_minutes"]),
            )
            analysed.append((t, tm, bm))

        ranking = attribution.score_vessels(analysed, cfg["attribution"])
        result.ranking = ranking
        stages.append(Stage("Vessel scoring completed", True,
                            f"{len(ranking)} candidates scored"))
        stages.append(Stage("Ranking generated", bool(ranking),
                            f"Top candidate: {ranking[0].name} ({ranking[0].total_score:.1f}/100)"
                            if ranking else "No candidates passed the filters"))
        stages.append(Stage("GIS visualization ready", True, "Map layers prepared"))

    except Exception as exc:  # surfaced in the UI rather than crashing
        result.error = f"{type(exc).__name__}: {exc}"
        stages.append(Stage("Pipeline halted", False, result.error))

    return result
