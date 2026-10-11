"""
Attribution engine.

An interpretable weighted score - not a trained ranker. Every point in the
final number can be traced back to a measured quantity, which is what an
investigator actually needs.

    score = 30% proximity
          + 25% temporal correlation
          + 20% trajectory alignment
          + 15% behavioural anomaly
          + 10% AIS continuity

The output is an INVESTIGATION PRIORITY, not a finding of responsibility.
The top row is labelled "Highest Correlated Candidate" and nowhere does
the system state that a vessel caused the spill.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VesselScore:
    rank: int
    mmsi: str
    name: str
    vessel_type: str
    total_score: float
    confidence_band: str
    components: dict = field(default_factory=dict)     # raw 0-100 sub-scores
    contributions: dict = field(default_factory=dict)  # weighted points
    trajectory: object = None
    behaviour: object = None
    track: object = None
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "mmsi": self.mmsi,
            "vessel_name": self.name,
            "vessel_type": self.vessel_type,
            "score": self.total_score,
            "confidence": self.confidence_band,
            "components": self.components,
            "contributions": self.contributions,
            "reasons": self.reasons,
        }


def _band(score: float, bands: dict) -> str:
    if score >= float(bands.get("high", 80.0)):
        return "HIGH"
    if score >= float(bands.get("medium", 60.0)):
        return "MEDIUM"
    return "LOW"


def _reasons(traj, behav, band: str) -> list[str]:
    """Plain-language justification for the ranking."""
    out = [
        f"{traj.distance_km:.1f} km from the probable origin at closest approach",
        (f"Closest approach {abs(traj.time_delta_hours):.2f} h "
         f"{'after' if traj.time_delta_hours >= 0 else 'before'} the estimated release"),
        (f"Course {traj.course_bearing:.0f} deg vs reconstructed drift "
         f"{traj.drift_bearing:.0f} deg - {traj.alignment_score:.0f}% trajectory alignment"),
    ]
    out.extend(behav.indicators)
    out.append(f"AIS track {behav.continuity_score:.0f}% continuous over the observed period")
    out.append(
        f"Overall correlation with the spill origin is {band.lower()}; "
        "this is an investigation priority, not a determination of responsibility."
    )
    return out


def score_vessels(analysed: list, cfg: dict) -> list[VesselScore]:
    """Rank candidates. `analysed` is a list of (track, trajectory, behaviour)."""
    w = cfg["weights"]
    bands = cfg.get("confidence_bands", {"high": 80.0, "medium": 60.0})

    scored: list[VesselScore] = []
    for track, traj, behav in analysed:
        components = {
            "proximity": traj.proximity_score,
            "temporal": traj.temporal_score,
            "trajectory": traj.alignment_score,
            "behaviour": behav.anomaly_score,
            "ais_continuity": behav.continuity_score,
        }
        contributions = {k: round(components[k] * float(w[k]), 2) for k in components}
        total = round(sum(contributions.values()), 1)
        band = _band(total, bands)

        scored.append(
            VesselScore(
                rank=0,
                mmsi=track.mmsi,
                name=track.name,
                vessel_type=track.vessel_type,
                total_score=total,
                confidence_band=band,
                components=components,
                contributions=contributions,
                trajectory=traj,
                behaviour=behav,
                track=track,
                reasons=_reasons(traj, behav, band),
            )
        )

    scored.sort(key=lambda v: (-v.total_score, v.name))
    for i, v in enumerate(scored, start=1):
        v.rank = i
    return scored


def weight_table(cfg: dict) -> list[dict]:
    labels = {
        "proximity": "Proximity to probable origin",
        "temporal": "Temporal correlation with release window",
        "trajectory": "Trajectory alignment with reconstructed drift",
        "behaviour": "Behavioural anomaly score",
        "ais_continuity": "AIS track continuity",
    }
    return [
        {"Component": labels[k], "Weight": f"{float(v) * 100:.0f}%"}
        for k, v in cfg["weights"].items()
    ]
