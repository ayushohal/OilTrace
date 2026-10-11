"""
Oil-spill detection / segmentation.

Three modes, tried in the order configured in ``config.yaml``:

  Mode 1  "U-Net Model"                     - a trained model, if one exists
  Mode 2  "Demo Mode - Reference Ground Truth" - dataset-supplied mask
  Mode 3  "Prototype Segmentation - Image Processing" - deterministic fallback

The application never crashes because a model is missing: the fallback is
pure NumPy/OpenCV and always returns a mask.

Confidence is DERIVED from image evidence (contrast, homogeneity, shape),
never assigned. When the reference ground-truth mask is used, that fact is
stated explicitly - the prototype does not claim AI generated it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODE_UNET = "U-Net Model"
MODE_REFERENCE = "Demo Mode - Reference Ground Truth"
MODE_FALLBACK = "Prototype Segmentation - Image Processing"


@dataclass
class DetectionResult:
    mask: np.ndarray                 # bool, same shape as image
    mode: str                        # human-readable detection mode
    confidence: float                # 0-1, computed from image evidence
    ai_generated: bool               # False when a reference mask was used
    diagnostics: dict                # component scores, for the UI
    notes: list[str]


# ---------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------
def compute_confidence(image: np.ndarray, mask: np.ndarray, weights: dict) -> tuple[float, dict]:
    """Score how strongly the image supports the detected region being oil.

    Three independent pieces of evidence, all measured from the pixels:
      contrast    - oil damps capillary waves, so it is darker than the sea
      homogeneity - a slick's return is smoother than wind-roughened water
      shape       - slicks are elongated; compact blobs are usually look-alikes
    """
    if mask.sum() == 0:
        return 0.0, {"contrast": 0.0, "homogeneity": 0.0, "shape": 0.0}

    m8 = mask.astype(np.uint8)
    # Sample the slick interior only. Boundary pixels are mixtures of oil
    # and open water; including them understates both the contrast and the
    # smoothness of the slick.
    core = cv2.erode(m8, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if core.sum() < 50:
        core = mask
    inside = image[core]

    # Background = an annulus around the slick, offset far enough to skip
    # the transition zone, rather than the whole scene.
    near = cv2.dilate(m8, np.ones((15, 15), np.uint8), iterations=1).astype(bool)
    far = cv2.dilate(m8, np.ones((61, 61), np.uint8), iterations=1).astype(bool)
    ring = far & ~near
    if ring.sum() > 100:
        outside = image[ring]
    elif (~mask).sum() > 100:
        outside = image[~mask]
    else:
        # The mask covers essentially the whole scene: there is no
        # background left to compare against, so no evidence of a slick.
        return 0.0, {"contrast": 0.0, "homogeneity": 0.0, "shape": 0.0}

    if inside.size == 0 or outside.size == 0:
        return 0.0, {"contrast": 0.0, "homogeneity": 0.0, "shape": 0.0}

    m_in, m_out = float(inside.mean()), float(outside.mean())
    s_in, s_out = float(inside.std()), float(outside.std())

    contrast = np.clip((m_out - m_in) / (m_out + 1e-6), 0.0, 1.0)
    homogeneity = np.clip(1.0 - (s_in / (s_out + 1e-6)), 0.0, 1.0)

    # Elongation from the second moments of the mask.
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys]).astype(np.float64)
    cov = np.cov(pts)
    if cov.shape == (2, 2):
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(ev[0] / max(ev[1], 1e-9)))
    else:
        elong = 1.0
    shape = float(np.clip((elong - 1.0) / 5.0, 0.0, 1.0))

    parts = {"contrast": float(contrast), "homogeneity": float(homogeneity), "shape": shape}
    score = (
        weights["contrast"] * parts["contrast"]
        + weights["homogeneity"] * parts["homogeneity"]
        + weights["shape"] * parts["shape"]
    )
    return float(np.clip(score, 0.0, 1.0)), parts


# ---------------------------------------------------------------------
# Mode 3 - deterministic image-processing segmentation
# ---------------------------------------------------------------------
def segment_image_processing(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Dark-region segmentation tuned for SAR-like oil slicks.

    Deterministic: no randomness, no learned parameters. Given the same
    image it always returns the same mask.
    """
    notes: list[str] = []
    img8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    k = int(cfg.get("blur_kernel", 7)) | 1
    # Median first: it suppresses SAR speckle without softening slick edges.
    smooth = cv2.medianBlur(img8, k)
    smooth = cv2.GaussianBlur(smooth, (k, k), 0)

    h, w = smooth.shape
    # Local background: a heavily blurred copy of the scene. Comparing
    # against this rather than a global threshold makes the detector
    # immune to wind streaks and illumination gradients - and a slick
    # covering ~1% of the frame never dominates a global histogram.
    sigma = max(15.0, float(np.hypot(h, w)) / 20.0)
    background = cv2.GaussianBlur(smooth.astype(np.float32), (0, 0), sigma)

    ratio = float(cfg.get("dark_ratio", 0.62))
    dark = smooth.astype(np.float32) < (background * ratio)
    coverage = float(dark.mean())
    notes.append(
        f"Local-background dark-spot threshold at {ratio:.2f} x background "
        f"(sigma {sigma:.0f} px); {coverage * 100:.2f}% of the scene is dark."
    )

    # Guard rails: if the relative threshold selects nothing, or almost
    # everything, fall back to a global Otsu split.
    if coverage < 1e-5 or coverage > 0.45:
        thresh_val, otsu = cv2.threshold(smooth, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark = otsu > 0
        notes.append(
            f"Relative threshold was unusable; global Otsu applied at "
            f"{thresh_val:.0f}/255 instead."
        )

    binary = (dark.astype(np.uint8)) * 255

    mk = int(cfg.get("morph_kernel", 5)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_px = int(cfg.get("min_region_pixels", 400))
    mask = np.zeros_like(binary, dtype=bool)

    if n <= 1:
        notes.append("No dark regions survived filtering.")
        return mask, notes

    # Keep the largest region, plus nearby smaller patches of the same slick.
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    mask |= labels == main
    main_cx, main_cy = stats[main, cv2.CC_STAT_LEFT] + stats[main, cv2.CC_STAT_WIDTH] / 2, \
                       stats[main, cv2.CC_STAT_TOP] + stats[main, cv2.CC_STAT_HEIGHT] / 2
    kept = 1
    diag = float(np.hypot(image.shape[0], image.shape[1]))
    for i in range(1, n):
        if i == main or stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
        cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
        if np.hypot(cx - main_cx, cy - main_cy) < 0.22 * diag:
            mask |= labels == i
            kept += 1

    notes.append(f"{n - 1} dark regions found; {kept} retained as one slick complex.")

    # Sanity guard: a slick filling most of the frame is not a slick. It
    # means the scene is uniformly dark (no open water to contrast against),
    # so report nothing rather than a meaningless multi-thousand km2 result.
    coverage_final = float(mask.mean())
    max_cov = float(cfg.get("max_coverage_fraction", 0.60))
    if coverage_final > max_cov:
        notes.append(
            f"Detected region covers {coverage_final * 100:.0f}% of the scene, above "
            f"the {max_cov * 100:.0f}% plausibility limit. No open water remains for "
            "comparison, so the detection is rejected."
        )
        return np.zeros_like(mask), notes

    return mask, notes


# ---------------------------------------------------------------------
# Mode 1 - optional trained model
# ---------------------------------------------------------------------
def _try_unet(image: np.ndarray, model_path: Path) -> np.ndarray | None:
    """Run an ONNX U-Net if one is present. Returns None if unavailable."""
    if not model_path.exists():
        return None
    try:
        import onnxruntime as ort  # optional dependency
    except Exception:
        return None
    try:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        _, c, h, w = inp.shape
        h = int(h) if isinstance(h, int) else 256
        w = int(w) if isinstance(w, int) else 256
        c = int(c) if isinstance(c, int) else 1
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        batch = np.repeat(resized[None, None, :, :], c, axis=1).astype(np.float32)
        out = sess.run(None, {inp.name: batch})[0]
        prob = np.squeeze(out)
        if prob.ndim == 3:
            prob = prob[-1]
        prob = cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return prob > 0.5
    except Exception:
        return None


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def detect(
    image: np.ndarray,
    reference_mask: np.ndarray | None,
    cfg: dict,
    force_mode: str | None = None,
) -> DetectionResult:
    """Run detection using the highest-priority mode that is available."""
    notes: list[str] = []
    weights = cfg["confidence_weights"]
    order = list(cfg.get("mode_priority", ["unet", "reference_mask", "image_processing"]))
    if force_mode:
        order = [force_mode] + [m for m in order if m != force_mode]

    mask = None
    mode = MODE_FALLBACK
    ai_generated = True

    for candidate in order:
        if candidate == "unet":
            predicted = _try_unet(image, Path(cfg.get("model_path", "models/unet_oilspill.onnx")))
            if predicted is not None and predicted.sum() > 0:
                mask, mode, ai_generated = predicted, MODE_UNET, True
                notes.append("Segmentation produced by the trained U-Net model.")
                break
        elif candidate == "reference_mask":
            if reference_mask is not None and reference_mask.sum() > 0:
                mask, mode, ai_generated = reference_mask.copy(), MODE_REFERENCE, False
                notes.append(
                    "Demo segmentation using the provided reference mask. "
                    "This mask was NOT generated by a model."
                )
                break
        elif candidate == "image_processing":
            produced, fb_notes = segment_image_processing(image, cfg)
            if produced.sum() > 0:
                mask, mode, ai_generated = produced, MODE_FALLBACK, True
                notes.extend(fb_notes)
                notes.append("Deterministic image processing - no machine learning involved.")
                break

    if mask is None or mask.sum() == 0:
        produced, fb_notes = segment_image_processing(image, cfg)
        mask = produced
        mode = MODE_FALLBACK
        ai_generated = True
        notes.extend(fb_notes)
        notes.append("Higher-priority modes unavailable; fallback segmentation used.")

    confidence, parts = compute_confidence(image, mask, weights)
    return DetectionResult(
        mask=mask,
        mode=mode,
        confidence=confidence,
        ai_generated=ai_generated,
        diagnostics=parts,
        notes=notes,
    )


# ---------------------------------------------------------------------
# Optional evaluation against a real ground truth
# ---------------------------------------------------------------------
def segmentation_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict | None:
    """IoU / Dice / precision / recall. Returns None if no truth exists.

    Never called without a genuine ground-truth mask, and never called when
    the prediction *is* the ground-truth mask - that would be a meaningless
    perfect score.
    """
    if truth is None or predicted is None:
        return None
    if predicted.shape != truth.shape:
        truth = cv2.resize(truth.astype(np.uint8), (predicted.shape[1], predicted.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    tp = float(np.logical_and(predicted, truth).sum())
    fp = float(np.logical_and(predicted, ~truth).sum())
    fn = float(np.logical_and(~predicted, truth).sum())
    if tp + fp + fn == 0:
        return None
    return {
        "iou": tp / (tp + fp + fn),
        "dice": 2 * tp / (2 * tp + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }
