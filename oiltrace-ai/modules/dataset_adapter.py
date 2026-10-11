"""
Dataset adapter.

Nothing about the Kaggle oil-spill dataset is assumed. This module walks
whatever is placed under ``data/kaggle_oilspill/``, reports the structure
it *actually* finds, and pairs images with masks when a pairing can be
established. If the folder is empty the adapter reports that honestly and
the app falls back to the built-in synthetic demo scene.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

# Directory names that conventionally hold masks / images.
MASK_DIR_HINTS = {"mask", "masks", "label", "labels", "annotation", "annotations",
                  "groundtruth", "ground_truth", "gt", "segmentation", "seg", "target", "targets"}
IMAGE_DIR_HINTS = {"image", "images", "img", "imgs", "input", "inputs", "jpeg_images",
                   "sar", "scene", "scenes", "data", "train", "test", "val"}

# Filename decorations that distinguish a mask from its image.
MASK_TOKENS = ("_mask", "-mask", "_label", "-label", "_gt", "-gt", "_seg", "-seg", "_target")


@dataclass
class DatasetItem:
    """One usable image, optionally with a corresponding mask."""

    image_path: Path
    mask_path: Path | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        return self.image_path.name

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None and self.mask_path.exists()


@dataclass
class DatasetReport:
    """Everything the adapter could determine about the dataset on disk."""

    root: Path
    present: bool = False
    total_files: int = 0
    image_count: int = 0
    mask_count: int = 0
    paired_count: int = 0
    directories: list[str] = field(default_factory=list)
    formats: dict[str, int] = field(default_factory=dict)
    dimensions: list[str] = field(default_factory=list)
    channel_summary: dict[str, int] = field(default_factory=dict)
    mode_summary: dict[str, int] = field(default_factory=dict)
    class_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    items: list[DatasetItem] = field(default_factory=list)

    @property
    def dataset_type_label(self) -> str:
        """Never call a generic image 'Sentinel-1 SAR' without evidence."""
        if not self.present or self.image_count == 0:
            return "No external dataset loaded"
        looks_sar = any(k in {"L", "I;16", "F"} for k in self.mode_summary)
        hint = "SAR-like (single-channel)" if looks_sar else "Multi-channel / RGB imagery"
        return f"Oil-Spill Image Dataset - {hint}"


def _normalise(stem: str) -> str:
    """Strip mask decorations so an image and its mask share a key."""
    s = stem.lower()
    for token in MASK_TOKENS:
        s = s.replace(token, "")
    return re.sub(r"[^a-z0-9]", "", s)


def _looks_like_mask(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if parts & MASK_DIR_HINTS:
        return True
    return any(t in path.stem.lower() for t in MASK_TOKENS)


def _probe(path: Path) -> dict | None:
    """Read image metadata without loading full pixel data where possible."""
    try:
        with Image.open(path) as im:
            return {"size": im.size, "mode": im.mode, "format": im.format or path.suffix.lstrip(".").upper()}
    except Exception:
        return None


def inspect_dataset(root: str | os.PathLike, max_items: int = 400) -> DatasetReport:
    """Walk `root` and describe what is genuinely there."""
    root = Path(root)
    report = DatasetReport(root=root)

    if not root.exists():
        report.notes.append(f"Directory {root} does not exist.")
        return report

    files = [p for p in root.rglob("*") if p.is_file()]
    report.total_files = len(files)
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTS]

    if not image_files:
        report.notes.append(
            "No image files found. Copy the extracted Kaggle dataset into "
            f"{root} - the adapter will detect its structure on the next run."
        )
        return report

    report.present = True
    report.directories = sorted({str(p.parent.relative_to(root)) for p in image_files})[:25]

    masks: dict[str, Path] = {}
    images: dict[str, Path] = {}
    dims: set[tuple[int, int]] = set()

    for p in image_files[:max_items]:
        meta = _probe(p)
        if meta is None:
            continue
        rel = p.relative_to(root)
        fmt = meta["format"]
        report.formats[fmt] = report.formats.get(fmt, 0) + 1
        report.mode_summary[meta["mode"]] = report.mode_summary.get(meta["mode"], 0) + 1
        dims.add(meta["size"])

        key = _normalise(p.stem)
        if _looks_like_mask(rel):
            masks.setdefault(key, p)
        else:
            images.setdefault(key, p)

    report.image_count = len(images)
    report.mask_count = len(masks)
    report.dimensions = [f"{w} x {h}" for (w, h) in sorted(dims)][:8]

    # Class labels: folder names that are neither image nor mask containers.
    candidates = set()
    for p in image_files[:max_items]:
        for part in p.relative_to(root).parts[:-1]:
            low = part.lower()
            if low not in MASK_DIR_HINTS and low not in IMAGE_DIR_HINTS:
                candidates.add(part)
    report.class_labels = sorted(candidates)[:12]

    for key, img_path in images.items():
        item = DatasetItem(image_path=img_path, mask_path=masks.get(key))
        if item.has_mask:
            report.paired_count += 1
        item.label = img_path.parent.name
        report.items.append(item)

    # Deterministic ordering: paired images first, then alphabetical.
    report.items.sort(key=lambda it: (not it.has_mask, str(it.image_path)))

    if report.paired_count:
        report.notes.append(
            f"{report.paired_count} image/mask pairs matched by filename. "
            "Reference-mask demo mode is available."
        )
    else:
        report.notes.append(
            "No image/mask pairs could be matched. Detection will use the "
            "deterministic image-processing fallback."
        )
    report.notes.append(
        "This dataset carries no geographic metadata; the synthetic demo "
        "bounding box is applied after segmentation only."
    )
    return report


def load_image_gray(path: str | os.PathLike) -> np.ndarray:
    """Load any supported image as float32 grayscale in [0, 1]."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def load_mask_binary(path: str | os.PathLike) -> np.ndarray:
    """Load a mask as a boolean array. Handles 0/1, 0/255 and palette masks."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    threshold = 0.5 if arr.max() <= 1 else 127
    return arr > threshold
