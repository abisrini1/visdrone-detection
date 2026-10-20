"""Small helpers shared by the scripts: paths, labels, CSV logging."""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Folder names produced by Ultralytics' VisDrone.yaml auto-download.
VISDRONE_SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def parse_device(value):
    """'0' -> 0, 'cpu' -> 'cpu', '0,1' stays a string."""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def visdrone_images_dir(split: str = "val", data_dir: str | None = None) -> Path:
    """Locate VisDrone images. Default base: <Ultralytics datasets_dir>/VisDrone.

    Two layouts are supported:
      <base>/images/<split>                       (what Ultralytics' auto-download produces)
      <base>/VisDrone2019-DET-<split>/images      (the original zip layout)
    """
    if data_dir:
        base = Path(data_dir)
    else:
        from ultralytics.utils import SETTINGS  # imported lazily so unit tests need no torch

        base = Path(SETTINGS["datasets_dir"]) / "VisDrone"
    candidates = [base / "images" / split, base / VISDRONE_SPLITS[split] / "images"]
    for images in candidates:
        if images.exists():
            return images
    raise FileNotFoundError(
        "VisDrone images not found. Looked in:\n  " + "\n  ".join(str(c) for c in candidates)
        + "\nRun a training or val command once so Ultralytics downloads VisDrone, or pass "
        "--data-dir pointing at the VisDrone folder."
    )


def label_path_for(image_path) -> Path:
    """Ultralytics convention: swap the last 'images' folder for 'labels' and use a .txt extension."""
    parts = list(Path(image_path).parts)
    if "images" not in parts:
        raise ValueError(f"Expected an 'images' folder in the path: {image_path}")
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def list_images(images_dir: Path, limit: int | None = None):
    files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def load_yolo_labels(label_path: Path, width: int, height: int):
    """Read a YOLO label file (cls cx cy w h, normalized) -> (xyxy pixels, classes)."""
    label_path = Path(label_path)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    arr = np.loadtxt(label_path, ndmin=2)
    cls = arr[:, 0].astype(int)
    cx, cy, w, h = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return xyxy, cls


def append_csv(path: Path, row: dict) -> None:
    """Append a row to a CSV, keeping every column seen so far."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)
