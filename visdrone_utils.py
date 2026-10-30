"""Shared helpers for the VisDrone small-object detection demo."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping and resolve its relative paths from the file."""
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file) or {}
    if not isinstance(content, dict):
        raise ValueError(f"配置文件必须是 YAML mapping：{path}")

    root_value = content.get("path", ".")
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    content["_config_path"] = path
    content["_root"] = root
    return content


def resolve_config_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a dataset path field relative to the dataset root."""
    if key not in config:
        raise ValueError(f"数据集配置缺少 {key!r} 字段")
    value = Path(str(config[key])).expanduser()
    if not value.is_absolute():
        value = config["_root"] / value
    return value.resolve()


def get_split_paths(config: dict[str, Any], split: str) -> tuple[Path, Path, Path]:
    """Return image, original annotation, and converted-label directories."""
    image_dir = resolve_config_path(config, split)
    split_root = image_dir.parent
    annotation_key = f"{split}_annotations"
    label_key = f"{split}_labels"
    annotation_dir = (
        resolve_config_path(config, annotation_key)
        if annotation_key in config
        else split_root / "annotations"
    )
    label_dir = (
        resolve_config_path(config, label_key)
        if label_key in config
        else split_root / "labels"
    )
    return image_dir, annotation_dir, label_dir


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"找不到图片目录：{image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_visdrone_row(
    line: str,
) -> tuple[float, float, float, float, int, int, int, int]:
    """Parse x, y, w, h, score, category, truncation, and occlusion."""
    fields = line.replace(",", " ").split()
    if len(fields) != 8:
        raise ValueError(f"预期 8 个字段，实际为 {len(fields)} 个")
    values = [float(field) for field in fields]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("字段中包含非有限数值")
    left, top, width, height = values[:4]
    score, category, truncation, occlusion = values[4:]
    if not all(value.is_integer() for value in (score, category, truncation, occlusion)):
        raise ValueError("score、category、truncation、occlusion 必须是整数")
    return (
        left,
        top,
        width,
        height,
        int(score),
        int(category),
        int(truncation),
        int(occlusion),
    )


def read_annotation_file(
    path: Path,
) -> list[tuple[int, tuple[float, float, float, float, int, int, int, int]]]:
    """Parse annotation rows, retaining line numbers for useful diagnostics."""
    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            rows.append((line_number, parse_visdrone_row(line)))
    return rows
