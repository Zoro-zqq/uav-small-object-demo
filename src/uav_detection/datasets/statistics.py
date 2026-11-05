"""读取原始VisDrone或COCO标注，输出三类分布；不转换训练标签。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from uav_detection.common.config import load_yaml
from uav_detection.common.paths import project_path
from uav_detection.datasets.visdrone import CLASS_NAMES, get_split_paths, list_images, parse_visdrone_row


@dataclass
class DatasetStats:
    name: str
    split: str
    source: str
    image_counts: dict[str, int] = field(default_factory=dict)
    class_counts: Counter[str] = field(default_factory=Counter)
    scales: list[float] = field(default_factory=list)
    image_sizes: Counter[str] = field(default_factory=Counter)
    issues: Counter[str] = field(default_factory=Counter)
    source_sha256: str | None = None
    image_check: dict[str, Any] = field(default_factory=lambda: {"performed": False})

    def add_box(self, key: str, category: str, bbox: Any, image_size: tuple[int, int]) -> None:
        try:
            if len(bbox) != 4:
                raise ValueError("bbox必须有四列")
            x, y, width, height = map(float, bbox)
            area = width * height
            valid = all(math.isfinite(v) for v in (x, y, width, height, area))
            if not valid or width <= 0 or height <= 0 or area <= 0:
                raise ValueError("非有限或非正尺寸")
        except (TypeError, ValueError, OverflowError):
            self.issues["invalid_bbox_skipped"] += 1
            return
        if x < 0 or y < 0 or x + width > image_size[0] or y + height > image_size[1]:
            self.issues["out_of_bounds_included_without_clipping"] += 1
        # 统一按原图bbox的w*h统计，不使用可能表示分割面积的COCO area字段。
        self.scales.append(math.sqrt(area))
        self.image_counts[key] += 1
        self.class_counts[category] += 1


def unique_index(rows: list[dict[str, Any]], label: str) -> dict[Any, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identifier = row["id"]
        if identifier in indexed:
            raise ValueError(f"重复的{label} id：{identifier}")
        indexed[identifier] = row
    return indexed


def image_size(row: dict[str, Any]) -> tuple[int, int]:
    values = (row["width"], row["height"])
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 or int(v) != v for v in values):
        raise ValueError(f"图片尺寸无效：{values}")
    return int(values[0]), int(values[1])


def load_coco(path: Path, name: str, split: str, image_dir: Path | None = None) -> DatasetStats:
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8-sig"))
    images = unique_index(data["images"], "image")
    categories = unique_index(data["categories"], "category")
    unique_index(data["annotations"], "annotation")
    names = [row["name"] for row in categories.values()]
    if len(set(names)) != len(names):
        raise ValueError("类别名称重复，无法按名称汇总")
    result = DatasetStats(name, split, str(path.resolve()))
    result.source_sha256 = hashlib.sha256(payload).hexdigest()
    result.class_counts.update({category: 0 for category in names})
    sizes = {}
    for identifier, row in images.items():
        sizes[identifier] = image_size(row)
        result.image_counts[str(identifier)] = 0
        result.image_sizes[f"{sizes[identifier][0]}x{sizes[identifier][1]}"] += 1
    for row in data["annotations"]:
        identifier, category_id = row["image_id"], row["category_id"]
        if identifier not in images or category_id not in categories:
            raise ValueError(f"标注{row['id']}引用了不存在的image/category id")
        if row.get("ignore", 0) or row.get("iscrowd", 0):
            result.issues["ignored_or_crowd_skipped"] += 1
            continue
        result.add_box(str(identifier), categories[category_id]["name"], row.get("bbox"), sizes[identifier])
    if image_dir is not None:
        root = image_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"图片根目录不存在：{root}")
        failures = []
        failure_count = 0
        for identifier, row in images.items():
            target = (root / row["file_name"]).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"file_name超出图片根目录：{row['file_name']}")
            try:
                with Image.open(target) as image:
                    if image.size != sizes[identifier]:
                        raise ValueError(f"尺寸{image.size}与标注{sizes[identifier]}不一致")
            except (OSError, ValueError) as error:
                failure_count += 1
                if len(failures) < 10:
                    failures.append({"file": str(target), "error": str(error)})
        result.image_check = {
            "performed": True, "scope": "文件存在/图片头部尺寸；未完整解码或目视核对",
            "root": str(root), "failure_count": failure_count, "examples": failures,
        }
    return result


def load_visdrone(config_path: Path, split: str) -> DatasetStats:
    if split not in ("train", "val"):
        raise ValueError("VisDrone统计请分别选择train或val，不合并划分")
    config = load_yaml(config_path)
    image_dir, annotation_dir, _ = get_split_paths(config, split)
    images = list_images(image_dir)
    if not images:
        raise ValueError(f"没有图片：{image_dir}")
    stems = [image.stem for image in images]
    if len(stems) != len(set(stems)):
        raise ValueError("图片stem重复，不能确定一一对应的标注")
    annotation_stems = {p.stem for p in annotation_dir.glob("*.txt")}
    if set(stems) != annotation_stems:
        raise ValueError("图片与标注不是一一对应，请先运行data_audit.py")
    result = DatasetStats("VisDrone", split, str(config_path.resolve()))
    result.class_counts.update({category: 0 for category in CLASS_NAMES})
    for image_path in images:
        key = image_path.stem
        with Image.open(image_path) as image:
            size = image.size
        result.image_counts[key] = 0
        result.image_sizes[f"{size[0]}x{size[1]}"] += 1
        for line_number, line in enumerate((annotation_dir / f"{key}.txt").read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                x, y, w, h, score, category, _, _ = parse_visdrone_row(line)
            except ValueError as error:
                raise ValueError(f"{key}.txt:{line_number}: {error}") from error
            if score != 1 or not 1 <= category <= len(CLASS_NAMES):
                result.issues["ignored_or_unknown_skipped"] += 1
                continue
            result.add_box(key, CLASS_NAMES[category - 1], [x, y, w, h], size)
    return result


SIZE_EDGES = (0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, math.inf)
INSTANCE_EDGES = (0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, math.inf)


def histogram(values: list[float], edges: tuple[float, ...]) -> list[dict[str, Any]]:
    import numpy as np

    counts, _ = np.histogram(values, bins=edges)
    labels = [f"[{left:g},{right:g})" for left, right in zip(edges, edges[1:])]
    return [{"interval": label, "count": int(count)} for label, count in zip(labels, counts)]


def write_statistics(stats: DatasetStats, output_dir: Path) -> dict[str, Any]:
    if not stats.image_counts or not stats.scales:
        raise ValueError("没有图片或有效框，无法生成完整统计")
    # 使用新目录，避免覆盖已有实验和数据统计。
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = list(stats.image_counts.values())
    size_hist = histogram(stats.scales, SIZE_EDGES)
    instance_hist = histogram(counts, INSTANCE_EDGES)
    summary = {
        "dataset": stats.name, "split_label": stats.split,
        "split_note": "COCO划分由所选JSON决定；split_label仅为记录，不会重新划分数据",
        "source": stats.source, "annotation_sha256": stats.source_sha256,
        "definition": "原图bbox sqrt(w*h)；忽略ignore/crowd、VisDrone非有效类别及非正/非有限框；越界框记录但不裁剪",
        "image_count": len(counts), "object_count": len(stats.scales),
        "zero_valid_object_image_count": counts.count(0),
        "sqrt_area_px_mean": statistics.mean(stats.scales),
        "sqrt_area_px_median": statistics.median(stats.scales),
        "instances_per_image_mean": statistics.mean(counts),
        "instances_per_image_median": statistics.median(counts),
        "instances_per_image_max": max(counts),
        "class_counts": dict(stats.class_counts.most_common()),
        "image_size_distribution": dict(stats.image_sizes),
        "issues": dict(stats.issues), "image_check": stats.image_check,
        "size_histogram": size_hist, "instances_histogram": instance_hist,
        "warning": "描述性统计，不是AP评测或完整图片审计；单个split不应强行等于全数据集论文总量",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, header, rows in (
        ("per_image_counts.csv", ["image_key", "valid_object_count"], stats.image_counts.items()),
        ("class_counts.csv", ["class", "count"], stats.class_counts.most_common()),
    ):
        with (output_dir / filename).open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(header)
            writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = f"{stats.name} / {stats.split}"
    for filename, title, data, xlabel in (
        ("object_sizes.png", "Object size distribution", size_hist, "sqrt(bbox width x height), original pixels"),
        ("instances_per_image.png", "Instances per image (including zero)", instance_hist, "Valid instances per image"),
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        total = sum(row["count"] for row in data)
        axis.bar([row["interval"] for row in data], [100 * row["count"] / total for row in data])
        axis.set(xlabel=xlabel, ylabel="Percentage (%)", title=f"{prefix}: {title}")
        axis.tick_params(axis="x", labelrotation=45)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5))
    pairs = stats.class_counts.most_common()
    axis.bar([name for name, _ in pairs], [count for _, count in pairs])
    axis.set(xlabel="Classes ranked by count", ylabel="Valid instances (log scale)", title=f"{prefix}: Class long tail")
    axis.set_yscale("log")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / "class_long_tail.png", dpi=150)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="VisDrone/COCO数据统计：尺寸、每图实例数、类别长尾三类图；不训练")
    parser.add_argument("--format", choices=("coco", "visdrone"), required=True)
    parser.add_argument("--annotations", type=Path, help="所选单一split的COCO标注JSON")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets/visdrone.local.yaml"))
    parser.add_argument("--image-dir", type=Path, help="可选：COCO file_name相对于此目录，检查存在性/头部尺寸")
    parser.add_argument("--name", default="AI-TOD-original", help="记录实际数据版本；不自动识别或切换标注版本")
    parser.add_argument("--split", default="train", help="COCO记录标签；VisDrone只接受train或val")
    parser.add_argument("--output-dir", type=Path, required=True, help="尚不存在的新输出目录")
    args = parser.parse_args()
    args.config = project_path(args.config)
    args.output_dir = project_path(args.output_dir)
    if args.annotations is not None:
        args.annotations = project_path(args.annotations)
    if args.image_dir is not None:
        args.image_dir = project_path(args.image_dir)
    if args.output_dir.exists():
        parser.error("输出目录已存在，请指定新目录，保留旧结果")
    if args.format == "coco":
        if args.annotations is None:
            parser.error("COCO格式需要--annotations")
        stats = load_coco(args.annotations.expanduser(), args.name, args.split, args.image_dir)
    else:
        if args.annotations is not None or args.image_dir is not None:
            parser.error("VisDrone从--config读取路径，不使用--annotations/--image-dir")
        stats = load_visdrone(args.config.expanduser(), args.split)
    summary = write_statistics(stats, args.output_dir)
    print(json.dumps({key: summary[key] for key in ("dataset", "image_count", "object_count", "sqrt_area_px_mean", "issues", "image_check")}, ensure_ascii=False, indent=2))
    print(f"三类图和统计摘要已保存：{args.output_dir.resolve()}")
    if stats.image_check.get("failure_count", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
