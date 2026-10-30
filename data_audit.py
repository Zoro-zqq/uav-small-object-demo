"""Audit VisDrone images, original annotations, and dataset splits."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from visdrone_utils import (
    CLASS_NAMES,
    IMAGE_SUFFIXES,
    get_split_paths,
    list_images,
    load_yaml,
    parse_visdrone_row,
)


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_split(split: str, image_dir: Path, annotation_dir: Path) -> dict[str, Any]:
    """Scan one split and return aggregate counts plus a few actionable examples."""
    image_paths = list_images(image_dir)
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"找不到原始标注目录：{annotation_dir}")
    annotation_paths = sorted(annotation_dir.glob("*.txt"))
    image_by_stem = {path.stem: path for path in image_paths}
    annotation_by_stem = {path.stem: path for path in annotation_paths}
    missing_annotation = sorted(image_by_stem.keys() - annotation_by_stem.keys())
    orphan_annotation = sorted(annotation_by_stem.keys() - image_by_stem.keys())

    class_counts: Counter[str] = Counter()
    score_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    object_size_counts: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    malformed_examples: list[str] = []
    invalid_box_examples: list[str] = []
    unknown_class_examples: list[str] = []
    corrupt_images: list[str] = []
    image_object_counts: list[int] = []
    row_count = 0
    valid_objects = 0
    ignored_objects = 0
    malformed_rows = 0
    invalid_boxes = 0
    out_of_bounds_boxes = 0
    unknown_classes = 0
    small_objects = 0
    image_dimensions: dict[str, tuple[int, int]] = {}

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.load()
                image_dimensions[image_path.stem] = image.size
                image_sizes[f"{image.width}x{image.height}"] += 1
        except (OSError, ValueError) as error:
            corrupt_images.append(f"{image_path}: {error}")

    for annotation_path in annotation_paths:
        image_size = image_dimensions.get(annotation_path.stem)
        object_count = 0
        try:
            lines = annotation_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as error:
            malformed_rows += 1
            if len(malformed_examples) < 10:
                malformed_examples.append(f"{annotation_path.name}: 无法读取：{error}")
            image_object_counts.append(0)
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                left, top, width, height, score, category, truncation, occlusion = (
                    parse_visdrone_row(line)
                )
            except (ValueError, OverflowError) as error:
                malformed_rows += 1
                if len(malformed_examples) < 10:
                    malformed_examples.append(
                        f"{annotation_path.name}:{line_number}: {error}"
                    )
                continue

            score_counts[str(score)] += 1
            category_counts[str(category)] += 1
            if score == 1 and not 1 <= category <= len(CLASS_NAMES):
                unknown_classes += 1
                if len(unknown_class_examples) < 10:
                    unknown_class_examples.append(
                        f"{annotation_path.name}:{line_number}: score=1, category={category}"
                    )

            if score != 1 or not 1 <= category <= len(CLASS_NAMES):
                ignored_objects += 1
                continue

            values = (left, top, width, height)
            if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
                invalid_boxes += 1
                if len(invalid_box_examples) < 10:
                    invalid_box_examples.append(
                        f"{annotation_path.name}:{line_number}: box={values}"
                    )
                continue

            if image_size is not None:
                image_width, image_height = image_size
                if (
                    left < 0
                    or top < 0
                    or left + width > image_width
                    or top + height > image_height
                ):
                    out_of_bounds_boxes += 1
                    if len(invalid_box_examples) < 10:
                        invalid_box_examples.append(
                            f"{annotation_path.name}:{line_number}: 超出 {image_width}x{image_height}"
                        )

            valid_objects += 1
            class_counts[CLASS_NAMES[category - 1]] += 1
            object_count += 1
            square_root_area = math.sqrt(width * height)
            if square_root_area < 32:
                bucket = "small (<32 px sqrt-area)"
                small_objects += 1
            elif square_root_area < 96:
                bucket = "medium (32-96 px sqrt-area)"
            else:
                bucket = "large (>=96 px sqrt-area)"
            object_size_counts[bucket] += 1
        image_object_counts.append(object_count)

    unsupported_images = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES
    )
    report = {
        "split": split,
        "image_dir": str(image_dir),
        "annotation_dir": str(annotation_dir),
        "image_count": len(image_paths),
        "annotation_file_count": len(annotation_paths),
        "missing_annotation_count": len(missing_annotation),
        "missing_annotation_examples": missing_annotation[:10],
        "orphan_annotation_count": len(orphan_annotation),
        "orphan_annotation_examples": orphan_annotation[:10],
        "unsupported_image_file_count": len(unsupported_images),
        "unsupported_image_file_examples": unsupported_images[:10],
        "corrupt_image_count": len(corrupt_images),
        "corrupt_image_examples": corrupt_images[:10],
        "image_size_distribution": dict(sorted(image_sizes.items())),
        "annotation_row_count": row_count,
        "valid_object_count": valid_objects,
        "ignored_or_unknown_object_count": ignored_objects,
        "score_distribution": dict(sorted(score_counts.items())),
        "category_distribution_including_ignored": dict(sorted(category_counts.items())),
        "class_distribution": dict(sorted(class_counts.items())),
        "object_size_distribution": dict(object_size_counts),
        "small_object_fraction": small_objects / valid_objects if valid_objects else None,
        "mean_valid_objects_per_annotated_image": (
            sum(image_object_counts) / len(image_object_counts)
            if image_object_counts
            else 0.0
        ),
        "malformed_row_count": malformed_rows,
        "malformed_row_examples": malformed_examples,
        "invalid_box_count": invalid_boxes,
        "out_of_bounds_box_count": out_of_bounds_boxes,
        "invalid_box_examples": invalid_box_examples,
        "unknown_class_count": unknown_classes,
        "unknown_class_examples": unknown_class_examples,
    }
    return report


def visualize_annotations(
    split: str,
    image_dir: Path,
    annotation_dir: Path,
    output_path: Path,
    sample_count: int,
    seed: int,
) -> Path:
    """Render a reproducible random sample of original VisDrone annotations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"图片目录为空：{image_dir}")
    sample = random.Random(seed).sample(image_paths, min(sample_count, len(image_paths)))
    columns = min(4, len(sample))
    rows = math.ceil(len(sample) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    colors = plt.get_cmap("tab10")
    legend_handles = [
        Patch(color="gray", label="ignored / score=0"),
        Patch(color=colors(0), label="valid box"),
    ]

    for axis, image_path in zip(axes.flat, sample):
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
                width, height = image.size
        except (OSError, ValueError) as error:
            print(f"[{split}] 无法可视化图片：{image_path}: {error}")
            axis.text(
                0.5, 0.5, "IMAGE DECODE FAILED", ha="center", va="center", color="red"
            )
            axis.set_title(image_path.name, fontsize=8)
            axis.axis("off")
            continue
        axis.imshow(image)
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if annotation_path.is_file():
            with annotation_path.open("r", encoding="utf-8-sig") as file:
                for line in file:
                    if not line.strip():
                        continue
                    try:
                        left, top, box_width, box_height, score, category, _, _ = (
                            parse_visdrone_row(line)
                        )
                    except ValueError:
                        continue
                    if box_width <= 0 or box_height <= 0:
                        continue
                    ignored = score != 1 or category < 1 or category > len(CLASS_NAMES)
                    color = "gray" if ignored else colors((category - 1) % 10)
                    axis.add_patch(
                        Rectangle(
                            (left, top),
                            box_width,
                            box_height,
                            fill=False,
                            edgecolor=color,
                            linewidth=0.65,
                        )
                    )
                    if not ignored:
                        axis.text(
                            left,
                            max(0, top - 2),
                            CLASS_NAMES[category - 1],
                            color="white",
                            fontsize=5,
                            bbox={"facecolor": color, "alpha": 0.7, "pad": 0.5},
                        )
        axis.set_title(f"{image_path.name} ({width}x{height})", fontsize=8)
        axis.axis("off")

    for axis in axes.flat[len(sample) :]:
        axis.axis("off")
    figure.legend(handles=legend_handles, loc="lower center", ncol=2)
    figure.suptitle(f"VisDrone {split}: {len(sample)} random annotated images (seed={seed})")
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="自动审计 VisDrone 图片、原始标注和数据划分")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--visualize", type=int, default=20, help="每个 split 随机绘制多少张图；0 表示不绘图")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir

    reports = []
    for split in splits:
        image_dir, annotation_dir, _ = get_split_paths(config, split)
        report = audit_split(split, image_dir, annotation_dir)
        reports.append(report)
        print(
            f"[{split}] 图片 {report['image_count']}, 标注文件 {report['annotation_file_count']}, "
            f"有效框 {report['valid_object_count']}, 忽略/未知框 {report['ignored_or_unknown_object_count']}, "
            f"坏行 {report['malformed_row_count']}, 越界框 {report['out_of_bounds_box_count']}"
        )
        print(
            f"[{split}] 缺少标注 {report['missing_annotation_count']}, "
            f"孤立标注 {report['orphan_annotation_count']}, 损坏图片 {report['corrupt_image_count']}"
        )
        print(f"[{split}] 类别分布：{report['class_distribution']}")
        for example in report["corrupt_image_examples"]:
            print(f"[{split}] 损坏图片：{example}")
        if args.visualize > 0:
            image_dir, annotation_dir, _ = get_split_paths(config, split)
            visual_path = visualize_annotations(
                split,
                image_dir,
                annotation_dir,
                output_dir / "predictions" / f"{split}_annotations_sample.png",
                args.visualize,
                args.seed,
            )
            report["visualization"] = str(visual_path)
            print(f"[{split}] 标注抽样图：{visual_path}")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "splits": reports,
    }
    report_path = output_dir / "metrics" / "data_audit_summary.json"
    _write_json(report_path, summary)
    print(f"审计摘要已保存：{report_path}")
    has_errors = any(
        report["missing_annotation_count"]
        or report["orphan_annotation_count"]
        or report["corrupt_image_count"]
        or report["malformed_row_count"]
        or report["invalid_box_count"]
        or report["unknown_class_count"]
        for report in reports
    )
    if has_errors:
        raise SystemExit("数据审计发现需要复核的问题；请先查看 JSON 摘要。")


if __name__ == "__main__":
    main()
