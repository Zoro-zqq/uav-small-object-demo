"""Summarize VisDrone object scale using the original validated boxes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from visdrone_utils import CLASS_NAMES, get_split_paths, load_yaml, parse_visdrone_row


def main() -> None:
    parser = argparse.ArgumentParser(description="统计目标尺寸和类别分布，生成直方图")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    config = load_yaml(config_path)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    invalid_box_count = 0
    invalid_box_examples: list[str] = []

    for split in splits:
        image_dir, annotation_dir, _ = get_split_paths(config, split)
        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"找不到标注目录：{annotation_dir}")
        for annotation_path in sorted(annotation_dir.glob("*.txt")):
            image_path = image_dir / f"{annotation_path.stem}.jpg"
            if not image_path.is_file():
                matching_images = [
                    path for path in image_dir.glob(f"{annotation_path.stem}.*") if path.is_file()
                ]
                if not matching_images:
                    raise FileNotFoundError(
                        f"标注找不到同名图片：{annotation_path}"
                    )
                image_path = matching_images[0]
            with Image.open(image_path) as image:
                image_area = image.width * image.height
            with annotation_path.open("r", encoding="utf-8-sig") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        _, _, width, height, score, category, _, _ = parse_visdrone_row(line)
                    except ValueError as error:
                        raise ValueError(f"{annotation_path.name}:{line_number}: {error}") from error
                    if score != 1 or category < 1 or category > len(CLASS_NAMES):
                        continue
                    if width <= 0 or height <= 0:
                        invalid_box_count += 1
                        if len(invalid_box_examples) < 10:
                            invalid_box_examples.append(
                                f"{annotation_path.name}:{line_number}: bbox={width}x{height}"
                            )
                        continue
                    square_root_area = math.sqrt(width * height)
                    rows.append(
                        {
                            "split": split,
                            "image": annotation_path.stem,
                            "class": CLASS_NAMES[category - 1],
                            "width_px": width,
                            "height_px": height,
                            "sqrt_area_px": square_root_area,
                            "area_fraction": width * height / image_area,
                        }
                    )
                    class_counts[CLASS_NAMES[category - 1]] += 1

    if not rows:
        raise ValueError("没有找到有效框，无法统计目标尺度")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.split}_object_scale_summary.json"
    summary = {
        "splits": splits,
        "object_count": len(rows),
        "invalid_box_count_skipped": invalid_box_count,
        "invalid_box_examples": invalid_box_examples,
        "class_distribution": dict(sorted(class_counts.items())),
        "sqrt_area_px_mean": sum(row["sqrt_area_px"] for row in rows) / len(rows),
        "sqrt_area_px_median": statistics.median(row["sqrt_area_px"] for row in rows),
        "sqrt_area_threshold_buckets": {
            "small (<32 px sqrt-area)": sum(row["sqrt_area_px"] < 32 for row in rows),
            "medium (32-96 px sqrt-area)": sum(32 <= row["sqrt_area_px"] < 96 for row in rows),
            "large (>=96 px sqrt-area)": sum(row["sqrt_area_px"] >= 96 for row in rows),
        },
        "median_area_fraction": statistics.median(row["area_fraction"] for row in rows),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("缺少 matplotlib；JSON 统计已写入，请安装项目依赖后重试。") from error

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist([row["sqrt_area_px"] for row in rows], bins=60, color="#377eb8", alpha=0.85)
    axis.set_xlabel("sqrt(box width x box height), pixels")
    axis.set_ylabel("Number of valid instances")
    axis.set_title(f"VisDrone {args.split}: object size distribution")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    chart_path = output_dir / f"{args.split}_object_scale_histogram.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"统计摘要：{json_path}\n尺寸直方图：{chart_path}")


if __name__ == "__main__":
    main()
