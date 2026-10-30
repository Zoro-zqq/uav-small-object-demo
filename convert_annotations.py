"""Convert VisDrone's 8-column annotations to Ultralytics YOLO labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from visdrone_utils import (
    CLASS_NAMES,
    get_split_paths,
    list_images,
    load_yaml,
    parse_visdrone_row,
)


def convert_split(
    split: str,
    image_dir: Path,
    annotation_dir: Path,
    label_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Convert valid scored boxes, excluding VisDrone ignore regions."""
    images = list_images(image_dir)
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"找不到原始标注目录：{annotation_dir}")
    missing = [image.name for image in images if not (annotation_dir / f"{image.stem}.txt").is_file()]
    if missing:
        raise ValueError(
            f"{split} 有 {len(missing)} 张图片缺少同名标注；示例：{missing[:5]}"
        )

    if label_dir.exists() and any(label_dir.glob("*.txt")) and not overwrite:
        raise FileExistsError(
            f"输出标签目录已有 .txt 文件：{label_dir}。确认后用 --overwrite 覆盖同名标签；"
            "脚本不会删除目录中的其他文件。"
        )

    converted: list[tuple[Path, list[str]]] = []
    class_counts: Counter[str] = Counter()
    ignored_count = 0
    clipped_count = 0
    invalid_box_count = 0
    invalid_box_examples: list[str] = []
    row_count = 0
    for image_path in images:
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
                image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(f"图片不可读：{image_path}: {error}") from error

        yolo_rows = []
        with annotation_path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row_count += 1
                try:
                    left, top, width, height, score, category, _, _ = parse_visdrone_row(line)
                except (ValueError, OverflowError) as error:
                    raise ValueError(
                        f"{annotation_path.name}:{line_number}: {error}"
                    ) from error

                if score == 0 or category == 0:
                    ignored_count += 1
                    continue
                if score != 1:
                    raise ValueError(
                        f"{annotation_path.name}:{line_number}: 预期 score 为 0 或 1，实际为 {score}"
                    )
                if category < 1 or category > len(CLASS_NAMES):
                    raise ValueError(
                        f"{annotation_path.name}:{line_number}: 未知 category={category}"
                    )
                if width <= 0 or height <= 0:
                    invalid_box_count += 1
                    if len(invalid_box_examples) < 10:
                        invalid_box_examples.append(
                            f"{annotation_path.name}:{line_number}: bbox 宽高必须为正数，实际 {width}x{height}"
                        )
                    continue

                right = left + width
                bottom = top + height
                clipped_left = min(max(left, 0.0), float(image_width))
                clipped_top = min(max(top, 0.0), float(image_height))
                clipped_right = min(max(right, 0.0), float(image_width))
                clipped_bottom = min(max(bottom, 0.0), float(image_height))
                if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
                    invalid_box_count += 1
                    if len(invalid_box_examples) < 10:
                        invalid_box_examples.append(
                            f"{annotation_path.name}:{line_number}: bbox 完全位于图像范围外"
                        )
                    continue
                if (
                    clipped_left != left
                    or clipped_top != top
                    or clipped_right != right
                    or clipped_bottom != bottom
                ):
                    clipped_count += 1

                clipped_width = clipped_right - clipped_left
                clipped_height = clipped_bottom - clipped_top
                center_x = (clipped_left + clipped_right) / 2 / image_width
                center_y = (clipped_top + clipped_bottom) / 2 / image_height
                normalized_width = clipped_width / image_width
                normalized_height = clipped_height / image_height
                class_id = category - 1
                yolo_rows.append(
                    f"{class_id} {center_x:.8f} {center_y:.8f} "
                    f"{normalized_width:.8f} {normalized_height:.8f}"
                )
                class_counts[CLASS_NAMES[class_id]] += 1

        converted.append((label_dir / f"{image_path.stem}.txt", yolo_rows))

    label_dir.mkdir(parents=True, exist_ok=True)
    for destination, rows in converted:
        destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    return {
        "split": split,
        "image_dir": str(image_dir),
        "annotation_dir": str(annotation_dir),
        "label_dir": str(label_dir),
        "image_count": len(images),
        "generated_label_count": len(converted),
        "raw_annotation_row_count": row_count,
        "converted_object_count": sum(class_counts.values()),
        "ignored_region_count": ignored_count,
        "clipped_box_count": clipped_count,
        "invalid_box_count": invalid_box_count,
        "invalid_box_examples": invalid_box_examples,
        "class_distribution": dict(sorted(class_counts.items())),
    }


def write_ultralytics_yaml(
    output_path: Path,
    train_image_dir: Path,
    val_image_dir: Path,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"数据集 YAML 已存在：{output_path}。确认后加 --overwrite 更新。"
        )
    content = {
        "path": str(train_image_dir.parent.parent.resolve()).replace("\\", "/"),
        "train": str(train_image_dir.resolve()).replace("\\", "/"),
        "val": str(val_image_dir.resolve()).replace("\\", "/"),
        "names": {class_id: name for class_id, name in enumerate(CLASS_NAMES)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="将 VisDrone 原始框标注转换为 YOLO 格式")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同名生成标签/YAML；不删除其他文件")
    parser.add_argument("--dataset-yaml", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    project_root = Path(__file__).resolve().parent
    dataset_yaml = args.dataset_yaml
    if not dataset_yaml.is_absolute():
        dataset_yaml = project_root / dataset_yaml
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    splits = ("train", "val") if args.split == "all" else (args.split,)
    reports = []
    for split in splits:
        image_dir, annotation_dir, label_dir = get_split_paths(config, split)
        report = convert_split(split, image_dir, annotation_dir, label_dir, args.overwrite)
        reports.append(report)
        print(
            f"[{split}] 图片 {report['image_count']}, 转换框 {report['converted_object_count']}, "
            f"忽略区域 {report['ignored_region_count']}, 无效框 {report['invalid_box_count']}, "
            f"裁剪越界框 {report['clipped_box_count']}"
        )
        for example in report["invalid_box_examples"]:
            print(f"[{split}] 跳过无效框：{example}")

    if args.split == "all":
        train_image_dir, _, _ = get_split_paths(config, "train")
        val_image_dir, _, _ = get_split_paths(config, "val")
        write_ultralytics_yaml(dataset_yaml, train_image_dir, val_image_dir, args.overwrite)
        print(f"Ultralytics 数据配置：{dataset_yaml}")
        print(
            "注意：score=0/类别 0 区域已剔除。常规 YOLO 验证不会复现 VisDrone 官方"
            "忽略区域协议，报告中不能把这组 mAP 称为官方 benchmark 分数。"
        )

    report_path = output_dir / "metrics" / "annotation_conversion_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"splits": reports, "dataset_yaml": str(dataset_yaml)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"转换摘要已保存：{report_path}")


if __name__ == "__main__":
    main()
