"""Evaluate a trained YOLO model on the converted VisDrone validation labels."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from visdrone_utils import load_yaml, resolve_config_path


def project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 YOLO 权重并保存验证指标")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--size-groups", action="store_true", help="额外执行原图面积分组评估")
    parser.add_argument("--conf", type=float, default=0.001, help="评估预测的最低置信度")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--recall-conf", type=float, default=0.25, help="分组 Recall50 的置信度阈值")
    args = parser.parse_args()
    if args.batch <= 0 or args.imgsz <= 0 or args.max_det <= 0:
        parser.error("batch、imgsz、max-det 必须为正")
    if not 0 <= args.conf <= args.recall_conf <= 1 or not 0 < args.nms_iou <= 1:
        parser.error("需要 0 <= conf <= recall-conf <= 1，且 0 < nms-iou <= 1")

    project_root = Path(__file__).resolve().parent
    weights = args.weights if args.weights.is_absolute() else project_root / args.weights
    data_path = args.data if args.data.is_absolute() else project_root / args.data
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    if not weights.is_file():
        raise FileNotFoundError(f"找不到模型权重：{weights}")
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到数据 YAML：{data_path}；先运行 convert_annotations.py")
    config = load_yaml(data_path)
    val_labels = resolve_config_path(config, "val").parent / "labels"
    if not val_labels.is_dir():
        raise FileNotFoundError(f"找不到转换后的验证标签：{val_labels}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("当前环境未安装 Ultralytics，请先配置训练用环境。") from error

    model = YOLO(str(weights))
    results = model.val(
        data=str(data_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name=f"{weights.stem}_val",
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        plots=True,
        save_json=False,
    )
    metrics = results.box
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights.resolve()),
        "dataset_yaml": str(data_path.resolve()),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "ultralytics_version": version("ultralytics"),
        "metrics": {
            "precision": float(metrics.mp),
            "recall": float(metrics.mr),
            "map50": float(metrics.map50),
            "map50_95": float(metrics.map),
        },
        "metric_scope_warning": (
            "验证使用转换后的 YOLO 标签；score=0/类别0忽略区域不进入常规验证，"
            "因此不能将结果称为 VisDrone 官方 benchmark 指标。"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{weights.stem}_val_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"评估摘要：{report_path}")
    if args.size_groups:
        from size_evaluation import (
            IOU_THRESHOLDS,
            collect_predictions,
            evaluate_size_groups,
            file_sha256,
            load_validation_records,
        )

        started = time.perf_counter()
        records, data_fingerprint = load_validation_records(
            resolve_config_path(config, "val"), val_labels
        )
        predictions = collect_predictions(
            model, records, args.imgsz, args.batch, args.device,
            args.conf, args.nms_iou, args.max_det,
        )
        size_summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "name": "demo_original_area_ap101_v1",
                "ground_truth": "转换后的 YOLO 框还原到原图像素坐标；不是原始忽略区域协议",
                "area_ranges_px2": {"small": "[0,1024)", "medium": "[1024,9216)", "large": "[9216,infinity)"},
                "matching": "按类别、图片、置信度降序一对一匹配；优先组内真实框；组外匹配忽略，组外面积的未匹配预测忽略",
                "ap": "IoU 0.50:0.05:0.95，101 点插值 AP；按有真实框的类别宏平均",
                "ar": "相同 IoU 范围的最大召回率宏平均，不是固定阈值 Recall",
                "recall": "IoU=0.50、score>=recall_conf 的分类别 Recall 宏平均",
                "iou_thresholds": IOU_THRESHOLDS,
                "rect": False,
                "augment": False,
                "agnostic_nms": False,
                "max_det_scope": "NMS 后每张图片所有类别合计上限",
                "warning": "自定义 Demo 指标，不是官方 COCO AP_s/VisDrone benchmark；不等同于 Ultralytics 整体验证指标",
            },
            "settings": {**vars(args), "weights": str(weights.resolve()), "data": str(data_path.resolve()), "output_dir": str(output_dir)},
            "environment": {"python": platform.python_version(), "ultralytics": version("ultralytics"), "torch": version("torch"), "numpy": version("numpy")},
            "weights_sha256": file_sha256(weights),
            "dataset_yaml_sha256": file_sha256(data_path),
            "validation_images_and_labels_sha256": data_fingerprint,
            "image_count": len(records),
            "prediction_count": sum(len(predicted) for predicted in predictions),
            "groups": evaluate_size_groups(records, predictions, args.max_det, args.recall_conf),
            "evaluation_seconds": time.perf_counter() - started,
        }
        size_path = output_dir / f"{weights.stem}_size_summary.json"
        size_path.write_text(json.dumps(size_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        csv_path = output_dir / f"{weights.stem}_size_metrics.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=[
                "group", "gt_count", "classes_with_gt", "AP50_95", "AP50",
                "AR50_95", "recall50_at_conf",
            ])
            writer.writeheader()
            for group, result in size_summary["groups"].items():
                writer.writerow({"group": group, **{key: value for key, value in result.items() if key != "per_class"}})
        inputs_path = output_dir / f"{weights.stem}_size_inputs.json"
        inputs_path.write_text(json.dumps({"records": records, "predictions": predictions}, ensure_ascii=False), encoding="utf-8")
        for group, result in size_summary["groups"].items():
            print(f"[{group}] { {key: value for key, value in result.items() if key != 'per_class'} }")
        print(f"尺寸分组报告：{size_path}\n尺寸指标表：{csv_path}\n分组评估输入存档：{inputs_path}")


if __name__ == "__main__":
    main()
