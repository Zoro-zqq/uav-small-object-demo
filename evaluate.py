"""Evaluate a trained YOLO model on the converted VisDrone validation labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
