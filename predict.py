"""Run inference on a user-selected image, directory, or video."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="用 YOLO 权重预测图片/目录/视频")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--name", type=str, default="visdrone_predictions")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    weights = args.weights if args.weights.is_absolute() else project_root / args.weights
    source = args.source.expanduser()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    if not weights.is_file():
        raise FileNotFoundError(f"找不到模型权重：{weights}")
    if not source.exists():
        raise FileNotFoundError(f"找不到推理输入：{source}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("置信度阈值 --conf 必须在 0 到 1 之间")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("当前环境未安装 Ultralytics，请先配置训练用环境。") from error

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(output_dir),
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=False,
    )
    print(f"完成 {len(results)} 个输入；预测图和 txt 结果保存在 {output_dir / args.name}")


if __name__ == "__main__":
    main()
