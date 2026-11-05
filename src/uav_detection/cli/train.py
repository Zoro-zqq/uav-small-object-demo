"""Train a YOLO baseline; use --smoke before launching a full run."""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_detection.backends import create_model
from uav_detection.common.config import load_yaml
from uav_detection.common.paths import project_path, project_root as get_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 VisDrone YOLO 基线")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/visdrone/yolov8n.yaml"))
    parser.add_argument("--smoke", action="store_true", help="使用约 0.5%% 训练图片、1 epoch 做管线冒烟")
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    project_root = get_project_root()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_yaml(config_path)
    data_path = project_path(str(config.get("dataset_yaml", "configs/datasets/visdrone-yolo.local.yaml")), project_root)
    if not data_path.is_file():
        raise SystemExit(
            f"找不到转换后的 YOLO 数据配置：{data_path}\n"
            "先运行：uav-detect convert --split all"
        )

    model_name = args.model or str(config.get("model", "yolov8n.pt"))
    epochs = args.epochs if args.epochs is not None else int(config.get("epochs", 20))
    image_size = args.imgsz if args.imgsz is not None else int(config.get("imgsz", 960))
    batch_size = args.batch if args.batch is not None else int(config.get("batch", 4))
    device = args.device if args.device is not None else str(config.get("device", 0))
    workers = args.workers if args.workers is not None else int(config.get("workers", 4))
    output_root = project_path(str(config.get("project", "outputs/checkpoints")), project_root)
    name = str(config.get("name", "visdrone_yolov8n"))
    fraction = 0.005 if args.smoke else 1.0
    if args.smoke:
        epochs = 1
        image_size = min(image_size, 640)
        batch_size = min(batch_size, 2)
        workers = 0
        name = f"{name}_smoke"

    print(f"模型：{model_name}; 数据：{data_path}; device={device}; imgsz={image_size}; batch={batch_size}")
    if args.smoke:
        print("冒烟模式：只取约 0.5% 训练数据、1 epoch；仅验证数据管线，不代表模型精度。")
    model = create_model(model_name, backend=str(config.get("backend", "ultralytics")))
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project=str(output_root),
        name=name,
        pretrained=bool(config.get("pretrained", True)),
        seed=int(config.get("seed", 42)),
        deterministic=bool(config.get("deterministic", True)),
        patience=int(config.get("patience", 25)),
        amp=bool(config.get("amp", True)),
        fraction=fraction,
        cache=False,
        val=True,
        plots=True,
        save=True,
        exist_ok=False,
    )
    print(f"訓練已完成；結果目錄：{getattr(results, 'save_dir', output_root)}")


if __name__ == "__main__":
    main()
