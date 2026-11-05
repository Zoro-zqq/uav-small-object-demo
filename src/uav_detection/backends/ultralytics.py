"""Ultralytics模型加载；不同YOLO模型由权重或模型YAML选择。"""


def create_model(source: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            '当前环境未安装Ultralytics。请在基线环境安装：pip install -e ".[ultralytics]"'
        ) from error
    return YOLO(source)
