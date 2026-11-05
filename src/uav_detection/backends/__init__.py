"""模型实现入口；扩展框架时在这里注册适配器。"""
from __future__ import annotations

from importlib import import_module


BACKENDS = {"ultralytics": "uav_detection.backends.ultralytics"}


def create_model(source: str, backend: str = "ultralytics"):
    if backend not in BACKENDS:
        raise ValueError(f"未实现的模型后端：{backend}；可用：{', '.join(BACKENDS)}")
    return import_module(BACKENDS[backend]).create_model(source)
