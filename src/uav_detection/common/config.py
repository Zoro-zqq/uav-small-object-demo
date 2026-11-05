"""YAML配置及数据根目录字段解析。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file) or {}
    if not isinstance(content, dict):
        raise ValueError(f"配置文件必须是 YAML mapping：{path}")
    root = Path(str(content.get("path", "."))).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    content["_config_path"] = path
    content["_root"] = root
    return content


def resolve_config_path(config: dict[str, Any], key: str) -> Path:
    if key not in config:
        raise ValueError(f"数据集配置缺少 {key!r} 字段")
    value = Path(str(config[key])).expanduser()
    return (value if value.is_absolute() else config["_root"] / value).resolve()
