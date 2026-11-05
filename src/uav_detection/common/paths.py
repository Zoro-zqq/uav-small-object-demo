"""集中解析项目路径，避免模块移动后把产物写入src。"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    configured = os.environ.get("UAV_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"UAV_PROJECT_ROOT不存在：{root}")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/uav_detection").is_dir():
            return parent
    # 普通wheel安装不包含研究工作区，使用调用目录作为工作区。
    return Path.cwd().resolve()


def project_path(value: str | Path, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else ((root or project_root()) / path).resolve()
