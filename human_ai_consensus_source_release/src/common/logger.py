"""运行目录、环境清单和 JSONL 日志。"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config_hash, freeze_config
from .encoding import write_json


def collect_environment_info() -> dict[str, Any]:
    """收集复现实验所需的软件和设备信息。"""

    packages = {}
    for name in ["numpy", "scipy", "torch", "matplotlib", "pandas", "pyyaml", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "stdout_encoding": sys.stdout.encoding,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }


def create_run_directory(config: dict[str, Any], project_root: str | Path, stage: str) -> Path:
    """使用阶段、时间和配置哈希创建唯一运行目录。"""

    root = Path(project_root)
    output_root = root / config["experiment"]["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    digest = config_hash(config, length=8)
    run_dir = output_root / f"{stage}_{timestamp}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    freeze_config(config, run_dir)
    write_json(collect_environment_info(), run_dir / "environment.json")
    return run_dir


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """以 UTF-8 追加一行结构化日志。"""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

