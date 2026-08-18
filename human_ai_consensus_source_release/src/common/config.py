"""实验配置读取、验证和冻结。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """以 UTF-8 读取 YAML 配置并执行基础验证。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是映射。")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """验证阶段 A 已确定的关键结构和边界。"""

    required = {
        "experiment",
        "data",
        "consensus",
        "response",
        "reward",
        "ppo",
        "maml",
        "evaluation",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"配置缺少顶层字段：{sorted(missing)}")

    data = config["data"]
    if data["num_experts"] != data["num_ais"]:
        raise ValueError("当前模型要求专家与 AI 一一配对。")
    if min(data["num_experts"], data["num_issues"]) <= 1:
        raise ValueError("专家数和议题数必须大于 1。")

    consensus = config["consensus"]
    threshold = float(consensus["threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("共识门槛必须位于 (0, 1]。")
    planning_margin = float(consensus.get("planning_margin", 0.0))
    if planning_margin < 0.0 or threshold + planning_margin > 1.0:
        raise ValueError("规划余量必须非负，且共识门槛与规划余量之和不能超过 1。")

    bins = [float(value) for value in config["response"]["suggestion_bins"]]
    if bins != [0.0, 0.3, 0.7, 1.0]:
        raise ValueError("主配置的建议区间必须固定为 0.3/0.4/0.3。")

    probabilities = [float(value) for value in config["response"]["type_probabilities"]]
    if abs(sum(probabilities) - 1.0) > 1e-10:
        raise ValueError("专家响应类型概率之和必须为 1。")


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    """对排序后的配置生成稳定哈希。"""

    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """以 UTF-8 YAML 保存配置。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return output_path


def freeze_config(config: dict[str, Any], directory: str | Path) -> tuple[Path, str]:
    """保存不可变配置快照并返回路径和哈希。"""

    frozen = deepcopy(config)
    digest = config_hash(frozen)
    frozen.setdefault("experiment", {})["config_hash"] = digest
    path = Path(directory) / f"config_{digest}.yaml"
    save_config(frozen, path)
    return path, digest
