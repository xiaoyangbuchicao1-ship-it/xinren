"""团队对建议幅度敏感度的一维元任务定义。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, order=True)
class ResponseElasticityTask:
    """正值表示执行率随建议幅度上升而下降得更快。"""

    magnitude_sensitivity: float

    def to_serializable(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ResponseElasticityTaskSplit:
    train: tuple[ResponseElasticityTask, ...]
    validation: tuple[ResponseElasticityTask, ...]
    test: tuple[ResponseElasticityTask, ...]
    split_seed: int
    split_strategy: str = "stratified_holdout"
    train_range: tuple[float, float] | None = None
    test_ranges: tuple[tuple[float, float], ...] | None = None

    def to_serializable(self) -> dict[str, object]:
        return {
            "train": [task.to_serializable() for task in self.train],
            "validation": [task.to_serializable() for task in self.validation],
            "test": [task.to_serializable() for task in self.test],
            "split_seed": self.split_seed,
            "split_strategy": self.split_strategy,
            "train_range": list(self.train_range) if self.train_range else None,
            "test_ranges": (
                [list(values) for values in self.test_ranges]
                if self.test_ranges
                else None
            ),
        }


def make_response_elasticity_task_split(
    *,
    split_seed: int = 2026,
    minimum_sensitivity: float = -0.10,
    maximum_sensitivity: float = 0.10,
    task_count: int = 15,
) -> ResponseElasticityTaskSplit:
    """低、中、高敏感度各留一个验证任务和一个测试任务。"""

    if (
        not minimum_sensitivity < maximum_sensitivity
        or task_count < 9
        or task_count % 3 != 0
    ):
        raise ValueError("敏感度区间必须递增，任务数至少为9且能被3整除。")
    tasks = [
        ResponseElasticityTask(float(value))
        for value in np.linspace(minimum_sensitivity, maximum_sensitivity, task_count)
    ]
    rng = np.random.default_rng(split_seed)
    train: list[ResponseElasticityTask] = []
    validation: list[ResponseElasticityTask] = []
    test: list[ResponseElasticityTask] = []
    for stratum in np.array_split(np.arange(task_count), 3):
        order = rng.permutation(stratum)
        validation.append(tasks[int(order[0])])
        test.append(tasks[int(order[1])])
        train.extend(tasks[int(index)] for index in order[2:])
    return ResponseElasticityTaskSplit(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        split_seed=int(split_seed),
        split_strategy="stratified_holdout",
    )


def make_response_elasticity_ood_task_split(
    *,
    split_seed: int = 2026,
    range_profile: str = "moderate",
) -> ResponseElasticityTaskSplit:
    """训练覆盖完整动作规律，区间内验证，并在双侧轻度外推测试。

    moderate保留原审计范围；wide用于检验任务差异不足的假设。两种方案
    都只改变同一个响应敏感度参数，不增加任务维度。
    """

    if range_profile == "moderate":
        train_values = (-0.20, -0.133, -0.067, 0.0, 0.067, 0.133, 0.20)
        validation_values = (-0.17, -0.10, 0.10, 0.17)
        test_values = (-0.25, -0.225, 0.225, 0.25)
        train_range = (-0.20, 0.20)
        test_ranges = ((-0.25, -0.225), (0.225, 0.25))
    elif range_profile == "wide":
        train_values = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)
        validation_values = (-0.25, -0.15, 0.15, 0.25)
        test_values = (-0.35, -0.325, 0.325, 0.35)
        train_range = (-0.30, 0.30)
        test_ranges = ((-0.35, -0.325), (0.325, 0.35))
    else:
        raise ValueError("响应敏感度范围方案只支持moderate或wide。")
    return ResponseElasticityTaskSplit(
        train=tuple(ResponseElasticityTask(value) for value in train_values),
        validation=tuple(ResponseElasticityTask(value) for value in validation_values),
        test=tuple(ResponseElasticityTask(value) for value in test_values),
        split_seed=int(split_seed),
        split_strategy=f"range_ood_{range_profile}",
        train_range=train_range,
        test_ranges=test_ranges,
    )


def config_for_response_elasticity_task(
    config: dict[str, object],
    task: ResponseElasticityTask,
) -> dict[str, object]:
    """只改变响应曲线斜率，保持各类型中等建议的基准执行率不变。"""

    sensitivity = float(task.magnitude_sensitivity)
    if not np.isfinite(sensitivity) or not -0.35 <= sensitivity <= 0.35:
        raise ValueError("建议幅度敏感度必须位于[-0.35, 0.35]。")
    task_config = deepcopy(config)
    response = task_config["response"]
    direction = np.asarray([1.0, 0.0, -1.0], dtype=np.float64)
    response["response_table"] = {
        response_type: np.clip(
            np.asarray(values, dtype=np.float64) + sensitivity * direction,
            0.0,
            1.0,
        ).tolist()
        for response_type, values in response["response_table"].items()
    }
    return task_config


def estimate_response_elasticity_signal(
    support_summary: dict[str, object],
    base_config: dict[str, object],
) -> tuple[float, float]:
    """按档位响应残差对[1,0,-1]回归，估计建议幅度敏感度。"""

    counts = np.asarray(support_summary["suggestion_bin_counts"], dtype=np.float64)
    observed_values = support_summary["response_rate_mean_by_bin"]
    if counts.shape != (3,) or len(observed_values) != 3 or counts.sum() <= 0.0:
        raise ValueError("支持集必须提供三个建议档位的响应统计。")
    response = base_config["response"]
    probabilities = np.asarray(response["type_probabilities"], dtype=np.float64)
    table = np.asarray(
        [response["response_table"][name] for name in response["type_names"]],
        dtype=np.float64,
    )
    expected_by_bin = probabilities @ table
    direction = np.asarray([1.0, 0.0, -1.0], dtype=np.float64)
    observed = np.asarray(
        [
            expected_by_bin[index] if value is None else float(value)
            for index, value in enumerate(observed_values)
        ],
        dtype=np.float64,
    )
    denominator = float(np.sum(counts * direction**2))
    estimate = (
        float(np.sum(counts * direction * (observed - expected_by_bin)) / denominator)
        if denominator > 0.0
        else 0.0
    )
    # 返回基础响应曲线的小幅—大幅期望差，供日志审计。
    baseline_gap = float(expected_by_bin[0] - expected_by_bin[2])
    return estimate, baseline_gap
