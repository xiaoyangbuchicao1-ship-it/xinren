"""复用共享快参数MAML核心的响应弹性任务接口。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.agents.maml_ppo import MetaGradientOptimizer
from src.agents.ppo import PPOTrainer
from src.experiments.group_receptiveness_maml import (
    adapt_continuous_to_group_receptiveness,
    evaluate_group_receptiveness_adaptation,
    train_group_receptiveness_meta_iteration,
)
from src.experiments.response_elasticity_task import (
    ResponseElasticityTask,
    config_for_response_elasticity_task,
    estimate_response_elasticity_signal,
)


def sample_response_elasticity_tasks(
    tasks: Sequence[ResponseElasticityTask],
    count: int,
    rng: np.random.Generator,
) -> tuple[ResponseElasticityTask, ...]:
    if count <= 0 or count > len(tasks):
        raise ValueError("响应弹性元批次大小超出训练任务集合。")
    indices = rng.choice(len(tasks), size=count, replace=False)
    return tuple(tasks[int(index)] for index in indices)


def sample_symmetric_response_elasticity_tasks(
    tasks: Sequence[ResponseElasticityTask],
    count: int,
    rng: np.random.Generator,
) -> tuple[ResponseElasticityTask, ...]:
    """按绝对值成对采样正负弹性任务，避免单次元梯度长期偏向一侧。"""

    if count <= 0 or count % 2 != 0:
        raise ValueError("对称响应弹性元批次必须包含正偶数个任务。")
    pairs: list[tuple[ResponseElasticityTask, ResponseElasticityTask]] = []
    for positive in (task for task in tasks if task.magnitude_sensitivity > 0.0):
        negative = next(
            (
                task
                for task in tasks
                if np.isclose(
                    task.magnitude_sensitivity,
                    -positive.magnitude_sensitivity,
                )
            ),
            None,
        )
        if negative is not None:
            pairs.append((negative, positive))
    pair_count = count // 2
    if pair_count > len(pairs):
        raise ValueError("训练任务中没有足够的正负对称响应弹性任务对。")
    selected = rng.choice(len(pairs), size=pair_count, replace=False)
    batch = [task for index in selected for task in pairs[int(index)]]
    order = rng.permutation(len(batch))
    return tuple(batch[int(index)] for index in order)


def adapt_continuous_to_response_elasticity(
    initialization: PPOTrainer,
    config: dict[str, Any],
    task: ResponseElasticityTask,
    **kwargs: Any,
):
    return adapt_continuous_to_group_receptiveness(
        initialization,
        config,
        task,
        task_config_factory=config_for_response_elasticity_task,
        support_signal_estimator=estimate_response_elasticity_signal,
        **kwargs,
    )


def evaluate_response_elasticity_adaptation(
    initialization: PPOTrainer,
    config: dict[str, Any],
    tasks: Sequence[ResponseElasticityTask],
    **kwargs: Any,
) -> dict[str, object]:
    return evaluate_group_receptiveness_adaptation(
        initialization,
        config,
        tasks,
        task_config_factory=config_for_response_elasticity_task,
        support_signal_estimator=estimate_response_elasticity_signal,
        **kwargs,
    )


def train_response_elasticity_meta_iteration(
    meta_trainer: PPOTrainer,
    meta_optimizer: MetaGradientOptimizer,
    config: dict[str, Any],
    tasks: Sequence[ResponseElasticityTask],
    **kwargs: Any,
) -> dict[str, object]:
    return train_group_receptiveness_meta_iteration(
        meta_trainer,
        meta_optimizer,
        config,
        tasks,
        task_config_factory=config_for_response_elasticity_task,
        support_signal_estimator=estimate_response_elasticity_signal,
        **kwargs,
    )
