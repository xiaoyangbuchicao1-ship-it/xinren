"""以隐藏群体响应函数为元任务的连续 FOMAML-PPO。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from src.agents.maml_ppo import (
    FirstOrderMetaOptimizer,
    clone_task_trainer,
    compute_query_gradients,
)
from src.agents.ppo import PPOTrainer
from src.experiments.continuous_ppo import (
    aggregate_continuous_episodes,
    collect_continuous_rollout,
    evaluate_continuous_trainer,
)
from src.experiments.train_ppo import make_validation_cases


@dataclass(frozen=True, order=True)
class ResponseFunctionTask:
    """一个群体稳定但不直接暴露给策略的整体响应水平。"""

    receptiveness_shift: float

    def to_serializable(self) -> dict[str, float]:
        return {"receptiveness_shift": float(self.receptiveness_shift)}


@dataclass(frozen=True)
class ResponseFunctionTaskSplit:
    """固定且互不重叠的任务级训练、验证和测试划分。"""

    train: tuple[ResponseFunctionTask, ...]
    validation: tuple[ResponseFunctionTask, ...]
    test: tuple[ResponseFunctionTask, ...]
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


@dataclass(frozen=True)
class ResponseFunctionAdaptationMetrics:
    """一个隐藏响应函数任务上的支持集更新记录。"""

    task: Any
    support_summaries: tuple[dict[str, object], ...]
    inner_updates: tuple[dict[str, float | int], ...]

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def make_response_function_task_split(
    *,
    split_seed: int = 2026,
    minimum_shift: float = -0.10,
    maximum_shift: float = 0.10,
    task_count: int = 15,
) -> ResponseFunctionTaskSplit:
    """在可审计的一维响应区间上生成固定留出任务。"""

    if (
        not minimum_shift < maximum_shift
        or task_count < 9
        or task_count % 3 != 0
    ):
        raise ValueError("响应区间必须递增，候选任务数至少为9且能分成三个区间。")
    tasks = [
        ResponseFunctionTask(float(value))
        for value in np.linspace(minimum_shift, maximum_shift, task_count)
    ]
    # 低、中、高响应区间各留出一个验证任务和一个测试任务，
    # 防止随机划分恰好让测试集全部落在接近基准响应表的简单区域。
    rng = np.random.default_rng(split_seed)
    train: list[ResponseFunctionTask] = []
    validation: list[ResponseFunctionTask] = []
    test: list[ResponseFunctionTask] = []
    for stratum in np.array_split(np.arange(task_count), 3):
        order = rng.permutation(stratum)
        validation.append(tasks[int(order[0])])
        test.append(tasks[int(order[1])])
        train.extend(tasks[int(index)] for index in order[2:])
    return ResponseFunctionTaskSplit(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        split_seed=int(split_seed),
        split_strategy="stratified_holdout",
    )


def make_response_function_ood_task_split(
    *,
    split_seed: int = 2026,
) -> ResponseFunctionTaskSplit:
    """构造可直接审计的接纳度插值验证与双侧外推OOD测试。

    训练任务只覆盖[-0.10, 0.10]；验证任务位于该区间内部但不与训练点
    重合；测试任务位于训练支持集两侧的[-0.20,-0.15]和[0.15,0.20]。
    """

    train_values = (-0.10, -0.067, -0.033, 0.0, 0.033, 0.067, 0.10)
    validation_values = (-0.05, 0.05)
    test_values = (-0.20, -0.15, 0.15, 0.20)
    return ResponseFunctionTaskSplit(
        train=tuple(ResponseFunctionTask(value) for value in train_values),
        validation=tuple(ResponseFunctionTask(value) for value in validation_values),
        test=tuple(ResponseFunctionTask(value) for value in test_values),
        split_seed=int(split_seed),
        split_strategy="range_ood",
        train_range=(-0.10, 0.10),
        test_ranges=((-0.20, -0.15), (0.15, 0.20)),
    )


def config_for_response_function_task(
    config: dict[str, Any],
    task: ResponseFunctionTask,
    *,
    response_floor: float = 0.05,
    response_ceiling: float = 0.95,
) -> dict[str, Any]:
    """复制配置并施加任务级响应偏移，不修改冻结基础配置。"""

    shift = float(task.receptiveness_shift)
    if not np.isfinite(shift) or not -0.20 <= shift <= 0.20:
        raise ValueError("试验阶段的群体响应偏移必须位于[-0.20, 0.20]。")
    if not 0.0 <= response_floor < response_ceiling <= 1.0:
        raise ValueError("响应截断边界必须位于[0, 1]且严格递增。")
    task_config = deepcopy(config)
    table = task_config["response"]["response_table"]
    task_config["response"]["response_table"] = {
        response_type: [
            float(np.clip(float(value) + shift, response_floor, response_ceiling))
            for value in values
        ]
        for response_type, values in table.items()
    }
    return task_config


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def adapt_continuous_to_response_function(
    initialization: PPOTrainer,
    config: dict[str, Any],
    task: Any,
    *,
    inner_steps: int,
    support_episodes: int,
    inner_learning_rate: float,
    support_seed: int,
    task_config_factory: Callable[[dict[str, Any], Any], dict[str, Any]] = (
        config_for_response_function_task
    ),
) -> tuple[PPOTrainer, ResponseFunctionAdaptationMetrics]:
    """在同一隐藏响应函数下收集支持回合并执行快速适应。"""

    if inner_steps < 0 or support_episodes <= 0 or inner_learning_rate <= 0.0:
        raise ValueError("内循环步数、支持回合数和学习率必须有效。")
    task_config = task_config_factory(config, task)
    task_trainer = clone_task_trainer(
        initialization,
        inner_learning_rate=inner_learning_rate,
        inner_update_epochs=1,
    )
    seed_rng = np.random.default_rng(support_seed)
    summaries: list[dict[str, object]] = []
    updates: list[dict[str, float | int]] = []
    for _ in range(inner_steps):
        task_seed, type_seed, response_seed, torch_seed, update_seed = (
            int(value)
            for value in seed_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=5,
                dtype=np.uint32,
            )
        )
        _set_torch_seed(torch_seed)
        buffer, summary = collect_continuous_rollout(
            task_trainer,
            task_config,
            task_rng=np.random.default_rng(task_seed),
            type_rng=np.random.default_rng(type_seed),
            response_seed_rng=np.random.default_rng(response_seed),
            episode_target=support_episodes,
            fixed_response_composition=None,
        )
        batch = buffer.to_batch(
            task_trainer.device,
            gamma=float(task_config["ppo"]["gamma"]),
            gae_lambda=float(task_config["ppo"]["gae_lambda"]),
        )
        metrics = task_trainer.update(batch, np.random.default_rng(update_seed))
        summaries.append(summary)
        updates.append(metrics.to_serializable())
    return task_trainer, ResponseFunctionAdaptationMetrics(
        task=task,
        support_summaries=tuple(summaries),
        inner_updates=tuple(updates),
    )


def evaluate_response_function_adaptation(
    initialization: PPOTrainer,
    config: dict[str, Any],
    tasks: Sequence[Any],
    *,
    inner_steps: int,
    support_episodes: int,
    query_episodes: int,
    inner_learning_rate: float,
    evaluation_seed: int,
    task_config_factory: Callable[[dict[str, Any], Any], dict[str, Any]] = (
        config_for_response_function_task
    ),
) -> dict[str, object]:
    """在固定查询病例上配对评价响应函数任务的适应收益。"""

    if not tasks or query_episodes <= 0:
        raise ValueError("适应评价必须包含任务和查询回合。")
    seed_rng = np.random.default_rng(evaluation_seed)
    zero_episodes = []
    adapted_episodes = []
    per_task = []
    for task in tasks:
        task_config = task_config_factory(config, task)
        task_seed, type_seed, response_seed, support_seed = (
            int(value)
            for value in seed_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=4,
                dtype=np.uint32,
            )
        )
        cases = make_validation_cases(
            task_config,
            query_episodes,
            task_seed=task_seed,
            type_seed=type_seed,
            response_seed=response_seed,
        )
        zero_summary, zero_task_episodes = evaluate_continuous_trainer(
            initialization,
            task_config,
            cases,
            deterministic=True,
        )
        adapted, adaptation = adapt_continuous_to_response_function(
            initialization,
            config,
            task,
            inner_steps=inner_steps,
            support_episodes=support_episodes,
            inner_learning_rate=inner_learning_rate,
            support_seed=support_seed,
            task_config_factory=task_config_factory,
        )
        adapted_summary, adapted_task_episodes = evaluate_continuous_trainer(
            adapted,
            task_config,
            cases,
            deterministic=True,
        )
        zero_episodes.extend(zero_task_episodes)
        adapted_episodes.extend(adapted_task_episodes)
        per_task.append(
            {
                "task": task.to_serializable(),
                "zero_step": zero_summary,
                "adapted": adapted_summary,
                "adaptation": adaptation.to_serializable(),
            }
        )
    zero = aggregate_continuous_episodes(zero_episodes)
    adapted = aggregate_continuous_episodes(adapted_episodes)
    return {
        "task_count": len(tasks),
        "inner_steps": inner_steps,
        "support_episodes": support_episodes,
        "query_episodes_per_task": query_episodes,
        "inner_learning_rate": inner_learning_rate,
        "zero_step": zero,
        "adapted": adapted,
        "adaptation_gain": {
            "success_rate": float(adapted["success_rate"])
            - float(zero["success_rate"]),
            "mean_first_step_reward": (
                float(adapted["mean_first_step_reward"])
                - float(zero["mean_first_step_reward"])
            ),
            "mean_total_reward": float(adapted["mean_total_reward"])
            - float(zero["mean_total_reward"]),
            "mean_rounds": float(zero["mean_rounds"])
            - float(adapted["mean_rounds"]),
        },
        "per_task": per_task,
    }


def train_response_function_meta_iteration(
    meta_trainer: PPOTrainer,
    meta_optimizer: FirstOrderMetaOptimizer,
    config: dict[str, Any],
    tasks: Sequence[Any],
    *,
    support_episodes: int,
    query_episodes: int,
    inner_steps: int,
    inner_learning_rate: float,
    iteration_seed: int,
    task_config_factory: Callable[[dict[str, Any], Any], dict[str, Any]] = (
        config_for_response_function_task
    ),
) -> dict[str, object]:
    """执行一次以响应函数为任务的连续FOMAML外循环。"""

    if not tasks:
        raise ValueError("响应函数元训练至少需要一个任务。")
    seed_rng = np.random.default_rng(iteration_seed)
    gradients = []
    task_records = []
    environment_steps = 0
    environment_episodes = 0
    for task in tasks:
        task_config = task_config_factory(config, task)
        support_seed, task_seed, type_seed, response_seed, torch_seed = (
            int(value)
            for value in seed_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=5,
                dtype=np.uint32,
            )
        )
        task_trainer, adaptation = adapt_continuous_to_response_function(
            meta_trainer,
            config,
            task,
            inner_steps=inner_steps,
            support_episodes=support_episodes,
            inner_learning_rate=inner_learning_rate,
            support_seed=support_seed,
            task_config_factory=task_config_factory,
        )
        _set_torch_seed(torch_seed)
        query_buffer, query_summary = collect_continuous_rollout(
            task_trainer,
            task_config,
            task_rng=np.random.default_rng(task_seed),
            type_rng=np.random.default_rng(type_seed),
            response_seed_rng=np.random.default_rng(response_seed),
            episode_target=query_episodes,
            fixed_response_composition=None,
        )
        query_batch = query_buffer.to_batch(
            task_trainer.device,
            gamma=float(task_config["ppo"]["gamma"]),
            gae_lambda=float(task_config["ppo"]["gae_lambda"]),
        )
        query_gradient = compute_query_gradients(task_trainer, query_batch)
        gradients.append(query_gradient)
        support_steps = sum(
            int(item["transitions"]) for item in adaptation.support_summaries
        )
        support_episode_count = sum(
            int(item["episodes"]) for item in adaptation.support_summaries
        )
        environment_steps += support_steps + int(query_summary["transitions"])
        environment_episodes += support_episode_count + int(query_summary["episodes"])
        task_records.append(
            {
                "task": task.to_serializable(),
                "adaptation": adaptation.to_serializable(),
                "query_rollout": query_summary,
                "query_gradient": query_gradient.metrics_to_serializable(),
            }
        )
    meta_metrics = meta_optimizer.step(gradients)
    return {
        "task_count": len(tasks),
        "environment_steps": environment_steps,
        "environment_episodes": environment_episodes,
        "tasks": task_records,
        "meta_update": meta_metrics.to_serializable(),
    }
