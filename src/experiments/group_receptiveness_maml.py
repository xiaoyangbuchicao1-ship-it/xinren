"""以团队整体接纳度为元任务的一维快适应MAML-PPO。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from src.agents.maml_ppo import (
    MetaGradientOptimizer,
    clone_task_trainer,
    compute_query_gradients,
    compute_second_order_fast_query_gradients,
    differentiable_fast_adaptation,
)
from src.agents.ppo import PPOTrainer
from src.experiments.continuous_ppo import (
    aggregate_continuous_episodes,
    collect_continuous_rollout,
    evaluate_continuous_trainer,
)
from src.experiments.response_function_maml import (
    ResponseFunctionTask,
    ResponseFunctionTaskSplit,
    config_for_response_function_task,
    make_response_function_task_split,
)
from src.experiments.train_ppo import make_validation_cases


@dataclass(frozen=True)
class GroupReceptivenessAdaptationMetrics:
    """一个团队接纳度任务的一维共享快偏置更新记录。"""

    task: Any
    support_summary: dict[str, object] | None
    inner_update: dict[str, float | int] | None
    shared_offset_change: float = 0.0
    estimated_receptiveness_residual: float = 0.0
    expected_baseline_response: float = 0.0
    shared_policy_gradient: float = 0.0
    shared_auxiliary_gradient: float = 0.0
    shared_combined_gradient: float = 0.0
    shared_fast_offset: bool = True
    advantage_estimator: str = "monte_carlo"

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def make_group_receptiveness_task_split(
    *,
    split_seed: int = 2026,
    minimum_shift: float = -0.10,
    maximum_shift: float = 0.10,
    task_count: int = 15,
) -> ResponseFunctionTaskSplit:
    """复用分层响应函数划分，确保低、中、高接纳度均有留出任务。"""

    return make_response_function_task_split(
        split_seed=split_seed,
        minimum_shift=minimum_shift,
        maximum_shift=maximum_shift,
        task_count=task_count,
    )


def sample_group_receptiveness_tasks(
    tasks: Sequence[ResponseFunctionTask],
    count: int,
    rng: np.random.Generator,
) -> tuple[ResponseFunctionTask, ...]:
    if count <= 0 or count > len(tasks):
        raise ValueError("团队接纳度任务采样数量超出候选集合。")
    indices = rng.choice(len(tasks), size=count, replace=False)
    return tuple(tasks[int(index)] for index in indices)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def estimate_receptiveness_residual(
    support_summary: dict[str, object],
    base_config: dict[str, Any],
) -> tuple[float, float]:
    """用历史响应减去基础响应表期望，估计不暴露隐藏标签的团队偏移。"""

    counts = np.asarray(support_summary["suggestion_bin_counts"], dtype=np.float64)
    if counts.shape != (3,) or np.any(counts < 0.0) or counts.sum() <= 0.0:
        raise ValueError("支持集必须包含三个建议档位的非负且非空计数。")
    response = base_config["response"]
    probabilities = np.asarray(response["type_probabilities"], dtype=np.float64)
    table = np.asarray(
        [response["response_table"][name] for name in response["type_names"]],
        dtype=np.float64,
    )
    expected_by_bin = probabilities @ table
    expected = float(np.dot(counts, expected_by_bin) / counts.sum())
    observed = float(support_summary["active_response_rate_mean"])
    return float(observed - expected), expected


def adapt_continuous_to_group_receptiveness(
    initialization: PPOTrainer,
    config: dict[str, Any],
    task: ResponseFunctionTask,
    *,
    inner_steps: int,
    support_episodes: int,
    inner_learning_rate: float,
    support_seed: int,
    calibration_coefficient: float = 0.0,
    policy_gradient_coefficient: float = 1.0,
    task_config_factory: Callable[[dict[str, Any], Any], dict[str, Any]] = (
        config_for_response_function_task
    ),
    support_signal_estimator: Callable[
        [dict[str, object], dict[str, Any]], tuple[float, float]
    ] = estimate_receptiveness_residual,
) -> tuple[PPOTrainer, GroupReceptivenessAdaptationMetrics]:
    """用同一团队过去的决策，仅校准一个共享动作均值偏置。"""

    if inner_steps not in (0, 1):
        raise ValueError("当前二阶团队快适应只支持0或1个内循环步骤。")
    if support_episodes <= 0 or inner_learning_rate <= 0.0:
        raise ValueError("支持回合数和内学习率必须为正。")
    task_config = task_config_factory(config, task)
    task_trainer = clone_task_trainer(
        initialization,
        inner_learning_rate=inner_learning_rate,
        inner_update_epochs=1,
        inner_optimizer="sgd",
    )
    if inner_steps == 0:
        return task_trainer, GroupReceptivenessAdaptationMetrics(task, None, None)

    seed_rng = np.random.default_rng(support_seed)
    task_seed, type_seed, response_seed, torch_seed = (
        int(value)
        for value in seed_rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=4,
            dtype=np.uint32,
        )
    )
    _set_torch_seed(torch_seed)
    support_buffer, support_summary = collect_continuous_rollout(
        initialization,
        task_config,
        task_rng=np.random.default_rng(task_seed),
        type_rng=np.random.default_rng(type_seed),
        response_seed_rng=np.random.default_rng(response_seed),
        episode_target=support_episodes,
    )
    support_batch = support_buffer.to_batch(
        initialization.device,
        gamma=float(task_config["ppo"]["gamma"]),
        gae_lambda=float(task_config["ppo"]["gae_lambda"]),
        advantage_estimator="monte_carlo",
    )
    residual, expected_response = support_signal_estimator(
        support_summary,
        config,
    )
    adaptation = differentiable_fast_adaptation(
        initialization,
        support_batch,
        inner_learning_rate=inner_learning_rate,
        shared_offset=True,
        shared_auxiliary_signal=residual,
        shared_auxiliary_coefficient=calibration_coefficient,
        shared_policy_gradient_coefficient=policy_gradient_coefficient,
    )
    with torch.no_grad():
        task_trainer.actor.expert_mean_offsets.copy_(
            adaptation.adapted_offsets.detach()
        )
    shared_offset_change = float(
        (
            adaptation.adapted_offsets.detach()
            - initialization.actor.expert_mean_offsets.detach()
        )
        .mean()
        .item()
    )
    return task_trainer, GroupReceptivenessAdaptationMetrics(
        task=task,
        support_summary=support_summary,
        inner_update=adaptation.metrics.to_serializable(),
        shared_offset_change=shared_offset_change,
        estimated_receptiveness_residual=residual,
        expected_baseline_response=expected_response,
        shared_policy_gradient=float(adaptation.shared_policy_gradient or 0.0),
        shared_auxiliary_gradient=float(adaptation.shared_auxiliary_gradient or 0.0),
        shared_combined_gradient=float(adaptation.shared_combined_gradient or 0.0),
    )


def evaluate_group_receptiveness_adaptation(
    initialization: PPOTrainer,
    config: dict[str, Any],
    tasks: Sequence[Any],
    *,
    inner_steps: int,
    support_episodes: int,
    query_episodes: int,
    inner_learning_rate: float,
    evaluation_seed: int,
    calibration_coefficient: float = 0.0,
    policy_gradient_coefficient: float = 1.0,
    task_config_factory: Callable[[dict[str, Any], Any], dict[str, Any]] = (
        config_for_response_function_task
    ),
    support_signal_estimator: Callable[
        [dict[str, object], dict[str, Any]], tuple[float, float]
    ] = estimate_receptiveness_residual,
) -> dict[str, object]:
    """在配对查询病例上比较团队级快适应前后的策略。"""

    if not tasks or query_episodes <= 0:
        raise ValueError("团队接纳度评价必须包含任务与查询回合。")
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
        adapted, adaptation = adapt_continuous_to_group_receptiveness(
            initialization,
            config,
            task,
            inner_steps=inner_steps,
            support_episodes=support_episodes,
            inner_learning_rate=inner_learning_rate,
            support_seed=support_seed,
            calibration_coefficient=calibration_coefficient,
            policy_gradient_coefficient=policy_gradient_coefficient,
            task_config_factory=task_config_factory,
            support_signal_estimator=support_signal_estimator,
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
        "calibration_coefficient": float(calibration_coefficient),
        "policy_gradient_coefficient": float(policy_gradient_coefficient),
        "shared_fast_offset": True,
        "advantage_estimator": "monte_carlo",
        "zero_step": zero,
        "adapted": adapted,
        "adaptation_gain": {
            "success_rate": float(adapted["success_rate"])
            - float(zero["success_rate"]),
            "mean_first_step_reward": float(adapted["mean_first_step_reward"])
            - float(zero["mean_first_step_reward"]),
            "mean_total_reward": float(adapted["mean_total_reward"])
            - float(zero["mean_total_reward"]),
            "mean_rounds": float(zero["mean_rounds"])
            - float(adapted["mean_rounds"]),
        },
        "per_task": per_task,
    }


def train_group_receptiveness_meta_iteration(
    meta_trainer: PPOTrainer,
    meta_optimizer: MetaGradientOptimizer,
    config: dict[str, Any],
    tasks: Sequence[Any],
    *,
    support_episodes: int,
    query_episodes: int,
    inner_learning_rate: float,
    iteration_seed: int,
    paired_scenarios: bool = True,
    calibration_coefficient: float = 0.0,
    policy_gradient_coefficient: float = 1.0,
    outer_update_epochs: int = 1,
    second_order_fast: bool = False,
    task_config_factory: Callable[[dict[str, Any], Any], dict[str, Any]] = (
        config_for_response_function_task
    ),
    support_signal_estimator: Callable[
        [dict[str, object], dict[str, Any]], tuple[float, float]
    ] = estimate_receptiveness_residual,
) -> dict[str, object]:
    """执行一次共享一维快偏置的FOMAML或二阶MAML-PPO元更新。"""

    if (
        not tasks
        or support_episodes <= 0
        or query_episodes <= 0
        or outer_update_epochs <= 0
    ):
        raise ValueError("元更新必须包含任务、支持回合和查询回合。")
    seed_rng = np.random.default_rng(iteration_seed)
    paired_values = seed_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=8,
        dtype=np.uint32,
    )
    gradients = []
    task_batches = []
    task_records = []
    environment_steps = 0
    for task in tasks:
        values = (
            paired_values
            if paired_scenarios
            else seed_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=8,
                dtype=np.uint32,
            )
        )
        (
            support_task_seed,
            support_type_seed,
            support_response_seed,
            support_torch_seed,
            query_task_seed,
            query_type_seed,
            query_response_seed,
            query_torch_seed,
        ) = (int(value) for value in values)
        task_config = task_config_factory(config, task)

        _set_torch_seed(support_torch_seed)
        support_buffer, support_summary = collect_continuous_rollout(
            meta_trainer,
            task_config,
            task_rng=np.random.default_rng(support_task_seed),
            type_rng=np.random.default_rng(support_type_seed),
            response_seed_rng=np.random.default_rng(support_response_seed),
            episode_target=support_episodes,
        )
        support_batch = support_buffer.to_batch(
            meta_trainer.device,
            gamma=float(task_config["ppo"]["gamma"]),
            gae_lambda=float(task_config["ppo"]["gae_lambda"]),
            advantage_estimator="monte_carlo",
        )
        residual, expected_response = support_signal_estimator(
            support_summary,
            config,
        )
        adaptation = differentiable_fast_adaptation(
            meta_trainer,
            support_batch,
            inner_learning_rate=inner_learning_rate,
            shared_offset=True,
            shared_auxiliary_signal=residual,
            shared_auxiliary_coefficient=calibration_coefficient,
            shared_policy_gradient_coefficient=policy_gradient_coefficient,
        )
        task_trainer = clone_task_trainer(
            meta_trainer,
            inner_learning_rate=inner_learning_rate,
            inner_update_epochs=1,
            inner_optimizer="sgd",
        )
        with torch.no_grad():
            task_trainer.actor.expert_mean_offsets.copy_(
                adaptation.adapted_offsets.detach()
            )

        _set_torch_seed(query_torch_seed)
        query_buffer, query_summary = collect_continuous_rollout(
            task_trainer,
            task_config,
            task_rng=np.random.default_rng(query_task_seed),
            type_rng=np.random.default_rng(query_type_seed),
            response_seed_rng=np.random.default_rng(query_response_seed),
            episode_target=query_episodes,
        )
        query_batch = query_buffer.to_batch(
            meta_trainer.device,
            gamma=float(task_config["ppo"]["gamma"]),
            gae_lambda=float(task_config["ppo"]["gae_lambda"]),
            advantage_estimator="monte_carlo",
        )
        query_gradient = (
            compute_second_order_fast_query_gradients(
                meta_trainer,
                adaptation.adapted_offsets,
                query_batch,
            )
            if second_order_fast
            else compute_query_gradients(task_trainer, query_batch)
        )
        gradients.append(query_gradient)
        task_batches.append((support_batch, query_batch, residual))
        environment_steps += len(support_buffer) + len(query_buffer)
        task_records.append(
            {
                "task": task.to_serializable(),
                "support_rollout": support_summary,
                "inner_update": adaptation.metrics.to_serializable(),
                "estimated_receptiveness_residual": residual,
                "expected_baseline_response": expected_response,
                "shared_policy_gradient": adaptation.shared_policy_gradient,
                "shared_auxiliary_gradient": adaptation.shared_auxiliary_gradient,
                "shared_combined_gradient": adaptation.shared_combined_gradient,
                "shared_offset_change": float(
                    (
                        adaptation.adapted_offsets.detach()
                        - meta_trainer.actor.expert_mean_offsets.detach()
                    )
                    .mean()
                    .item()
                ),
                "query_rollout": query_summary,
                "query_gradient": query_gradient.metrics_to_serializable(),
                "scenario_seeds": {
                    "support_task": support_task_seed,
                    "support_type": support_type_seed,
                    "support_response": support_response_seed,
                    "query_task": query_task_seed,
                    "query_type": query_type_seed,
                    "query_response": query_response_seed,
                },
            }
        )
    # 普通PPO会在一批旧策略数据上执行多轮裁剪更新；元PPO也应充分利用
    # 查询数据。每轮必须从当前元参数重新计算内适应，不能复用脱图梯度。
    epoch_metrics = [meta_optimizer.step(gradients)]
    early_stopped = False
    for _ in range(1, outer_update_epochs):
        epoch_gradients = []
        for support_batch, query_batch, residual in task_batches:
            adaptation = differentiable_fast_adaptation(
                meta_trainer,
                support_batch,
                inner_learning_rate=inner_learning_rate,
                shared_offset=True,
                shared_auxiliary_signal=residual,
                shared_auxiliary_coefficient=calibration_coefficient,
                shared_policy_gradient_coefficient=policy_gradient_coefficient,
            )
            if second_order_fast:
                query_gradient = compute_second_order_fast_query_gradients(
                    meta_trainer,
                    adaptation.adapted_offsets,
                    query_batch,
                )
            else:
                task_trainer = clone_task_trainer(
                    meta_trainer,
                    inner_learning_rate=inner_learning_rate,
                    inner_update_epochs=1,
                    inner_optimizer="sgd",
                )
                with torch.no_grad():
                    task_trainer.actor.expert_mean_offsets.copy_(
                        adaptation.adapted_offsets.detach()
                    )
                query_gradient = compute_query_gradients(task_trainer, query_batch)
            epoch_gradients.append(
                query_gradient
            )
        metrics = meta_optimizer.step(epoch_gradients)
        epoch_metrics.append(metrics)
        if metrics.mean_query_approximate_kl > 1.5 * meta_trainer.target_kl:
            early_stopped = True
            break
    serialized_epochs = [item.to_serializable() for item in epoch_metrics]
    meta_update = {
        key: (
            int(serialized_epochs[-1][key])
            if key == "task_count"
            else float(np.mean([float(item[key]) for item in serialized_epochs]))
        )
        for key in serialized_epochs[0]
    }
    meta_update["epochs_completed"] = len(epoch_metrics)
    meta_update["early_stopped"] = early_stopped
    return {
        "task_count": len(tasks),
        "environment_steps": environment_steps,
        "environment_episodes": len(tasks) * (support_episodes + query_episodes),
        "shared_fast_offset": True,
        "calibration_coefficient": float(calibration_coefficient),
        "policy_gradient_coefficient": float(policy_gradient_coefficient),
        "outer_update_epochs": int(outer_update_epochs),
        "meta_gradient_order": "second" if second_order_fast else "first",
        "paired_scenarios": bool(paired_scenarios),
        "tasks": task_records,
        "meta_update": meta_update,
        "meta_update_epochs": serialized_epochs,
    }
