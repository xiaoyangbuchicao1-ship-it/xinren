"""MAML-PPO 的任务克隆、一阶/二阶查询梯度与外循环更新。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.agents.ppo import (
    PPOBatch,
    PPOTrainer,
    PPOUpdateMetrics,
    clipped_surrogate_loss,
)
from src.env.response_model import sample_response_types


@dataclass(frozen=True)
class QueryGradientResult:
    """一个适应后元任务的查询梯度与诊断量。"""

    actor_gradients: tuple[torch.Tensor, ...]
    critic_gradients: tuple[torch.Tensor, ...]
    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float

    def metrics_to_serializable(self) -> dict[str, float]:
        values = asdict(self)
        values.pop("actor_gradients")
        values.pop("critic_gradients")
        return values


@dataclass(frozen=True)
class DifferentiableFastAdaptationResult:
    """五维快参数的一步可微SGD结果。"""

    adapted_offsets: torch.Tensor
    metrics: PPOUpdateMetrics
    shared_policy_gradient: float | None = None
    shared_auxiliary_gradient: float | None = None
    shared_combined_gradient: float | None = None


@dataclass(frozen=True)
class MetaUpdateMetrics:
    """一次一阶外循环更新的梯度诊断。"""

    task_count: int
    actor_gradient_norm: float
    critic_gradient_norm: float
    actor_pairwise_cosine_mean: float
    actor_pairwise_cosine_min: float
    critic_pairwise_cosine_mean: float
    critic_pairwise_cosine_min: float
    mean_query_actor_loss: float
    mean_query_critic_loss: float
    mean_query_entropy: float
    mean_query_approximate_kl: float
    mean_query_clip_fraction: float

    def to_serializable(self) -> dict[str, float | int]:
        return asdict(self)


def sample_meta_task(
    config: dict[str, object],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """采样一个只由五位专家响应类型组合定义的元任务。"""

    data = config["data"]
    response = config["response"]
    assert isinstance(data, dict) and isinstance(response, dict)
    return sample_response_types(
        int(data["num_experts"]),
        response["type_names"],
        response["type_probabilities"],
        rng,
    )


def clone_task_trainer(
    meta_trainer: PPOTrainer,
    *,
    inner_learning_rate: float,
    inner_update_epochs: int = 1,
    inner_optimizer: str = "adam",
) -> PPOTrainer:
    """克隆元参数并创建全新内循环优化器，不共享可变状态。"""

    if inner_learning_rate <= 0.0 or inner_update_epochs <= 0:
        raise ValueError("内学习率和内循环 PPO epoch 必须为正。")
    optimizer_name = str(inner_optimizer).lower()
    if optimizer_name not in {"adam", "sgd"}:
        raise ValueError("内循环优化器只支持Adam或SGD。")
    actor = deepcopy(meta_trainer.actor)
    critic = deepcopy(meta_trainer.critic)
    trainer = PPOTrainer(
        actor,
        critic,
        learning_rate=inner_learning_rate,
        clip_range=meta_trainer.clip_range,
        update_epochs=inner_update_epochs,
        minibatch_size=meta_trainer.minibatch_size,
        entropy_coefficient=meta_trainer.entropy_coefficient,
        value_coefficient=meta_trainer.value_coefficient,
        max_gradient_norm=meta_trainer.max_gradient_norm,
        target_kl=meta_trainer.target_kl,
    )
    if optimizer_name == "sgd":
        # MAML任务内梯度下降保留不同支持任务的梯度幅度，外循环仍使用Adam。
        trainer.actor_optimizer = torch.optim.SGD(
            actor.parameters(),
            lr=inner_learning_rate,
        )
        trainer.critic_optimizer = torch.optim.SGD(
            critic.parameters(),
            lr=inner_learning_rate,
        )
    return trainer


def compute_query_gradients(
    task_trainer: PPOTrainer,
    batch: PPOBatch,
) -> QueryGradientResult:
    """计算适应后参数上的查询梯度，不执行任务内或元参数更新。"""

    new_log_probability, entropy = task_trainer.actor.evaluate_actions(
        batch.states,
        batch.actions,
    )
    actor_loss, approximate_kl, clip_fraction = clipped_surrogate_loss(
        new_log_probability,
        batch.old_log_probabilities,
        batch.advantages,
        task_trainer.clip_range,
    )
    predicted_values = task_trainer.critic(batch.states)
    critic_loss = torch.mean((predicted_values - batch.returns) ** 2)
    entropy_mean = entropy.mean()
    actor_objective = actor_loss - task_trainer.entropy_coefficient * entropy_mean
    critic_objective = task_trainer.value_coefficient * critic_loss
    if not torch.isfinite(actor_objective) or not torch.isfinite(critic_objective):
        raise FloatingPointError("FOMAML 查询目标出现 NaN 或无穷值。")

    actor_gradients = torch.autograd.grad(
        actor_objective,
        tuple(task_trainer.actor.parameters()),
    )
    critic_gradients = torch.autograd.grad(
        critic_objective,
        tuple(task_trainer.critic.parameters()),
    )
    detached_actor = tuple(gradient.detach().clone() for gradient in actor_gradients)
    detached_critic = tuple(gradient.detach().clone() for gradient in critic_gradients)
    if any(not torch.isfinite(gradient).all() for gradient in (*detached_actor, *detached_critic)):
        raise FloatingPointError("FOMAML 查询梯度出现 NaN 或无穷值。")
    return QueryGradientResult(
        actor_gradients=detached_actor,
        critic_gradients=detached_critic,
        actor_loss=float(actor_loss.detach().item()),
        critic_loss=float(critic_loss.detach().item()),
        entropy=float(entropy_mean.detach().item()),
        approximate_kl=float(approximate_kl.detach().item()),
        clip_fraction=float(clip_fraction.detach().item()),
    )


def differentiable_fast_adaptation(
    meta_trainer: PPOTrainer,
    batch: PPOBatch,
    *,
    inner_learning_rate: float,
    shared_offset: bool = False,
    shared_auxiliary_signal: float = 0.0,
    shared_auxiliary_coefficient: float = 0.0,
    shared_policy_gradient_coefficient: float = 1.0,
) -> DifferentiableFastAdaptationResult:
    """对专家均值偏置做一步可微SGD，并保留二阶计算图。"""

    if inner_learning_rate <= 0.0:
        raise ValueError("快参数内学习率必须为正。")
    if (
        not np.isfinite(shared_auxiliary_signal)
        or shared_auxiliary_coefficient < 0.0
        or shared_policy_gradient_coefficient < 0.0
    ):
        raise ValueError("共享辅助信号必须有限，两个梯度系数必须非负。")
    if not shared_offset and (
        shared_auxiliary_coefficient != 0.0
        or shared_policy_gradient_coefficient != 1.0
    ):
        raise ValueError("共享辅助校准只能用于团队级共享快参数。")
    if not hasattr(meta_trainer.actor, "fast_adaptation_parameters"):
        raise TypeError("二阶快适应要求连续专家编号Actor。")
    fast_parameters = meta_trainer.actor.fast_adaptation_parameters()
    if len(fast_parameters) != 1:
        raise ValueError("当前二阶实现要求一个五维专家均值偏置张量。")
    offsets = fast_parameters[0]
    new_log_probability, entropy = meta_trainer.actor.evaluate_actions(
        batch.states,
        batch.actions,
        expert_mean_offsets=offsets,
    )
    actor_loss, approximate_kl, clip_fraction = clipped_surrogate_loss(
        new_log_probability,
        batch.old_log_probabilities,
        batch.advantages,
        meta_trainer.clip_range,
    )
    entropy_mean = entropy.mean()
    actor_objective = actor_loss - meta_trainer.entropy_coefficient * entropy_mean
    if not torch.isfinite(actor_objective):
        raise FloatingPointError("二阶MAML支持集Actor目标出现 NaN 或无穷值。")
    gradient = torch.autograd.grad(
        actor_objective,
        offsets,
        create_graph=True,
    )[0]
    # 个体档案任务使用五维独立梯度；团队接纳度任务只适应一个共享方向，
    # 等价于给五位专家的状态依赖动作共同增加一个团队级校准偏置。
    shared_policy_gradient = gradient.sum()
    shared_auxiliary_gradient = torch.as_tensor(
        shared_auxiliary_coefficient * shared_auxiliary_signal,
        dtype=gradient.dtype,
        device=gradient.device,
    )
    shared_combined_gradient = (
        shared_policy_gradient_coefficient * shared_policy_gradient
        + shared_auxiliary_gradient
    )
    update_gradient = (
        shared_combined_gradient.expand_as(gradient)
        if shared_offset
        else gradient
    )
    gradient_norm = (
        torch.abs(shared_combined_gradient)
        if shared_offset
        else torch.linalg.vector_norm(gradient)
    )
    clip_scale = torch.clamp(
        meta_trainer.max_gradient_norm / (gradient_norm + 1.0e-8),
        max=1.0,
    )
    adapted_offsets = offsets - inner_learning_rate * update_gradient * clip_scale

    with torch.no_grad():
        predicted_values = meta_trainer.critic(batch.states)
        critic_loss = torch.mean((predicted_values - batch.returns) ** 2)
        target_variance = torch.var(batch.returns, unbiased=False)
        explained_variance = (
            1.0
            - torch.var(batch.returns - predicted_values, unbiased=False)
            / torch.clamp(target_variance, min=1.0e-8)
        )
    return DifferentiableFastAdaptationResult(
        adapted_offsets=adapted_offsets,
        metrics=PPOUpdateMetrics(
            actor_loss=float(actor_loss.detach().item()),
            critic_loss=float(critic_loss.detach().item()),
            entropy=float(entropy_mean.detach().item()),
            approximate_kl=float(approximate_kl.detach().item()),
            clip_fraction=float(clip_fraction.detach().item()),
            actor_gradient_norm=float(gradient_norm.detach().item()),
            critic_gradient_norm=0.0,
            explained_variance=float(explained_variance.item()),
            epochs_completed=1,
            minibatches=1,
        ),
        shared_policy_gradient=(
            float(shared_policy_gradient.detach().item()) if shared_offset else None
        ),
        shared_auxiliary_gradient=(
            float(shared_auxiliary_gradient.detach().item()) if shared_offset else None
        ),
        shared_combined_gradient=(
            float(shared_combined_gradient.detach().item()) if shared_offset else None
        ),
    )


def compute_second_order_fast_query_gradients(
    meta_trainer: PPOTrainer,
    adapted_offsets: torch.Tensor,
    batch: PPOBatch,
) -> QueryGradientResult:
    """通过五维可微内更新计算真正的二阶Actor元梯度。"""

    new_log_probability, entropy = meta_trainer.actor.evaluate_actions(
        batch.states,
        batch.actions,
        expert_mean_offsets=adapted_offsets,
    )
    actor_loss, approximate_kl, clip_fraction = clipped_surrogate_loss(
        new_log_probability,
        batch.old_log_probabilities,
        batch.advantages,
        meta_trainer.clip_range,
    )
    predicted_values = meta_trainer.critic(batch.states)
    critic_loss = torch.mean((predicted_values - batch.returns) ** 2)
    entropy_mean = entropy.mean()
    actor_objective = actor_loss - meta_trainer.entropy_coefficient * entropy_mean
    critic_objective = meta_trainer.value_coefficient * critic_loss
    if not torch.isfinite(actor_objective) or not torch.isfinite(critic_objective):
        raise FloatingPointError("二阶MAML查询目标出现 NaN 或无穷值。")

    actor_gradients = torch.autograd.grad(
        actor_objective,
        tuple(meta_trainer.actor.parameters()),
    )
    critic_gradients = torch.autograd.grad(
        critic_objective,
        tuple(meta_trainer.critic.parameters()),
    )
    detached_actor = tuple(gradient.detach().clone() for gradient in actor_gradients)
    detached_critic = tuple(gradient.detach().clone() for gradient in critic_gradients)
    if any(
        not torch.isfinite(gradient).all()
        for gradient in (*detached_actor, *detached_critic)
    ):
        raise FloatingPointError("二阶MAML查询梯度出现 NaN 或无穷值。")
    return QueryGradientResult(
        actor_gradients=detached_actor,
        critic_gradients=detached_critic,
        actor_loss=float(actor_loss.detach().item()),
        critic_loss=float(critic_loss.detach().item()),
        entropy=float(entropy_mean.detach().item()),
        approximate_kl=float(approximate_kl.detach().item()),
        clip_fraction=float(clip_fraction.detach().item()),
    )


def aggregate_fomaml_gradients(
    task_gradients: Sequence[Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    """按参数位置平均多个任务的一阶查询梯度。"""

    if not task_gradients:
        raise ValueError("至少需要一个任务梯度。")
    parameter_count = len(task_gradients[0])
    if parameter_count == 0 or any(len(task) != parameter_count for task in task_gradients):
        raise ValueError("所有任务必须提供等长的非空参数梯度。")
    averaged: list[torch.Tensor] = []
    for parameter_index in range(parameter_count):
        gradients = [task[parameter_index] for task in task_gradients]
        reference_shape = gradients[0].shape
        if any(gradient.shape != reference_shape for gradient in gradients):
            raise ValueError("同一参数位置的任务梯度形状必须一致。")
        stacked = torch.stack(gradients)
        if not torch.isfinite(stacked).all():
            raise FloatingPointError("待聚合元梯度出现 NaN 或无穷值。")
        averaged.append(stacked.mean(dim=0))
    return tuple(averaged)


def gradient_pairwise_cosines(
    task_gradients: Sequence[Sequence[torch.Tensor]],
) -> dict[str, float]:
    """计算任务梯度展平后的两两余弦相似度。"""

    if len(task_gradients) < 2:
        return {"mean": 1.0, "min": 1.0, "max": 1.0}
    flattened = [
        torch.cat([gradient.reshape(-1) for gradient in task])
        for task in task_gradients
    ]
    if any(not torch.isfinite(vector).all() for vector in flattened):
        raise FloatingPointError("余弦诊断中的任务梯度出现 NaN 或无穷值。")
    cosines: list[float] = []
    for left_index in range(len(flattened)):
        for right_index in range(left_index + 1, len(flattened)):
            left = flattened[left_index]
            right = flattened[right_index]
            denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
            cosine = (
                torch.dot(left, right) / denominator
                if float(denominator.item()) > 1.0e-12
                else torch.zeros((), device=left.device)
            )
            cosines.append(float(cosine.item()))
    values = np.asarray(cosines, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


class MetaGradientOptimizer:
    """聚合并施加查询元梯度；兼容一阶或二阶梯度来源。"""

    def __init__(
        self,
        meta_trainer: PPOTrainer,
        *,
        meta_learning_rate: float,
        actor_fast_only: bool = False,
        shared_actor_fast_update: bool = False,
    ) -> None:
        if meta_learning_rate <= 0.0:
            raise ValueError("元学习率必须为正。")
        self.meta_trainer = meta_trainer
        self.actor_fast_only = bool(actor_fast_only)
        self.shared_actor_fast_update = bool(shared_actor_fast_update)
        all_actor_parameters = tuple(meta_trainer.actor.parameters())
        if self.actor_fast_only:
            if not hasattr(meta_trainer.actor, "fast_adaptation_parameters"):
                raise TypeError("快参数外循环要求Actor暴露fast_adaptation_parameters。")
            self.actor_parameters = tuple(
                meta_trainer.actor.fast_adaptation_parameters()
            )
            if not self.actor_parameters:
                raise ValueError("快参数外循环至少需要一个Actor参数。")
            self.actor_parameter_indices = tuple(
                next(
                    index
                    for index, parameter in enumerate(all_actor_parameters)
                    if parameter is fast_parameter
                )
                for fast_parameter in self.actor_parameters
            )
        else:
            self.actor_parameters = all_actor_parameters
            self.actor_parameter_indices = tuple(range(len(all_actor_parameters)))
        if self.shared_actor_fast_update and (
            not self.actor_fast_only or len(self.actor_parameters) != 1
        ):
            raise ValueError("共享外循环只支持单个快参数张量。")
        self.actor_optimizer = torch.optim.Adam(
            self.actor_parameters,
            lr=meta_learning_rate,
            eps=1.0e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            meta_trainer.critic.parameters(),
            lr=meta_learning_rate,
            eps=1.0e-5,
        )

    def step(self, results: Sequence[QueryGradientResult]) -> MetaUpdateMetrics:
        if not results:
            raise ValueError("外循环至少需要一个元任务。")
        actor_task_gradients = [
            tuple(
                result.actor_gradients[index]
                for index in self.actor_parameter_indices
            )
            for result in results
        ]
        if self.shared_actor_fast_update:
            actor_task_gradients = [
                (gradients[0].sum().expand_as(gradients[0]),)
                for gradients in actor_task_gradients
            ]
        critic_task_gradients = [result.critic_gradients for result in results]
        actor_cosines = gradient_pairwise_cosines(actor_task_gradients)
        critic_cosines = gradient_pairwise_cosines(critic_task_gradients)
        actor_gradients = aggregate_fomaml_gradients(actor_task_gradients)
        critic_gradients = aggregate_fomaml_gradients(critic_task_gradients)

        self.actor_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(
            self.actor_parameters,
            actor_gradients,
        ):
            parameter.grad = gradient.to(parameter.device).clone()
        actor_norm = nn.utils.clip_grad_norm_(
            self.actor_parameters,
            self.meta_trainer.max_gradient_norm,
        )
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(
            self.meta_trainer.critic.parameters(),
            critic_gradients,
        ):
            parameter.grad = gradient.to(parameter.device).clone()
        critic_norm = nn.utils.clip_grad_norm_(
            self.meta_trainer.critic.parameters(),
            self.meta_trainer.max_gradient_norm,
        )
        self.critic_optimizer.step()
        if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
            raise FloatingPointError("元梯度范数出现 NaN 或无穷值。")
        return MetaUpdateMetrics(
            task_count=len(results),
            actor_gradient_norm=float(actor_norm.detach().item()),
            critic_gradient_norm=float(critic_norm.detach().item()),
            actor_pairwise_cosine_mean=actor_cosines["mean"],
            actor_pairwise_cosine_min=actor_cosines["min"],
            critic_pairwise_cosine_mean=critic_cosines["mean"],
            critic_pairwise_cosine_min=critic_cosines["min"],
            mean_query_actor_loss=float(np.mean([item.actor_loss for item in results])),
            mean_query_critic_loss=float(np.mean([item.critic_loss for item in results])),
            mean_query_entropy=float(np.mean([item.entropy for item in results])),
            mean_query_approximate_kl=float(
                np.mean([item.approximate_kl for item in results])
            ),
            mean_query_clip_fraction=float(
                np.mean([item.clip_fraction for item in results])
            ),
        )


# 兼容早期FOMAML实验入口；新二阶响应弹性代码使用语义中性的主名称。
FirstOrderMetaOptimizer = MetaGradientOptimizer
