"""GAE、PPO 裁剪更新、轨迹缓存与检查点。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.agents.networks import FactorizedActor, ValueNetwork


@dataclass(frozen=True)
class PPOBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


@dataclass(frozen=True)
class PPOUpdateMetrics:
    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    actor_gradient_norm: float
    critic_gradient_norm: float
    explained_variance: float
    epochs_completed: int
    minibatches: int

    def to_serializable(self) -> dict[str, float | int]:
        return asdict(self)


def balanced_minibatch_indices(
    indices: np.ndarray,
    minibatch_size: int,
) -> list[np.ndarray]:
    """把索引均衡切分，避免最后一个极小批次主导一次梯度更新。"""

    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim != 1 or index_array.size == 0:
        raise ValueError("批次索引必须是非空一维数组。")
    if minibatch_size < 1:
        raise ValueError("minibatch_size 必须为正整数。")
    number_of_batches = int(np.ceil(index_array.size / minibatch_size))
    return [part for part in np.array_split(index_array, number_of_batches) if part.size]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 GAE；done=True 的成功或超时转移均不进行价值自举。"""

    reward_array = np.asarray(rewards, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    done_array = np.asarray(dones, dtype=bool)
    if reward_array.ndim != 1 or value_array.shape != reward_array.shape:
        raise ValueError("奖励与价值必须是等长一维数组。")
    if done_array.shape != reward_array.shape:
        raise ValueError("终止标记与奖励数组形状必须一致。")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma 和 GAE lambda 必须位于 [0, 1]。")

    advantages = np.zeros_like(reward_array)
    running_advantage = 0.0
    for index in range(reward_array.size - 1, -1, -1):
        nonterminal = 1.0 - float(done_array[index])
        following_value = (
            float(next_value) if index == reward_array.size - 1 else value_array[index + 1]
        )
        delta = (
            reward_array[index]
            + gamma * following_value * nonterminal
            - value_array[index]
        )
        running_advantage = (
            delta + gamma * gae_lambda * nonterminal * running_advantage
        )
        advantages[index] = running_advantage
    returns = advantages + value_array
    return advantages.astype(np.float32), returns.astype(np.float32)


def compute_discounted_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """计算不依赖Critic的蒙特卡洛折扣回报，并在episode边界清零。"""

    reward_array = np.asarray(rewards, dtype=np.float64)
    done_array = np.asarray(dones, dtype=bool)
    if reward_array.ndim != 1 or done_array.shape != reward_array.shape:
        raise ValueError("奖励与终止标记必须是等长一维数组。")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("折扣因子 gamma 必须位于 [0, 1]。")
    returns = np.zeros_like(reward_array)
    running_return = 0.0
    for index in range(reward_array.size - 1, -1, -1):
        running_return = reward_array[index] + gamma * running_return * (
            1.0 - float(done_array[index])
        )
        returns[index] = running_return
    return returns.astype(np.float32)


def compute_rollout_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    next_value: float,
    gamma: float,
    gae_lambda: float,
    advantage_estimator: str,
) -> tuple[np.ndarray, np.ndarray]:
    """为普通PPO选择GAE，或为元任务选择无价值偏差的蒙特卡洛回报。"""

    if advantage_estimator == "gae":
        return compute_gae(
            rewards,
            values,
            dones,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    if advantage_estimator == "monte_carlo":
        returns = compute_discounted_returns(rewards, dones, gamma=gamma)
        # 批内标准化会充当与动作无关的任务基线，避免共享Critic看不到隐藏档案时
        # 向MAML快参数注入有偏优势估计。
        return returns.copy(), returns
    raise ValueError("advantage_estimator 必须是 'gae' 或 'monte_carlo'。")


def clipped_surrogate_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 PPO Actor 损失、近似 KL 和裁剪比例。"""

    log_ratio = new_log_probabilities - old_log_probabilities
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    actor_loss = -torch.minimum(unclipped, clipped).mean()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    return actor_loss, approximate_kl, clip_fraction


class RolloutBuffer:
    """保存一批 on-policy 转移，并在结束时计算 GAE。"""

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probabilities: list[float] = []
        self.dones: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_probability: float,
        done: bool,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probabilities.append(float(log_probability))
        self.dones.append(bool(done))

    def to_batch(
        self,
        device: torch.device,
        *,
        gamma: float,
        gae_lambda: float,
        next_value: float = 0.0,
        normalize_advantages: bool = True,
        advantage_estimator: str = "gae",
    ) -> PPOBatch:
        if not self.rewards:
            raise ValueError("空轨迹缓存不能生成 PPO 批次。")
        advantages, returns = compute_rollout_targets(
            np.asarray(self.rewards),
            np.asarray(self.values),
            np.asarray(self.dones),
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
            advantage_estimator=advantage_estimator,
        )
        if normalize_advantages and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        return PPOBatch(
            states=torch.as_tensor(np.stack(self.states), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.stack(self.actions), dtype=torch.long, device=device),
            old_log_probabilities=torch.as_tensor(
                self.log_probabilities,
                dtype=torch.float32,
                device=device,
            ),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        )


class ContinuousRolloutBuffer(RolloutBuffer):
    """保存连续倍率动作，并复用同一套GAE计算。"""

    def __init__(self) -> None:
        super().__init__()
        self.factorized_rewards: list[np.ndarray] = []

    def extend(self, other: "ContinuousRolloutBuffer") -> None:
        """合并同一旧策略采集的完整轨迹，供一次PPO更新共同使用。"""

        if not isinstance(other, ContinuousRolloutBuffer):
            raise TypeError("只能合并连续动作轨迹缓存。")
        if not other.rewards:
            return
        self_has_factorized = bool(self.factorized_rewards)
        other_has_factorized = bool(other.factorized_rewards)
        if self.rewards and self_has_factorized != other_has_factorized:
            raise ValueError("不能混合含逐专家奖励和不含逐专家奖励的轨迹。")
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.log_probabilities.extend(other.log_probabilities)
        self.dones.extend(other.dones)
        self.factorized_rewards.extend(other.factorized_rewards)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_probability: float,
        done: bool,
        factorized_reward: np.ndarray | None = None,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probabilities.append(float(log_probability))
        self.dones.append(bool(done))
        if factorized_reward is not None:
            values = np.asarray(factorized_reward, dtype=np.float32)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("逐专家奖励必须是有限的一维数组。")
            if self.factorized_rewards and values.shape != self.factorized_rewards[0].shape:
                raise ValueError("同一轨迹中的逐专家奖励维度必须一致。")
            self.factorized_rewards.append(values.copy())

    def factorized_returns(
        self,
        device: torch.device,
        *,
        gamma: float,
    ) -> torch.Tensor:
        """计算逐专家回报到达值，终止边界不跨回合传播。"""

        if len(self.factorized_rewards) != len(self.rewards):
            raise ValueError("轨迹中的每个转移都必须提供逐专家奖励。")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("折扣因子 gamma 必须位于 [0, 1]。")
        rewards = np.stack(self.factorized_rewards).astype(np.float64, copy=False)
        returns = np.zeros_like(rewards)
        running = np.zeros(rewards.shape[1], dtype=np.float64)
        for index in range(rewards.shape[0] - 1, -1, -1):
            running = rewards[index] + gamma * running * (1.0 - float(self.dones[index]))
            returns[index] = running
        return torch.as_tensor(returns, dtype=torch.float32, device=device)

    def to_batch(
        self,
        device: torch.device,
        *,
        gamma: float,
        gae_lambda: float,
        next_value: float = 0.0,
        normalize_advantages: bool = True,
        advantage_estimator: str = "gae",
    ) -> PPOBatch:
        if not self.rewards:
            raise ValueError("空连续轨迹缓存不能生成 PPO 批次。")
        advantages, returns = compute_rollout_targets(
            np.asarray(self.rewards),
            np.asarray(self.values),
            np.asarray(self.dones),
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
            advantage_estimator=advantage_estimator,
        )
        if normalize_advantages and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        return PPOBatch(
            states=torch.as_tensor(np.stack(self.states), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.stack(self.actions), dtype=torch.float32, device=device),
            old_log_probabilities=torch.as_tensor(
                self.log_probabilities,
                dtype=torch.float32,
                device=device,
            ),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        )


class PPOTrainer:
    """联合优化 Actor 与 Critic 的 PPO 训练器。"""

    def __init__(
        self,
        actor: FactorizedActor,
        critic: ValueNetwork,
        *,
        learning_rate: float,
        clip_range: float,
        update_epochs: int,
        minibatch_size: int,
        entropy_coefficient: float,
        value_coefficient: float,
        max_gradient_norm: float,
        target_kl: float,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.clip_range = float(clip_range)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.entropy_coefficient = float(entropy_coefficient)
        self.value_coefficient = float(value_coefficient)
        self.max_gradient_norm = float(max_gradient_norm)
        self.target_kl = float(target_kl)
        # Actor 与 Critic 没有共享参数，分别优化和裁剪可避免价值梯度压制策略梯度。
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=float(learning_rate),
            eps=1.0e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=float(learning_rate),
            eps=1.0e-5,
        )

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions, log_probability, _ = self.actor.act(
            state_tensor,
            deterministic=deterministic,
        )
        value = self.critic(state_tensor)
        return (
            actions.squeeze(0).cpu().numpy(),
            float(log_probability.item()),
            float(value.item()),
        )

    def update(self, batch: PPOBatch, rng: np.random.Generator) -> PPOUpdateMetrics:
        """联合更新策略网络与价值网络。"""

        return self._update(batch, rng, update_critic=True)

    def update_actor_only(
        self,
        batch: PPOBatch,
        rng: np.random.Generator,
    ) -> PPOUpdateMetrics:
        """只更新策略网络，把Critic作为跨任务共享的固定价值基线。"""

        return self._update(batch, rng, update_critic=False)

    def update_actor_factorized(
        self,
        batch: PPOBatch,
        factorized_returns: torch.Tensor,
        rng: np.random.Generator,
    ) -> PPOUpdateMetrics:
        """用逐专家回报更新因子化连续Actor，Critic保持跨任务共享。"""

        batch_size = int(batch.states.shape[0])
        if batch_size < 1 or factorized_returns.shape != batch.actions.shape:
            raise ValueError("逐专家回报必须与连续动作批次形状一致。")
        if not hasattr(self.actor, "evaluate_actions_per_expert") or not hasattr(
            self.actor,
            "active_expert_mask",
        ):
            raise TypeError("逐专家信用分配只适用于连续因子化Actor。")
        with torch.no_grad():
            old_per_expert_log, _ = self.actor.evaluate_actions_per_expert(
                batch.states,
                batch.actions,
            )
            active_mask = self.actor.active_expert_mask(batch.states)
        if not bool(active_mask.any().item()):
            raise ValueError("逐专家PPO批次中至少要有一个有效动作。")

        # 仅在真正影响环境的动作上标准化，避免未激活专家注入伪梯度。
        advantages = torch.zeros_like(factorized_returns)
        active_returns = factorized_returns[active_mask]
        if active_returns.numel() > 1:
            normalized = (active_returns - active_returns.mean()) / (
                active_returns.std(unbiased=False) + 1.0e-8
            )
        else:
            normalized = active_returns
        advantages[active_mask] = normalized

        metric_lists: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approximate_kl": [],
            "clip_fraction": [],
            "actor_gradient_norm": [],
        }
        epochs_completed = 0
        stop_early = False
        for epoch in range(self.update_epochs):
            indices = rng.permutation(batch_size)
            for minibatch_indices in balanced_minibatch_indices(
                indices,
                self.minibatch_size,
            ):
                selected = torch.as_tensor(
                    minibatch_indices,
                    dtype=torch.long,
                    device=self.device,
                )
                selected_mask = active_mask[selected]
                if not bool(selected_mask.any().item()):
                    continue
                new_per_expert_log, per_expert_entropy = (
                    self.actor.evaluate_actions_per_expert(
                        batch.states[selected],
                        batch.actions[selected],
                    )
                )
                actor_loss, approximate_kl, clip_fraction = clipped_surrogate_loss(
                    new_per_expert_log[selected_mask],
                    old_per_expert_log[selected][selected_mask],
                    advantages[selected][selected_mask],
                    self.clip_range,
                )
                entropy_mean = per_expert_entropy[selected_mask].mean()
                actor_objective = actor_loss - self.entropy_coefficient * entropy_mean
                predicted_values = self.critic(batch.states[selected])
                critic_loss = torch.mean((predicted_values - batch.returns[selected]) ** 2)
                if not torch.isfinite(actor_objective) or not torch.isfinite(critic_loss):
                    raise FloatingPointError("逐专家PPO损失出现 NaN 或无穷值。")

                self.actor.zero_grad(set_to_none=True)
                actor_objective.backward()
                actor_gradient_norm = nn.utils.clip_grad_norm_(
                    self._actor_step_parameters(),
                    self.max_gradient_norm,
                )
                if not torch.isfinite(actor_gradient_norm):
                    raise FloatingPointError("逐专家Actor梯度出现 NaN 或无穷值。")
                self.actor_optimizer.step()

                values = {
                    "actor_loss": float(actor_loss.detach().item()),
                    "critic_loss": float(critic_loss.detach().item()),
                    "entropy": float(entropy_mean.detach().item()),
                    "approximate_kl": float(approximate_kl.detach().item()),
                    "clip_fraction": float(clip_fraction.detach().item()),
                    "actor_gradient_norm": float(actor_gradient_norm.detach().item()),
                }
                for key, value in values.items():
                    metric_lists[key].append(value)
                if values["approximate_kl"] > 1.5 * self.target_kl:
                    stop_early = True
                    break
            epochs_completed = epoch + 1
            if stop_early:
                break

        if not metric_lists["actor_loss"]:
            raise RuntimeError("逐专家PPO没有产生有效小批次。")
        with torch.no_grad():
            predictions = self.critic(batch.states)
            target_variance = torch.var(batch.returns, unbiased=False)
            explained_variance = (
                1.0
                - torch.var(batch.returns - predictions, unbiased=False)
                / torch.clamp(target_variance, min=1.0e-8)
            )
        return PPOUpdateMetrics(
            actor_loss=float(np.mean(metric_lists["actor_loss"])),
            critic_loss=float(np.mean(metric_lists["critic_loss"])),
            entropy=float(np.mean(metric_lists["entropy"])),
            approximate_kl=float(np.mean(metric_lists["approximate_kl"])),
            clip_fraction=float(np.mean(metric_lists["clip_fraction"])),
            actor_gradient_norm=float(np.mean(metric_lists["actor_gradient_norm"])),
            critic_gradient_norm=0.0,
            explained_variance=float(explained_variance.item()),
            epochs_completed=epochs_completed,
            minibatches=len(metric_lists["actor_loss"]),
        )

    def _actor_step_parameters(self) -> tuple[nn.Parameter, ...]:
        """返回当前Actor优化器真正负责的参数，支持五维快参数内循环。"""

        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for group in self.actor_optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if not parameters:
            raise RuntimeError("Actor优化器没有可更新参数。")
        return tuple(parameters)

    def _update(
        self,
        batch: PPOBatch,
        rng: np.random.Generator,
        *,
        update_critic: bool,
    ) -> PPOUpdateMetrics:
        batch_size = int(batch.states.shape[0])
        if batch_size < 1:
            raise ValueError("PPO 更新至少需要 1 个转移。")
        metric_lists: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approximate_kl": [],
            "clip_fraction": [],
            "actor_gradient_norm": [],
            "critic_gradient_norm": [],
        }
        epochs_completed = 0
        stop_early = False
        for epoch in range(self.update_epochs):
            indices = rng.permutation(batch_size)
            for minibatch_indices in balanced_minibatch_indices(
                indices,
                self.minibatch_size,
            ):
                selected = torch.as_tensor(
                    minibatch_indices,
                    dtype=torch.long,
                    device=self.device,
                )
                new_log_probability, entropy = self.actor.evaluate_actions(
                    batch.states[selected],
                    batch.actions[selected],
                )
                actor_loss, approximate_kl, clip_fraction = clipped_surrogate_loss(
                    new_log_probability,
                    batch.old_log_probabilities[selected],
                    batch.advantages[selected],
                    self.clip_range,
                )
                predicted_values = self.critic(batch.states[selected])
                critic_loss = torch.mean((predicted_values - batch.returns[selected]) ** 2)
                entropy_mean = entropy.mean()
                actor_objective = actor_loss - self.entropy_coefficient * entropy_mean
                critic_objective = self.value_coefficient * critic_loss
                if not torch.isfinite(actor_objective) or not torch.isfinite(
                    critic_objective
                ):
                    raise FloatingPointError("PPO 损失出现 NaN 或无穷值。")

                self.actor.zero_grad(set_to_none=True)
                actor_objective.backward()
                actor_gradient_norm = nn.utils.clip_grad_norm_(
                    self._actor_step_parameters(),
                    self.max_gradient_norm,
                )
                if not torch.isfinite(actor_gradient_norm):
                    raise FloatingPointError("Actor 梯度范数出现 NaN 或无穷值。")
                self.actor_optimizer.step()

                if update_critic:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_objective.backward()
                    critic_gradient_norm = nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.max_gradient_norm,
                    )
                    if not torch.isfinite(critic_gradient_norm):
                        raise FloatingPointError("Critic 梯度范数出现 NaN 或无穷值。")
                    self.critic_optimizer.step()
                else:
                    # 小支持集只负责识别任务并调整策略；价值函数留给外循环共享学习。
                    critic_gradient_norm = torch.zeros((), device=self.device)

                values = {
                    "actor_loss": float(actor_loss.detach().item()),
                    "critic_loss": float(critic_loss.detach().item()),
                    "entropy": float(entropy_mean.detach().item()),
                    "approximate_kl": float(approximate_kl.detach().item()),
                    "clip_fraction": float(clip_fraction.detach().item()),
                    "actor_gradient_norm": float(
                        actor_gradient_norm.detach().item()
                    ),
                    "critic_gradient_norm": float(
                        critic_gradient_norm.detach().item()
                    ),
                }
                for key, value in values.items():
                    metric_lists[key].append(value)
                if values["approximate_kl"] > 1.5 * self.target_kl:
                    stop_early = True
                    break
            epochs_completed = epoch + 1
            if stop_early:
                break

        with torch.no_grad():
            predictions = self.critic(batch.states)
            target_variance = torch.var(batch.returns, unbiased=False)
            explained_variance = (
                1.0
                - torch.var(batch.returns - predictions, unbiased=False)
                / torch.clamp(target_variance, min=1.0e-8)
            )
        return PPOUpdateMetrics(
            actor_loss=float(np.mean(metric_lists["actor_loss"])),
            critic_loss=float(np.mean(metric_lists["critic_loss"])),
            entropy=float(np.mean(metric_lists["entropy"])),
            approximate_kl=float(np.mean(metric_lists["approximate_kl"])),
            clip_fraction=float(np.mean(metric_lists["clip_fraction"])),
            actor_gradient_norm=float(
                np.mean(metric_lists["actor_gradient_norm"])
            ),
            critic_gradient_norm=float(
                np.mean(metric_lists["critic_gradient_norm"])
            ),
            explained_variance=float(explained_variance.item()),
            epochs_completed=epochs_completed,
            minibatches=len(metric_lists["actor_loss"]),
        )

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "metadata": metadata,
            },
            output,
        )
        return output

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_optimizer" in checkpoint and "critic_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return dict(checkpoint.get("metadata", {}))
