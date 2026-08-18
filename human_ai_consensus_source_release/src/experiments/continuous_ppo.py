"""连续倍率 PPO 的轨迹收集、评价与训练器构造。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch

from src.agents.networks import ContinuousFactorizedActor, ValueNetwork
from src.agents.ppo import ContinuousRolloutBuffer, PPOTrainer
from src.data.task_generator import StageBInstance, generate_stage_b_instance
from src.env.consensus_env import ConsensusFeedbackEnv
from src.env.response_model import (
    sample_response_types,
    sample_response_types_from_counts,
    suggestion_bin,
)
from src.experiments.train_ppo import ValidationCase


ContinuousActionSelector = Callable[[np.ndarray, ConsensusFeedbackEnv], np.ndarray]


@dataclass(frozen=True)
class ContinuousEpisodeEvaluation:
    """一个连续倍率策略回合的完整评价。"""

    initial_success: bool
    success: bool
    timeout: bool
    optimizer_failed: bool
    rounds: int
    final_min_acd: float
    final_mean_acd: float
    first_step_reward: float
    total_reward: float
    total_modification: float
    total_consensus_improvement: float
    total_modification_cost: float
    total_recommendation_cost: float
    total_remaining_deficit_cost: float
    total_unexecuted_recommendation_cost: float
    total_round_cost: float
    total_success_bonus: float
    total_timeout_penalty: float
    active_multipliers: tuple[float, ...]

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def aggregate_continuous_episodes(
    episodes: list[ContinuousEpisodeEvaluation],
) -> dict[str, object]:
    """聚合连续策略的效果指标和实际使用倍率分布。"""

    if not episodes:
        raise ValueError("至少需要一个连续策略回合。")
    multipliers = np.asarray(
        [value for episode in episodes for value in episode.active_multipliers],
        dtype=np.float64,
    )
    return {
        "episode_count": len(episodes),
        "initial_success_rate": float(np.mean([item.initial_success for item in episodes])),
        "success_rate": float(np.mean([item.success for item in episodes])),
        "timeout_rate": float(np.mean([item.timeout for item in episodes])),
        "optimizer_failure_rate": float(
            np.mean([item.optimizer_failed for item in episodes])
        ),
        "mean_rounds": float(np.mean([item.rounds for item in episodes])),
        "mean_final_min_acd": float(
            np.mean([item.final_min_acd for item in episodes])
        ),
        "mean_final_mean_acd": float(
            np.mean([item.final_mean_acd for item in episodes])
        ),
        "mean_first_step_reward": float(
            np.mean([item.first_step_reward for item in episodes])
        ),
        "mean_total_reward": float(np.mean([item.total_reward for item in episodes])),
        "mean_total_modification": float(
            np.mean([item.total_modification for item in episodes])
        ),
        "mean_reward_components": {
            "consensus_improvement": float(
                np.mean([item.total_consensus_improvement for item in episodes])
            ),
            "modification_cost": float(
                np.mean([item.total_modification_cost for item in episodes])
            ),
            "recommendation_cost": float(
                np.mean([item.total_recommendation_cost for item in episodes])
            ),
            "remaining_deficit_cost": float(
                np.mean([item.total_remaining_deficit_cost for item in episodes])
            ),
            "unexecuted_recommendation_cost": float(
                np.mean(
                    [item.total_unexecuted_recommendation_cost for item in episodes]
                )
            ),
            "round_cost": float(np.mean([item.total_round_cost for item in episodes])),
            "success_bonus": float(
                np.mean([item.total_success_bonus for item in episodes])
            ),
            "timeout_penalty": float(
                np.mean([item.total_timeout_penalty for item in episodes])
            ),
        },
        "active_multiplier_count": int(multipliers.size),
        "active_multiplier_mean": float(multipliers.mean()) if multipliers.size else 0.0,
        "active_multiplier_std": float(multipliers.std()) if multipliers.size else 0.0,
        "active_multiplier_min": float(multipliers.min()) if multipliers.size else 0.0,
        "active_multiplier_max": float(multipliers.max()) if multipliers.size else 0.0,
    }


def create_continuous_trainer(
    config: dict[str, Any],
    device: torch.device,
    *,
    learning_rate: float | None = None,
    entropy_coefficient: float = 1.0e-4,
    minibatch_size: int | None = None,
    preferred_multiplier: float = 1.0,
    initial_log_std: float = -1.0,
    include_expert_identity: bool = False,
    actor_initialization: str = "static",
    residual_head_gain: float = 0.15,
) -> PPOTrainer:
    """建立连续逐专家 PPO，可用于静态倍率或无优化器的直接建议量。"""

    ppo = config["ppo"]
    state_dim = int(config["data"]["num_experts"]) * 6 + 3
    guidance = config.get("guidance", {})
    bounds = np.asarray(
        guidance["action_bounds"]
        if guidance.get("mode") == "direct"
        else config["response"]["multipliers"],
        dtype=np.float64,
    )
    actor = ContinuousFactorizedActor(
        state_dim,
        int(config["data"]["num_experts"]),
        ppo["hidden_sizes"],
        ppo["activation"],
        multiplier_low=float(bounds.min()),
        multiplier_high=float(bounds.max()),
        include_expert_identity=include_expert_identity,
    ).to(device)
    if actor_initialization == "static":
        actor.initialize_multiplier_prior(preferred_multiplier, initial_log_std)
    elif actor_initialization == "residual":
        actor.initialize_multiplier_residual_prior(
            preferred_multiplier,
            initial_log_std,
            residual_head_gain,
        )
    elif actor_initialization == "random":
        # 保留网络构造时的标准随机权重与随机方差头，不注入任何建议量先验。
        pass
    else:
        raise ValueError("连续Actor初始化只支持static、residual或random。")
    critic = ValueNetwork(state_dim, ppo["hidden_sizes"], ppo["activation"]).to(device)
    return PPOTrainer(
        actor,
        critic,
        learning_rate=(
            float(ppo["learning_rate"]) if learning_rate is None else learning_rate
        ),
        clip_range=float(ppo["clip_range"]),
        update_epochs=int(ppo["update_epochs"]),
        minibatch_size=(
            int(ppo["minibatch_size"]) if minibatch_size is None else minibatch_size
        ),
        entropy_coefficient=float(entropy_coefficient),
        value_coefficient=float(ppo["value_coefficient"]),
        max_gradient_norm=float(ppo["max_gradient_norm"]),
        target_kl=float(ppo["target_kl"]),
    )


def run_continuous_policy_episode(
    env: ConsensusFeedbackEnv,
    instance: StageBInstance,
    action_selector: ContinuousActionSelector,
) -> ContinuousEpisodeEvaluation:
    """运行一个连续倍率回合，并审计全部奖励分量。"""

    state, reset_info = env.reset(instance)
    total_reward = 0.0
    first_step_reward = 0.0
    total_modification = 0.0
    component_totals = {
        "consensus_improvement": 0.0,
        "modification_cost": 0.0,
        "recommendation_cost": 0.0,
        "remaining_deficit_cost": 0.0,
        "unexecuted_recommendation_cost": 0.0,
        "round_cost": 0.0,
        "success_bonus": 0.0,
        "timeout_penalty": 0.0,
    }
    active_multipliers: list[float] = []
    timeout = False
    optimizer_failed = not bool(reset_info["optimizer_success"])
    while not env.done:
        multipliers = np.asarray(action_selector(state, env), dtype=np.float64)
        state, reward, _, _, info = env.step_continuous(multipliers)
        if env.round_index == 1:
            first_step_reward = float(reward)
        total_reward += float(reward)
        total_modification += float(info["reward"]["mean_modification"])
        for key in component_totals:
            component_totals[key] += float(info["reward"][key])
        active = np.asarray(
            info.get(
                "active_expert_mask",
                np.asarray(info["theoretical_deltas"]) > 1.0e-12,
            ),
            dtype=bool,
        )
        active_multipliers.extend(np.asarray(info["multipliers"])[active].tolist())
        timeout = bool(info["timeout"])
        optimizer_failed = bool(info["optimizer_failed"])
    return ContinuousEpisodeEvaluation(
        initial_success=bool(reset_info["initial_success"]),
        success=env.success,
        timeout=timeout,
        optimizer_failed=optimizer_failed,
        rounds=env.round_index,
        final_min_acd=float(env.metrics.min_acd),
        final_mean_acd=float(env.metrics.mean_acd),
        first_step_reward=first_step_reward,
        total_reward=total_reward,
        total_modification=total_modification,
        total_consensus_improvement=component_totals["consensus_improvement"],
        total_modification_cost=component_totals["modification_cost"],
        total_recommendation_cost=component_totals["recommendation_cost"],
        total_remaining_deficit_cost=component_totals["remaining_deficit_cost"],
        total_unexecuted_recommendation_cost=component_totals[
            "unexecuted_recommendation_cost"
        ],
        total_round_cost=component_totals["round_cost"],
        total_success_bonus=component_totals["success_bonus"],
        total_timeout_penalty=component_totals["timeout_penalty"],
        active_multipliers=tuple(float(value) for value in active_multipliers),
    )


def evaluate_continuous_policy_on_cases(
    config: dict[str, Any],
    cases: list[ValidationCase],
    action_selector: ContinuousActionSelector,
) -> tuple[dict[str, object], list[ContinuousEpisodeEvaluation]]:
    """在固定任务与固定响应噪声上评价任意连续策略。"""

    episodes = []
    for case in cases:
        env = ConsensusFeedbackEnv(
            config,
            np.random.default_rng(case.response_seed),
            response_types=case.response_types,
        )
        episodes.append(
            run_continuous_policy_episode(env, case.instance, action_selector)
        )
    return aggregate_continuous_episodes(episodes), episodes


def evaluate_continuous_trainer(
    trainer: PPOTrainer,
    config: dict[str, Any],
    cases: list[ValidationCase],
    *,
    deterministic: bool,
    action_seed: int | None = None,
) -> tuple[dict[str, object], list[ContinuousEpisodeEvaluation]]:
    """评价连续 PPO；随机评价不消耗训练阶段的Torch随机流。"""

    def evaluate() -> tuple[dict[str, object], list[ContinuousEpisodeEvaluation]]:
        return evaluate_continuous_policy_on_cases(
            config,
            cases,
            lambda state, _: trainer.act(state, deterministic=deterministic)[0],
        )

    if deterministic or action_seed is None:
        return evaluate()
    devices: list[int] = []
    if trainer.device.type == "cuda":
        devices.append(
            torch.cuda.current_device()
            if trainer.device.index is None
            else trainer.device.index
        )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(action_seed))
        return evaluate()


def collect_continuous_rollout(
    trainer: PPOTrainer,
    config: dict[str, Any],
    *,
    task_rng: np.random.Generator,
    type_rng: np.random.Generator,
    response_seed_rng: np.random.Generator,
    episode_target: int,
    fixed_response_composition: tuple[int, ...] | None = None,
    fixed_response_types: tuple[str, ...] | None = None,
) -> tuple[ContinuousRolloutBuffer, dict[str, object]]:
    """按完整回合收集连续 on-policy 轨迹，避免切断任务边界。"""

    if episode_target <= 0:
        raise ValueError("连续轨迹目标回合数必须为正。")
    response = config["response"]
    if fixed_response_composition is not None and fixed_response_types is not None:
        raise ValueError("固定响应人数构成与固定专家档案不能同时指定。")
    if fixed_response_composition is not None:
        if (
            len(fixed_response_composition) != len(response["type_names"])
            or any(int(value) < 0 for value in fixed_response_composition)
            or sum(int(value) for value in fixed_response_composition)
            != int(config["data"]["num_experts"])
        ):
            raise ValueError("固定响应人数构成与专家数不一致。")
    if fixed_response_types is not None:
        known_types = set(str(value) for value in response["type_names"])
        if (
            len(fixed_response_types) != int(config["data"]["num_experts"])
            or any(value not in known_types for value in fixed_response_types)
        ):
            raise ValueError("固定专家档案必须为每位专家提供一个已知响应类型。")

    buffer = ContinuousRolloutBuffer()
    rewards: list[float] = []
    successes: list[bool] = []
    rounds: list[int] = []
    active_multipliers: list[float] = []
    active_response_rates: list[float] = []
    active_recommendations: list[float] = []
    suggestion_bin_counts = np.zeros(3, dtype=np.int64)
    suggestion_bin_response_sums = np.zeros(3, dtype=np.float64)
    while len(rewards) < episode_target:
        instance = generate_stage_b_instance(config, task_rng)
        response_types = (
            tuple(fixed_response_types)
            if fixed_response_types is not None
            else sample_response_types_from_counts(
                response["type_names"],
                fixed_response_composition,
                type_rng,
            )
            if fixed_response_composition is not None
            else sample_response_types(
                int(config["data"]["num_experts"]),
                response["type_names"],
                response["type_probabilities"],
                type_rng,
            )
        )
        env = ConsensusFeedbackEnv(
            config,
            np.random.default_rng(
                int(response_seed_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            ),
            response_types=response_types,
        )
        state, _ = env.reset(instance)
        episode_reward = 0.0
        while not env.done:
            action, log_probability, value = trainer.act(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step_continuous(action)
            done = bool(terminated or truncated)
            buffer.add(
                state,
                action,
                reward,
                value,
                log_probability,
                done,
                factorized_reward=np.asarray(
                    info["factorized_reward"]["total"],
                    dtype=np.float32,
                ),
            )
            active = np.asarray(
                info.get(
                    "active_expert_mask",
                    np.asarray(info["theoretical_deltas"]) > 1.0e-12,
                ),
                dtype=bool,
            )
            active_multipliers.extend(np.asarray(action)[active].tolist())
            active_response_rates.extend(
                np.asarray(info["response_rates"])[active].tolist()
            )
            active_recommendations.extend(
                np.asarray(info["recommended_deltas"])[active].tolist()
            )
            for recommendation, response_rate in zip(
                np.asarray(info["recommended_deltas"])[active],
                np.asarray(info["response_rates"])[active],
                strict=True,
            ):
                bin_index = suggestion_bin(
                    float(recommendation),
                    response["suggestion_bins"],
                )
                suggestion_bin_counts[bin_index] += 1
                suggestion_bin_response_sums[bin_index] += float(response_rate)
            episode_reward += float(reward)
            state = next_state
        rewards.append(episode_reward)
        successes.append(env.success)
        rounds.append(env.round_index)

    values = np.asarray(active_multipliers, dtype=np.float64)
    response_values = np.asarray(active_response_rates, dtype=np.float64)
    recommendation_values = np.asarray(active_recommendations, dtype=np.float64)
    return buffer, {
        "transitions": len(buffer),
        "episodes": len(rewards),
        "mean_episode_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_rounds": float(np.mean(rounds)),
        "active_multiplier_count": int(values.size),
        "active_multiplier_mean": float(values.mean()) if values.size else 0.0,
        "active_multiplier_std": float(values.std()) if values.size else 0.0,
        "active_multiplier_min": float(values.min()) if values.size else 0.0,
        "active_multiplier_max": float(values.max()) if values.size else 0.0,
        # 这两个量完全来自历史交互，不暴露隐藏响应类型或任务偏移。
        # 元内循环可用它们估计“建议发出后实际被接受了多少”。
        "active_response_rate_mean": (
            float(response_values.mean()) if response_values.size else 0.0
        ),
        "active_response_rate_std": (
            float(response_values.std()) if response_values.size else 0.0
        ),
        "active_recommendation_mean": (
            float(recommendation_values.mean())
            if recommendation_values.size
            else 0.0
        ),
        "suggestion_bin_counts": suggestion_bin_counts.tolist(),
        "response_rate_mean_by_bin": [
            (
                float(suggestion_bin_response_sums[index] / count)
                if count > 0
                else None
            )
            for index, count in enumerate(suggestion_bin_counts)
        ],
    }
