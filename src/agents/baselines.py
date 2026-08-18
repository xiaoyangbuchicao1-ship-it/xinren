"""训练前校准使用的固定、随机和响应感知贪心策略。"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from src.env.consensus_env import ConsensusFeedbackEnv, consensus_reached


IntArray = NDArray[np.int64]


class FixedMultiplierPolicy:
    """所有专家、所有轮次都使用同一个倍率索引。"""

    def __init__(self, action_index: int) -> None:
        self.action_index = int(action_index)

    def act(self, env: ConsensusFeedbackEnv) -> IntArray:
        if not 0 <= self.action_index < env.action_count:
            raise ValueError("固定动作索引超出环境动作范围。")
        return np.full(env.num_experts, self.action_index, dtype=np.int64)


class RandomPolicy:
    """为每位专家独立均匀采样倍率。"""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, env: ConsensusFeedbackEnv) -> IntArray:
        return self.rng.integers(
            0,
            env.action_count,
            size=env.num_experts,
            dtype=np.int64,
        )


def _candidate_actions(num_experts: int, action_count: int) -> IntArray:
    """按字典序枚举联合动作，5 人 4 动作为 1024 个组合。"""

    return np.asarray(
        list(product(range(action_count), repeat=num_experts)),
        dtype=np.int64,
    )


def oracle_one_step_rewards(
    env: ConsensusFeedbackEnv,
    candidate_actions: NDArray[np.integer],
) -> NDArray[np.float64]:
    """使用隐藏响应均值计算联合动作的一步期望奖励，不读取隐藏参考 Y。"""

    profiles = np.asarray(
        [
            env.config["response"]["response_table"][response_type]
            for response_type in env.response_types
        ],
        dtype=np.float64,
    )
    return one_step_rewards_from_response_profiles(env, candidate_actions, profiles)


def one_step_rewards_from_response_profiles(
    env: ConsensusFeedbackEnv,
    candidate_actions: NDArray[np.integer],
    response_profiles: NDArray[np.floating],
) -> NDArray[np.float64]:
    """按给定的每位专家三档响应均值计算一步期望奖励。"""

    if env.current_opinions is None or env.metrics is None or env.optimization is None:
        raise RuntimeError("策略必须在环境 reset() 后调用。")
    actions = np.asarray(candidate_actions, dtype=np.int64)
    expected_shape = (actions.shape[0], env.num_experts)
    if actions.ndim != 2 or actions.shape != expected_shape:
        raise ValueError("候选联合动作形状错误。")
    if np.any((actions < 0) | (actions >= env.action_count)):
        raise ValueError("候选动作索引超出范围。")
    profiles = np.asarray(response_profiles, dtype=np.float64)
    if profiles.shape != (env.num_experts, 3):
        raise ValueError("响应均值必须是形状为 [专家数, 3] 的数组。")
    if np.any((profiles < 0.0) | (profiles > 1.0)):
        raise ValueError("响应均值必须位于 [0, 1]。")

    response = env.config["response"]
    multipliers = np.asarray(response["multipliers"], dtype=np.float64)
    theoretical = env.optimization.deltas
    recommended = np.clip(multipliers[actions] * theoretical[None, :], 0.0, 1.0)
    internal_boundaries = np.asarray(response["suggestion_bins"][1:-1], dtype=np.float64)
    bin_indices = np.searchsorted(internal_boundaries, recommended, side="right")

    response_means = np.zeros_like(recommended)
    for expert in range(env.num_experts):
        response_means[:, expert] = profiles[expert, bin_indices[:, expert]]
    effective = recommended * response_means

    opinions = env.current_opinions
    reference = env.metrics.reference
    mask = env.metrics.issue_mask
    moved = (
        (1.0 - effective[:, :, None]) * opinions[None, :, :]
        + effective[:, :, None] * reference[None, None, :]
    )
    updated = np.where(mask[None, :, :], moved, opinions[None, :, :])

    differences = np.abs(updated[:, :, None, :] - updated[:, None, :, :])
    similarities = np.clip(1.0 - differences, 0.0, 1.0)
    ace = (similarities.sum(axis=2) - 1.0) / (env.num_experts - 1)
    acd = np.clip(ace.mean(axis=2), 0.0, 1.0)
    mean_acd = acd.mean(axis=1)
    consensus = env.config["consensus"]
    threshold = env.success_threshold
    tolerance = float(consensus["constraint_tolerance"])
    success = np.asarray(
        [consensus_reached(values, threshold, tolerance) for values in acd],
        dtype=bool,
    )
    timeout = (env.round_index + 1 >= env.max_rounds) & ~success

    mean_modification = np.mean(np.sum(np.abs(updated - opinions[None, :, :]), axis=2), axis=1)
    reward = env.config["reward"]
    return (
        float(reward["consensus_improvement_weight"])
        * (mean_acd - env.metrics.mean_acd)
        - float(reward["modification_cost_weight"]) * mean_modification
        - float(reward["round_cost"])
        + float(reward["success_bonus"]) * success.astype(np.float64)
        - float(reward["timeout_penalty"]) * timeout.astype(np.float64)
    )


class ResponseAwareGreedyOracle:
    """观察隐藏响应类型、枚举联合动作并最大化一步期望奖励。"""

    def __init__(self) -> None:
        self.last_evaluated_action_count = 0

    def act(self, env: ConsensusFeedbackEnv) -> IntArray:
        candidates = _candidate_actions(env.num_experts, env.action_count)
        rewards = oracle_one_step_rewards(env, candidates)
        self.last_evaluated_action_count = int(candidates.shape[0])
        return candidates[int(np.argmax(rewards))].copy()


class PriorAwareGreedyPolicy:
    """仅用总体类型先验，不读取当前任务隐藏响应类型的一步贪心策略。"""

    def __init__(self) -> None:
        self.last_evaluated_action_count = 0

    def act(self, env: ConsensusFeedbackEnv) -> IntArray:
        response = env.config["response"]
        probabilities = np.asarray(response["type_probabilities"], dtype=np.float64)
        table = np.asarray(
            [response["response_table"][name] for name in response["type_names"]],
            dtype=np.float64,
        )
        prior_profile = probabilities @ table
        profiles = np.repeat(prior_profile[None, :], env.num_experts, axis=0)
        candidates = _candidate_actions(env.num_experts, env.action_count)
        rewards = one_step_rewards_from_response_profiles(env, candidates, profiles)
        self.last_evaluated_action_count = int(candidates.shape[0])
        return candidates[int(np.argmax(rewards))].copy()
