"""非学习策略的统一回合执行与指标聚合。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from src.data.task_generator import StageBInstance
from src.env.consensus_env import ConsensusFeedbackEnv


class Policy(Protocol):
    def act(self, env: ConsensusFeedbackEnv) -> np.ndarray:
        """根据当前环境返回每位专家的离散动作。"""


@dataclass(frozen=True)
class EpisodeEvaluation:
    """单个非学习策略回合的评价。"""

    initial_success: bool
    success: bool
    timeout: bool
    optimizer_failed: bool
    rounds: int
    total_reward: float
    total_modification: float
    total_consensus_improvement: float
    total_modification_cost: float
    total_round_cost: float
    total_success_bonus: float
    total_timeout_penalty: float
    active_action_counts: tuple[int, ...]

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def run_policy_episode(
    env: ConsensusFeedbackEnv,
    instance: StageBInstance,
    policy: Policy,
) -> EpisodeEvaluation:
    """运行至成功、超时或理论求解失败。"""

    _, reset_info = env.reset(instance)
    initial_success = bool(reset_info["initial_success"])
    total_reward = 0.0
    total_modification = 0.0
    component_totals = {
        "consensus_improvement": 0.0,
        "modification_cost": 0.0,
        "round_cost": 0.0,
        "success_bonus": 0.0,
        "timeout_penalty": 0.0,
    }
    action_counts = np.zeros(env.action_count, dtype=np.int64)
    timeout = False
    optimizer_failed = not bool(reset_info["optimizer_success"])

    while not env.done:
        actions = np.asarray(policy.act(env), dtype=np.int64)
        _, reward, _, _, info = env.step(actions)
        total_reward += float(reward)
        total_modification += float(info["reward"]["mean_modification"])
        for key in component_totals:
            component_totals[key] += float(info["reward"][key])
        timeout = bool(info["timeout"])
        optimizer_failed = bool(info["optimizer_failed"])
        active = np.asarray(info["theoretical_deltas"]) > 1.0e-12
        action_counts += np.bincount(actions[active], minlength=env.action_count)

    assert env.metrics is not None
    return EpisodeEvaluation(
        initial_success=initial_success,
        success=env.success,
        timeout=timeout,
        optimizer_failed=optimizer_failed,
        rounds=env.round_index,
        total_reward=total_reward,
        total_modification=total_modification,
        total_consensus_improvement=component_totals["consensus_improvement"],
        total_modification_cost=component_totals["modification_cost"],
        total_round_cost=component_totals["round_cost"],
        total_success_bonus=component_totals["success_bonus"],
        total_timeout_penalty=component_totals["timeout_penalty"],
        active_action_counts=tuple(int(value) for value in action_counts),
    )


def aggregate_episodes(episodes: list[EpisodeEvaluation]) -> dict[str, object]:
    """聚合成功率、回报、轮数、修改量和有效动作频率。"""

    if not episodes:
        raise ValueError("至少需要一个回合。")
    action_counts = np.asarray(
        [episode.active_action_counts for episode in episodes],
        dtype=np.int64,
    ).sum(axis=0)
    total_actions = int(action_counts.sum())
    return {
        "episode_count": len(episodes),
        "initial_success_rate": float(np.mean([item.initial_success for item in episodes])),
        "success_rate": float(np.mean([item.success for item in episodes])),
        "timeout_rate": float(np.mean([item.timeout for item in episodes])),
        "optimizer_failure_rate": float(
            np.mean([item.optimizer_failed for item in episodes])
        ),
        "mean_rounds": float(np.mean([item.rounds for item in episodes])),
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
            "round_cost": float(np.mean([item.total_round_cost for item in episodes])),
            "success_bonus": float(
                np.mean([item.total_success_bonus for item in episodes])
            ),
            "timeout_penalty": float(
                np.mean([item.total_timeout_penalty for item in episodes])
            ),
        },
        "active_action_counts": action_counts.tolist(),
        "active_action_proportions": (
            (action_counts / total_actions).tolist()
            if total_actions
            else np.zeros(action_counts.size).tolist()
        ),
    }
