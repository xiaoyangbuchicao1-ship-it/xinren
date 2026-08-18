"""中心反馈智能体使用的轻量级人机共识交互环境。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.data.task_generator import StageBInstance
from src.env.response_model import (
    action_to_multiplier,
    effective_adjustment,
    sample_response_rate,
    sample_response_types,
)
from src.model.consensus import ConsensusMetrics, evaluate_consensus
from src.model.harmony_optimizer import (
    HarmonyOptimizationResult,
    adjustment_distances,
    apply_theoretical_adjustment,
    solve_harmony_adjustment,
)


FloatArray = NDArray[np.float64]


def consensus_reached(
    acd: NDArray[np.floating],
    threshold: float,
    tolerance: float,
) -> bool:
    """使用与理论求解一致的数值容差判定共识。"""

    values = np.asarray(acd, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("ACD 必须是非空一维数组。")
    if tolerance < 0.0:
        raise ValueError("共识判定容差不能为负。")
    violation = max(0.0, float(threshold) - float(values.min()))
    return bool(violation <= tolerance)


def consensus_deficit(
    acd: NDArray[np.floating],
    threshold: float,
) -> float:
    """返回所有未达标专家相对执行阈值的平均连续缺口。"""

    values = np.asarray(acd, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("ACD 必须是非空一维数组。")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("共识缺口阈值必须位于 (0, 1]。")
    return float(np.maximum(0.0, float(threshold) - values).mean())


@dataclass(frozen=True)
class RewardBreakdown:
    """奖励及其可审计分量。"""

    total: float
    consensus_improvement: float
    modification_cost: float
    recommendation_cost: float
    remaining_deficit_cost: float
    unexecuted_recommendation_cost: float
    round_cost: float
    success_bonus: float
    timeout_penalty: float
    mean_modification: float

    def to_serializable(self) -> dict[str, float]:
        return asdict(self)


def build_state(
    human_to_ai_trust: NDArray[np.floating],
    ai_to_human_trust: NDArray[np.floating],
    acd: NDArray[np.floating],
    theoretical_deltas: NDArray[np.floating],
    previous_response: NDArray[np.floating],
    previous_recommendation: NDArray[np.floating],
    round_index: int,
    max_rounds: int,
) -> NDArray[np.float32]:
    """构造每位专家 6 维局部特征加 3 维群体特征的马尔可夫状态。"""

    columns = [
        np.asarray(values, dtype=np.float64)
        for values in (
            human_to_ai_trust,
            ai_to_human_trust,
            acd,
            theoretical_deltas,
            previous_response,
            previous_recommendation,
        )
    ]
    expert_count = columns[0].size
    if expert_count < 1 or any(values.shape != (expert_count,) for values in columns):
        raise ValueError("状态中的 6 组专家特征必须是等长一维数组。")
    if max_rounds <= 0 or not 0 <= round_index <= max_rounds:
        raise ValueError("轮次必须位于 [0, max_rounds]。")
    if np.any((columns[4] < -1.0) | (columns[4] > 1.0)):
        raise ValueError("上一轮响应率必须位于 [-1, 1]。")
    if np.any((columns[5] < -1.0) | (columns[5] > 1.0)):
        raise ValueError("上一轮实际建议量必须位于 [-1, 1]。")

    local = np.column_stack(columns).reshape(-1)
    group = np.asarray(
        [columns[2].mean(), columns[2].std(), round_index / max_rounds],
        dtype=np.float64,
    )
    return np.concatenate([local, group]).astype(np.float32)


def compute_reward(
    previous_opinions: NDArray[np.floating],
    current_opinions: NDArray[np.floating],
    previous_mean_acd: float,
    current_mean_acd: float,
    *,
    success: bool,
    timeout: bool,
    consensus_improvement_weight: float,
    modification_cost_weight: float,
    round_cost: float,
    success_bonus: float,
    timeout_penalty: float,
) -> RewardBreakdown:
    """按论文公式计算单步奖励并保留各分量。"""

    before = np.asarray(previous_opinions, dtype=np.float64)
    after = np.asarray(current_opinions, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("更新前后意见必须是形状一致的二维数组。")
    mean_modification = float(np.mean(np.sum(np.abs(after - before), axis=1)))
    improvement_term = float(
        consensus_improvement_weight * (current_mean_acd - previous_mean_acd)
    )
    modification_term = float(-modification_cost_weight * mean_modification)
    round_term = float(-round_cost)
    success_term = float(success_bonus if success else 0.0)
    timeout_term = float(-timeout_penalty if timeout else 0.0)
    total = improvement_term + modification_term + round_term + success_term + timeout_term
    return RewardBreakdown(
        total=float(total),
        consensus_improvement=improvement_term,
        modification_cost=modification_term,
        recommendation_cost=0.0,
        remaining_deficit_cost=0.0,
        unexecuted_recommendation_cost=0.0,
        round_cost=round_term,
        success_bonus=success_term,
        timeout_penalty=timeout_term,
        mean_modification=mean_modification,
    )


def compute_factorized_reward(
    previous_opinions: NDArray[np.floating],
    current_opinions: NDArray[np.floating],
    previous_acd: NDArray[np.floating],
    current_acd: NDArray[np.floating],
    *,
    success: bool,
    timeout: bool,
    consensus_improvement_weight: float,
    modification_cost_weight: float,
    round_cost: float,
    success_bonus: float,
    timeout_penalty: float,
) -> dict[str, FloatArray]:
    """把群体奖励精确分解到专家，局部奖励均值等于原群体奖励。"""

    before = np.asarray(previous_opinions, dtype=np.float64)
    after = np.asarray(current_opinions, dtype=np.float64)
    previous = np.asarray(previous_acd, dtype=np.float64)
    current = np.asarray(current_acd, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("更新前后意见必须是形状一致的二维数组。")
    expert_count = before.shape[0]
    if previous.shape != (expert_count,) or current.shape != (expert_count,):
        raise ValueError("逐专家ACD必须与意见矩阵的专家维度一致。")

    consensus_terms = consensus_improvement_weight * (current - previous)
    modification_terms = -modification_cost_weight * np.sum(
        np.abs(after - before),
        axis=1,
    )
    shared_term = (
        -float(round_cost)
        + (float(success_bonus) if success else 0.0)
        - (float(timeout_penalty) if timeout else 0.0)
    )
    shared_terms = np.full(expert_count, shared_term, dtype=np.float64)
    totals = consensus_terms + modification_terms + shared_terms
    return {
        "total": totals.astype(np.float64, copy=False),
        "consensus_improvement": consensus_terms.astype(np.float64, copy=False),
        "modification_cost": modification_terms.astype(np.float64, copy=False),
        "shared_event": shared_terms,
    }


def compute_deficit_reward(
    previous_opinions: NDArray[np.floating],
    current_opinions: NDArray[np.floating],
    previous_acd: NDArray[np.floating],
    current_acd: NDArray[np.floating],
    *,
    initial_deficit: float,
    threshold: float,
    deficit_epsilon: float,
    progress_weight: float,
    modification_cost_weight: float,
    round_cost: float,
    success_bonus: float,
    timeout_penalty: float,
    success: bool,
    timeout: bool,
) -> RewardBreakdown:
    """按归一化共识缺口减少计算连续群体奖励。"""

    before = np.asarray(previous_opinions, dtype=np.float64)
    after = np.asarray(current_opinions, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("更新前后意见必须是形状一致的二维数组。")
    if initial_deficit < 0.0 or deficit_epsilon <= 0.0 or progress_weight <= 0.0:
        raise ValueError("初始缺口、缺口稳定项和进展权重必须有效。")
    normalizer = max(float(initial_deficit), float(deficit_epsilon))
    previous_deficit = consensus_deficit(previous_acd, threshold)
    current_deficit = consensus_deficit(current_acd, threshold)
    mean_modification = float(np.mean(np.sum(np.abs(after - before), axis=1)))
    progress_term = float(
        progress_weight * (previous_deficit - current_deficit) / normalizer
    )
    modification_term = float(-modification_cost_weight * mean_modification)
    round_term = float(-round_cost)
    success_term = float(success_bonus if success else 0.0)
    timeout_term = float(-timeout_penalty if timeout else 0.0)
    return RewardBreakdown(
        total=float(
            progress_term
            + modification_term
            + round_term
            + success_term
            + timeout_term
        ),
        consensus_improvement=progress_term,
        modification_cost=modification_term,
        recommendation_cost=0.0,
        remaining_deficit_cost=0.0,
        unexecuted_recommendation_cost=0.0,
        round_cost=round_term,
        success_bonus=success_term,
        timeout_penalty=timeout_term,
        mean_modification=mean_modification,
    )


def compute_factorized_deficit_reward(
    previous_opinions: NDArray[np.floating],
    current_opinions: NDArray[np.floating],
    previous_acd: NDArray[np.floating],
    current_acd: NDArray[np.floating],
    *,
    initial_deficit: float,
    threshold: float,
    deficit_epsilon: float,
    progress_weight: float,
    modification_cost_weight: float,
    round_cost: float,
    success_bonus: float,
    timeout_penalty: float,
    success: bool,
    timeout: bool,
) -> dict[str, FloatArray]:
    """把连续缺口奖励分到专家，并保持局部均值等于群体奖励。"""

    before = np.asarray(previous_opinions, dtype=np.float64)
    after = np.asarray(current_opinions, dtype=np.float64)
    previous = np.asarray(previous_acd, dtype=np.float64)
    current = np.asarray(current_acd, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("更新前后意见必须是形状一致的二维数组。")
    expert_count = before.shape[0]
    if previous.shape != (expert_count,) or current.shape != (expert_count,):
        raise ValueError("逐专家ACD必须与意见矩阵的专家维度一致。")
    if initial_deficit < 0.0 or deficit_epsilon <= 0.0 or progress_weight <= 0.0:
        raise ValueError("初始缺口、缺口稳定项和进展权重必须有效。")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("共识缺口阈值必须位于 (0, 1]。")
    normalizer = max(float(initial_deficit), float(deficit_epsilon))
    previous_deficits = np.maximum(0.0, float(threshold) - previous)
    current_deficits = np.maximum(0.0, float(threshold) - current)
    progress_terms = progress_weight * (
        previous_deficits - current_deficits
    ) / normalizer
    modification_terms = -modification_cost_weight * np.sum(
        np.abs(after - before),
        axis=1,
    )
    shared_term = (
        -float(round_cost)
        + (float(success_bonus) if success else 0.0)
        - (float(timeout_penalty) if timeout else 0.0)
    )
    shared_terms = np.full(expert_count, shared_term, dtype=np.float64)
    return {
        "total": (progress_terms + modification_terms + shared_terms).astype(
            np.float64,
            copy=False,
        ),
        "consensus_improvement": progress_terms.astype(np.float64, copy=False),
        "modification_cost": modification_terms.astype(np.float64, copy=False),
        "shared_event": shared_terms,
    }


class ConsensusFeedbackEnv:
    """不依赖 Gym 的单环境接口，后续可直接由 PPO 采样器调用。"""

    def __init__(
        self,
        config: dict[str, Any],
        rng: np.random.Generator,
        response_types: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config
        self.rng = rng
        self.num_experts = int(config["data"]["num_experts"])
        self.max_rounds = int(config["consensus"]["max_rounds"])
        response_config = config["response"]
        self.response_types = response_types or sample_response_types(
            self.num_experts,
            response_config["type_names"],
            response_config["type_probabilities"],
            rng,
        )
        if len(self.response_types) != self.num_experts:
            raise ValueError("每位专家必须对应一个隐藏响应类型。")
        unknown = set(self.response_types).difference(response_config["response_table"])
        if unknown:
            raise ValueError(f"存在响应表未定义的类型：{sorted(unknown)}")

        self.instance: StageBInstance | None = None
        self.current_opinions: FloatArray | None = None
        self.previous_response = np.full(self.num_experts, -1.0, dtype=np.float64)
        self.previous_recommendation = np.full(
            self.num_experts,
            -1.0,
            dtype=np.float64,
        )
        self.round_index = 0
        self.metrics: ConsensusMetrics | None = None
        self.optimization: HarmonyOptimizationResult | None = None
        self.initial_consensus_deficit: float | None = None
        self.done = False

    @property
    def state_dim(self) -> int:
        return self.num_experts * 6 + 3

    @property
    def action_count(self) -> int:
        return len(self.config["response"]["multipliers"])

    @property
    def guidance_mode(self) -> str:
        """返回建议量语义：静态最优倍率或无优化器的直接建议量。"""

        mode = str(self.config.get("guidance", {}).get("mode", "static_optimizer"))
        if mode not in {"static_optimizer", "direct"}:
            raise ValueError("建议模式只支持static_optimizer或direct。")
        return mode

    @property
    def multiplier_bounds(self) -> tuple[float, float]:
        """返回连续策略与离散策略共同使用的倍率边界。"""

        if self.guidance_mode == "direct":
            bounds = np.asarray(
                self.config["guidance"]["action_bounds"],
                dtype=np.float64,
            )
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or not 0.0 < bounds[0] < bounds[1] <= 1.0
            ):
                raise ValueError("直接建议量边界必须满足0<下界<上界<=1。")
            return float(bounds[0]), float(bounds[1])
        values = np.asarray(self.config["response"]["multipliers"], dtype=np.float64)
        return float(values.min()), float(values.max())

    @property
    def success_threshold(self) -> float:
        """实际回合使用的共识成功门槛。"""

        return float(self.config["consensus"]["threshold"])

    @property
    def planning_threshold(self) -> float:
        """理论优化使用的安全目标门槛。"""

        consensus = self.config["consensus"]
        return self.success_threshold + float(consensus.get("planning_margin", 0.0))

    @property
    def success(self) -> bool:
        """返回考虑数值容差后的当前共识状态。"""

        if self.metrics is None:
            return False
        consensus = self.config["consensus"]
        return consensus_reached(
            self.metrics.acd,
            self.success_threshold,
            float(consensus["constraint_tolerance"]),
        )

    def _solve(self) -> HarmonyOptimizationResult:
        assert self.current_opinions is not None
        consensus = self.config["consensus"]
        return solve_harmony_adjustment(
            self.current_opinions,
            self.planning_threshold,
            max_iterations=int(consensus["solver_max_iterations"]),
            ftol=float(consensus["solver_ftol"]),
            constraint_tolerance=float(consensus["constraint_tolerance"]),
            restarts=int(consensus["solver_restarts"]),
        )

    def _state(self) -> NDArray[np.float32]:
        assert self.instance is not None and self.metrics is not None
        if self.guidance_mode == "static_optimizer":
            assert self.optimization is not None
            guidance_signal = self.optimization.deltas
        else:
            guidance_signal = self._direct_state_signal()
        return build_state(
            self.instance.human_to_ai_trust,
            self.instance.ai_to_human_information_trust,
            self.metrics.acd,
            guidance_signal,
            self.previous_response,
            self.previous_recommendation,
            self.round_index,
            self.max_rounds,
        )

    def _direct_state_signal(self) -> FloatArray:
        """在不增加状态维度时切换直接建议模式的第4个局部特征。"""

        assert self.current_opinions is not None and self.metrics is not None
        distance = adjustment_distances(
            self.current_opinions,
            self.metrics.issue_mask,
            self.metrics.reference,
        )
        effectful = self.metrics.expert_mask & (distance > 1.0e-12)
        deficit = np.where(
            effectful,
            np.maximum(0.0, self.planning_threshold - self.metrics.acd),
            0.0,
        )
        mode = str(
            self.config.get("guidance", {}).get(
                "state_signal",
                "consensus_deficit",
            )
        )
        if mode == "consensus_deficit":
            return deficit
        if mode == "adjustment_distance":
            return np.where(effectful, distance, 0.0)
        if mode == "distance_deficit_sum":
            # 距离与共识缺口都位于意见比例尺度；求和后截断可同时表达
            # “移动代价”和“达标紧迫度”，并保持原33维状态及动作掩码语义。
            return np.clip(distance + deficit, 0.0, 1.0)
        raise ValueError(
            "直接建议状态信号只支持consensus_deficit、"
            "adjustment_distance或distance_deficit_sum。"
        )

    def _direct_action_mask(self) -> NDArray[np.bool_]:
        """排除向群体参考移动也不会改变意见的伪活动专家。"""

        return self._direct_state_signal() > 1.0e-12

    def reset(self, instance: StageBInstance) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """载入一个新决策实例，但保持当前元任务的隐藏响应类型不变。"""

        expected = (self.num_experts, int(self.config["data"]["num_issues"]))
        if instance.initial_fused_opinions.shape != expected:
            raise ValueError(f"初始综合意见形状必须为 {expected}。")
        self.instance = instance
        self.current_opinions = np.array(
            instance.initial_fused_opinions,
            dtype=np.float64,
            copy=True,
        )
        self.previous_response = np.full(self.num_experts, -1.0, dtype=np.float64)
        self.previous_recommendation = np.full(
            self.num_experts,
            -1.0,
            dtype=np.float64,
        )
        self.round_index = 0
        self.metrics = evaluate_consensus(self.current_opinions, self.planning_threshold)
        self.initial_consensus_deficit = consensus_deficit(
            self.metrics.acd,
            self.success_threshold,
        )
        self.optimization = (
            self._solve() if self.guidance_mode == "static_optimizer" else None
        )
        initial_success = self.success
        optimizer_success = bool(
            self.optimization.success if self.optimization is not None else True
        )
        theoretical = (
            self.optimization.deltas.copy()
            if self.optimization is not None
            else np.zeros(self.num_experts, dtype=np.float64)
        )
        guidance_signal = (
            theoretical.copy()
            if self.guidance_mode == "static_optimizer"
            else self._direct_state_signal()
        )
        self.done = bool(initial_success or not optimizer_success)
        info = {
            "initial_success": initial_success,
            "optimizer_success": optimizer_success,
            "min_acd": self.metrics.min_acd,
            "mean_acd": self.metrics.mean_acd,
            "theoretical_deltas": theoretical,
            "guidance_signal": guidance_signal.copy(),
            "guidance_mode": self.guidance_mode,
            "active_expert_mask": (
                self.metrics.expert_mask.copy()
                if self.guidance_mode == "static_optimizer"
                else self._direct_action_mask()
            ),
            "success_threshold": self.success_threshold,
            "planning_threshold": self.planning_threshold,
            "initial_consensus_deficit": self.initial_consensus_deficit,
        }
        return self._state(), info

    def step(
        self,
        actions: NDArray[np.integer] | list[int],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """执行一次离散倍率联合反馈。"""

        if self.guidance_mode == "direct":
            raise ValueError("直接建议模式只支持连续动作。")
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.num_experts,):
            raise ValueError("必须为每位专家提供一个离散动作。")
        multipliers = action_to_multiplier(
            action_array,
            self.config["response"]["multipliers"],
        )
        return self._step_with_multipliers(
            multipliers,
            action_record=action_array,
            action_mode="discrete",
        )

    def step_continuous(
        self,
        multipliers: NDArray[np.floating] | list[float],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """执行一次连续倍率联合反馈，边界与原离散动作范围一致。"""

        values = np.asarray(multipliers, dtype=np.float64)
        if values.shape != (self.num_experts,):
            raise ValueError("必须为每位专家提供一个连续倍率。")
        low, high = self.multiplier_bounds
        if not np.isfinite(values).all() or np.any(values < low) or np.any(values > high):
            raise ValueError(f"连续倍率必须是位于 [{low}, {high}] 的有限值。")
        return self._step_with_multipliers(
            values,
            action_record=values,
            action_mode="continuous",
        )

    def _step_with_multipliers(
        self,
        multipliers: FloatArray,
        *,
        action_record: NDArray[np.integer] | NDArray[np.floating],
        action_mode: str,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """共享连续与离散动作之后的环境状态转移。"""

        if self.instance is None or self.current_opinions is None:
            raise RuntimeError("必须先调用 reset()。")
        if self.done:
            raise RuntimeError("当前回合已经终止，必须重新 reset()。")
        assert self.metrics is not None

        multiplier_array = np.asarray(multipliers, dtype=np.float64)
        if multiplier_array.shape != (self.num_experts,):
            raise ValueError("倍率向量必须与专家数量一致。")
        if self.guidance_mode == "static_optimizer":
            assert self.optimization is not None
            theoretical = self.optimization.deltas.copy()
            recommended = np.clip(multiplier_array * theoretical, 0.0, 1.0)
            active = recommended > 1.0e-12
        else:
            theoretical = np.zeros(self.num_experts, dtype=np.float64)
            active = self._direct_action_mask()
            recommended = np.zeros(self.num_experts, dtype=np.float64)
            recommended[active] = np.clip(multiplier_array[active], 0.0, 1.0)

        response_rates = np.full(self.num_experts, -1.0, dtype=np.float64)
        response_config = self.config["response"]
        for expert in np.flatnonzero(active):
            response_rates[expert] = sample_response_rate(
                self.response_types[expert],
                float(recommended[expert]),
                response_config["response_table"],
                float(response_config["response_noise_std"]),
                self.rng,
                response_config["suggestion_bins"],
                interpolation=str(response_config.get("interpolation", "step")),
            )

        effective = np.zeros(self.num_experts, dtype=np.float64)
        if np.any(active):
            effective[active] = effective_adjustment(
                recommended[active],
                response_rates[active],
            )

        previous_opinions = self.current_opinions.copy()
        previous_metrics = self.metrics
        self.current_opinions = apply_theoretical_adjustment(
            self.current_opinions,
            effective,
            previous_metrics.issue_mask,
            previous_metrics.reference,
        )
        self.previous_response = response_rates
        self.previous_recommendation = np.full(
            self.num_experts,
            -1.0,
            dtype=np.float64,
        )
        self.previous_recommendation[active] = recommended[active]
        self.round_index += 1
        self.metrics = evaluate_consensus(self.current_opinions, self.planning_threshold)

        success = self.success
        timeout = bool(self.round_index >= self.max_rounds and not success)
        if self.guidance_mode == "direct":
            self.optimization = None
            optimizer_failed = False
        else:
            if success:
                self.optimization = self._solve()
                optimizer_failed = False
            elif timeout:
                # 超时时不再做一次没有决策用途的优化，只保留当前理论量。
                optimizer_failed = False
            else:
                self.optimization = self._solve()
                optimizer_failed = not self.optimization.success

        reward_config = self.config["reward"]
        reward_mode = str(reward_config.get("mode", "legacy"))
        if reward_mode == "deficit":
            assert self.initial_consensus_deficit is not None
            deficit_parameters = {
                "initial_deficit": self.initial_consensus_deficit,
                "threshold": self.success_threshold,
                "deficit_epsilon": float(reward_config["deficit_epsilon"]),
                "progress_weight": float(reward_config["deficit_progress_weight"]),
                "modification_cost_weight": float(
                    reward_config["modification_cost_weight"]
                ),
                "round_cost": float(reward_config["round_cost"]),
                "success_bonus": float(reward_config["success_bonus"]),
                "timeout_penalty": float(reward_config["timeout_penalty"]),
                "success": success,
                "timeout": timeout,
            }
            reward = compute_deficit_reward(
                previous_opinions,
                self.current_opinions,
                previous_metrics.acd,
                self.metrics.acd,
                **deficit_parameters,
            )
            factorized_reward = compute_factorized_deficit_reward(
                previous_opinions,
                self.current_opinions,
                previous_metrics.acd,
                self.metrics.acd,
                **deficit_parameters,
            )
        elif reward_mode == "legacy":
            reward = compute_reward(
                previous_opinions,
                self.current_opinions,
                previous_metrics.mean_acd,
                self.metrics.mean_acd,
                success=success,
                timeout=timeout,
                consensus_improvement_weight=float(
                    reward_config["consensus_improvement_weight"]
                ),
                modification_cost_weight=float(
                    reward_config["modification_cost_weight"]
                ),
                round_cost=float(reward_config["round_cost"]),
                success_bonus=float(reward_config["success_bonus"]),
                timeout_penalty=float(reward_config["timeout_penalty"]),
            )
            factorized_reward = compute_factorized_reward(
                previous_opinions,
                self.current_opinions,
                previous_metrics.acd,
                self.metrics.acd,
                success=success,
                timeout=timeout,
                consensus_improvement_weight=float(
                    reward_config["consensus_improvement_weight"]
                ),
                modification_cost_weight=float(
                    reward_config["modification_cost_weight"]
                ),
                round_cost=float(reward_config["round_cost"]),
                success_bonus=float(reward_config["success_bonus"]),
                timeout_penalty=float(reward_config["timeout_penalty"]),
            )
        else:
            raise ValueError("奖励模式只支持legacy或deficit。")
        recommendation_cost_weight = float(reward_config.get("recommendation_cost_weight", 0.0))
        remaining_deficit_cost_weight = float(
            reward_config.get("remaining_deficit_cost_weight", 0.0)
        )
        unexecuted_cost_weight = float(
            reward_config.get("unexecuted_recommendation_cost_weight", 0.0)
        )
        if min(
            recommendation_cost_weight,
            remaining_deficit_cost_weight,
            unexecuted_cost_weight,
        ) < 0.0:
            raise ValueError("建议、剩余缺口和未执行建议成本权重不能为负。")
        recommendation_terms = -recommendation_cost_weight * np.square(recommended)
        recommendation_term = float(recommendation_terms.mean())
        if reward_mode == "deficit":
            assert self.initial_consensus_deficit is not None
            normalizer = max(
                float(self.initial_consensus_deficit),
                float(reward_config["deficit_epsilon"]),
            )
            remaining_deficit_terms = -remaining_deficit_cost_weight * (
                np.maximum(0.0, self.success_threshold - self.metrics.acd) / normalizer
            )
        else:
            remaining_deficit_terms = np.zeros(self.num_experts, dtype=np.float64)
        remaining_deficit_term = float(remaining_deficit_terms.mean())
        unexecuted_terms = np.zeros(self.num_experts, dtype=np.float64)
        if np.any(active):
            unexecuted_terms[active] = -unexecuted_cost_weight * recommended[active] * (
                1.0 - np.clip(response_rates[active], 0.0, 1.0)
            )
        unexecuted_term = float(unexecuted_terms.mean())
        reward = RewardBreakdown(
            total=(
                reward.total
                + recommendation_term
                + remaining_deficit_term
                + unexecuted_term
            ),
            consensus_improvement=reward.consensus_improvement,
            modification_cost=reward.modification_cost,
            recommendation_cost=recommendation_term,
            remaining_deficit_cost=remaining_deficit_term,
            unexecuted_recommendation_cost=unexecuted_term,
            round_cost=reward.round_cost,
            success_bonus=reward.success_bonus,
            timeout_penalty=reward.timeout_penalty,
            mean_modification=reward.mean_modification,
        )
        factorized_reward["recommendation_cost"] = recommendation_terms
        factorized_reward["remaining_deficit_cost"] = remaining_deficit_terms
        factorized_reward["unexecuted_recommendation_cost"] = unexecuted_terms
        factorized_reward["total"] = (
            factorized_reward["total"]
            + recommendation_terms
            + remaining_deficit_terms
            + unexecuted_terms
        )
        if not np.isclose(
            np.mean(factorized_reward["total"]),
            reward.total,
            atol=1.0e-10,
        ):
            raise RuntimeError("逐专家奖励分解与群体奖励不守恒。")
        self.done = bool(success or timeout or optimizer_failed)
        info = {
            "round": self.round_index,
            "success": success,
            "timeout": timeout,
            "optimizer_failed": optimizer_failed,
            "action_mode": action_mode,
            "actions": np.asarray(action_record).copy(),
            "multipliers": multiplier_array.copy(),
            "theoretical_deltas": theoretical,
            "guidance_mode": self.guidance_mode,
            "guidance_signal": (
                theoretical.copy()
                if self.guidance_mode == "static_optimizer"
                else 1.0 - previous_metrics.acd
            ),
            "active_expert_mask": active.copy(),
            "recommended_deltas": recommended,
            "response_rates": response_rates.copy(),
            "effective_deltas": effective,
            "acd": self.metrics.acd.copy(),
            "mean_acd": self.metrics.mean_acd,
            "min_acd": self.metrics.min_acd,
            "reward": reward.to_serializable(),
            "reward_mode": reward_mode,
            "factorized_reward": {
                key: value.copy() for key, value in factorized_reward.items()
            },
        }
        terminated = bool(success or optimizer_failed)
        truncated = timeout
        return self._state(), reward.total, terminated, truncated, info
