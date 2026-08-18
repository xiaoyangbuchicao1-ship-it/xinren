"""中心化 PPO 使用的因子化 Actor 与状态价值网络。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical, Normal


def _activation(name: str) -> type[nn.Module]:
    normalized = name.lower()
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "relu":
        return nn.ReLU
    raise ValueError(f"不支持的激活函数：{name}")


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: str,
    output_gain: float,
) -> nn.Sequential:
    """构造正交初始化的轻量级 MLP。"""

    if input_dim <= 0 or output_dim <= 0 or any(int(size) <= 0 for size in hidden_sizes):
        raise ValueError("网络各层维度必须为正整数。")
    activation_type = _activation(activation)
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        linear = nn.Linear(previous, int(size))
        nn.init.orthogonal_(linear.weight, gain=2.0**0.5)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, activation_type()])
        previous = int(size)
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class FactorizedActor(nn.Module):
    """用共享候选评分器输出逐专家的离散倍率分布。"""

    def __init__(
        self,
        state_dim: int,
        num_experts: int,
        action_count: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.25),
        suggestion_bins: Sequence[float] = (0.0, 0.3, 0.7, 1.0),
    ) -> None:
        super().__init__()
        if num_experts <= 0 or action_count <= 1:
            raise ValueError("专家数必须为正，动作数必须大于 1。")
        self.state_dim = int(state_dim)
        self.num_experts = int(num_experts)
        self.action_count = int(action_count)
        self.local_feature_count = 6
        self.group_feature_count = 3
        multiplier_values = tuple(float(value) for value in multipliers)
        bin_values = tuple(float(value) for value in suggestion_bins)
        if len(multiplier_values) != self.action_count:
            raise ValueError("候选倍率数量必须与动作数一致。")
        if any(value <= 0.0 for value in multiplier_values):
            raise ValueError("候选倍率必须为正数。")
        if (
            len(bin_values) != 4
            or bin_values[0] != 0.0
            or bin_values[-1] != 1.0
            or any(left >= right for left, right in zip(bin_values, bin_values[1:]))
        ):
            raise ValueError("建议区间必须是从 0 到 1 严格递增的四个边界。")
        self.register_buffer(
            "multipliers",
            torch.tensor(multiplier_values, dtype=torch.float32),
        )
        self.register_buffer(
            "suggestion_boundaries",
            torch.tensor(bin_values[1:-1], dtype=torch.float32),
        )
        expected_state_dim = (
            self.num_experts * self.local_feature_count + self.group_feature_count
        )
        if self.state_dim != expected_state_dim:
            raise ValueError(
                f"逐专家 Actor 要求状态维度为 {expected_state_dim}，"
                f"当前为 {self.state_dim}。"
            )
        self.network = _mlp(
            self.local_feature_count + self.group_feature_count + 5,
            hidden_sizes,
            1,
            activation,
            output_gain=0.01,
        )
        # 独立先验偏置保证初始化概率精确可控，不依赖候选特征。
        self.action_bias = nn.Parameter(torch.zeros(self.action_count))

    def initialize_action_prior(
        self,
        preferred_action_index: int,
        preferred_probability: float,
    ) -> None:
        """设置可解释的初始动作先验，同时保留其他倍率的探索概率。"""

        if not 0 <= preferred_action_index < self.action_count:
            raise ValueError("先验动作索引超出动作范围。")
        uniform_probability = 1.0 / self.action_count
        if not uniform_probability < preferred_probability < 1.0:
            raise ValueError("先验动作概率必须大于均匀概率且小于 1。")
        other_probability = (1.0 - preferred_probability) / (self.action_count - 1)
        output_layer = self.network[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("Actor 输出层必须是线性层。")
        with torch.no_grad():
            # 清零候选评分头，确保任意输入状态都从指定先验开始。
            output_layer.weight.zero_()
            output_layer.bias.zero_()
            self.action_bias.fill_(math.log(other_probability))
            self.action_bias[preferred_action_index] = math.log(preferred_probability)

    def candidate_features(self, states: torch.Tensor) -> torch.Tensor:
        """构造每位专家、每个候选倍率对应的 14 维可观测特征。"""

        if states.shape[-1] != self.state_dim:
            raise ValueError(f"Actor 状态末维必须为 {self.state_dim}。")
        local_end = self.num_experts * self.local_feature_count
        local = states[..., :local_end].reshape(
            *states.shape[:-1],
            self.num_experts,
            self.local_feature_count,
        )
        group = states[..., local_end:].unsqueeze(-2).expand(
            *states.shape[:-1],
            self.num_experts,
            self.group_feature_count,
        )
        base_features = torch.cat([local, group], dim=-1)
        base_features = base_features.unsqueeze(-2).expand(
            *base_features.shape[:-1],
            self.action_count,
            base_features.shape[-1],
        )

        # 第 4 个局部特征是当前理论调整量；倍率作用后得到实际候选建议量。
        multiplier_shape = [1] * (local.ndim - 2) + [1, self.action_count]
        multipliers = self.multipliers.view(*multiplier_shape)
        theoretical_delta = local[..., 3].unsqueeze(-1)
        recommendations = torch.clamp(theoretical_delta * multipliers, 0.0, 1.0)
        multipliers = multipliers.expand_as(recommendations)

        bin_indices = (
            (recommendations >= self.suggestion_boundaries[0]).long()
            + (recommendations >= self.suggestion_boundaries[1]).long()
        )
        bin_one_hot = nn.functional.one_hot(bin_indices, num_classes=3).to(states.dtype)
        return torch.cat(
            [
                base_features,
                multipliers.unsqueeze(-1),
                recommendations.unsqueeze(-1),
                bin_one_hot,
            ],
            dim=-1,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        candidate_scores = self.network(self.candidate_features(states)).squeeze(-1)
        return candidate_scores + self.action_bias

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self(states))

    def act(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回联合动作、联合对数概率和按专家平均的策略熵。"""

        distribution = self.distribution(states)
        actions = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        joint_log_probability = distribution.log_prob(actions).sum(dim=-1)
        mean_entropy = distribution.entropy().mean(dim=-1)
        return actions, joint_log_probability, mean_entropy

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        return (
            distribution.log_prob(actions).sum(dim=-1),
            distribution.entropy().mean(dim=-1),
        )


class ContinuousFactorizedActor(nn.Module):
    """用共享高斯策略为每位专家输出有界连续倍率。"""

    def __init__(
        self,
        state_dim: int,
        num_experts: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        multiplier_low: float = 0.5,
        multiplier_high: float = 1.25,
        include_expert_identity: bool = False,
    ) -> None:
        super().__init__()
        if num_experts <= 0 or not 0.0 < multiplier_low < multiplier_high:
            raise ValueError("专家数量和连续倍率边界必须有效。")
        self.state_dim = int(state_dim)
        self.num_experts = int(num_experts)
        self.local_feature_count = 6
        self.group_feature_count = 3
        self.include_expert_identity = bool(include_expert_identity)
        expected_state_dim = (
            self.num_experts * self.local_feature_count + self.group_feature_count
        )
        if self.state_dim != expected_state_dim:
            raise ValueError(
                f"连续逐专家 Actor 要求状态维度为 {expected_state_dim}，"
                f"当前为 {self.state_dim}。"
            )
        self.register_buffer(
            "multiplier_low",
            torch.tensor(float(multiplier_low), dtype=torch.float32),
        )
        self.register_buffer(
            "multiplier_high",
            torch.tensor(float(multiplier_high), dtype=torch.float32),
        )
        self.network = _mlp(
            self.local_feature_count
            + self.group_feature_count
            + (self.num_experts if self.include_expert_identity else 0),
            hidden_sizes,
            2,
            activation,
            output_gain=0.01,
        )
        if self.include_expert_identity:
            # 五个快参数只校准各专家的原始动作均值，不扩充环境状态。
            self.expert_mean_offsets = nn.Parameter(torch.zeros(self.num_experts))
        else:
            self.register_parameter("expert_mean_offsets", None)

    def initialize_multiplier_prior(
        self,
        preferred_multiplier: float = 1.0,
        initial_log_std: float = -1.0,
    ) -> None:
        """让初始确定性动作精确落在可解释的静态倍率先验上。"""

        low = float(self.multiplier_low.item())
        high = float(self.multiplier_high.item())
        if not low < preferred_multiplier < high:
            raise ValueError("先验倍率必须严格位于连续动作区间内部。")
        if not -5.0 <= initial_log_std <= 1.0:
            raise ValueError("初始对数标准差必须位于 [-5, 1]。")
        unit = 2.0 * (float(preferred_multiplier) - low) / (high - low) - 1.0
        output = self.network[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("连续 Actor 输出层必须是线性层。")
        with torch.no_grad():
            output.weight.zero_()
            output.bias[0] = math.atanh(unit)
            output.bias[1] = float(initial_log_std)
            if self.expert_mean_offsets is not None:
                self.expert_mean_offsets.zero_()

    def initialize_multiplier_residual_prior(
        self,
        preferred_multiplier: float = 1.0,
        initial_log_std: float = -1.0,
        mean_head_gain: float = 0.15,
    ) -> None:
        """围绕静态倍率保留一个小型随机残差头，而不是写死确定性动作。"""

        low = float(self.multiplier_low.item())
        high = float(self.multiplier_high.item())
        if not low < preferred_multiplier < high:
            raise ValueError("残差先验倍率必须严格位于连续动作区间内部。")
        if not -5.0 <= initial_log_std <= 1.0:
            raise ValueError("初始对数标准差必须位于 [-5, 1]。")
        if not 0.0 < mean_head_gain <= 1.0:
            raise ValueError("残差均值头增益必须位于 (0, 1]。")

        unit = 2.0 * (float(preferred_multiplier) - low) / (high - low) - 1.0
        output = self.network[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("连续 Actor 输出层必须是线性层。")
        with torch.no_grad():
            mean_weight = output.weight[0]
            current_norm = torch.linalg.vector_norm(mean_weight)
            if float(current_norm.item()) <= 1.0e-12:
                nn.init.normal_(mean_weight, mean=0.0, std=1.0)
                current_norm = torch.linalg.vector_norm(mean_weight)
            mean_weight.mul_(float(mean_head_gain) / current_norm)
            # 探索方差保持常数，避免状态相关方差掩盖均值残差的学习过程。
            output.weight[1].zero_()
            output.bias[0] = math.atanh(unit)
            output.bias[1] = float(initial_log_std)
            if self.expert_mean_offsets is not None:
                self.expert_mean_offsets.zero_()

    def fast_adaptation_parameters(self) -> tuple[nn.Parameter, ...]:
        """返回团队内循环专用的五个专家均值偏置。"""

        if self.expert_mean_offsets is None:
            raise ValueError("只有启用专家编号的Actor才具有个体快参数。")
        return (self.expert_mean_offsets,)

    def expert_features(self, states: torch.Tensor) -> torch.Tensor:
        """把33维状态拆成逐专家特征，并按需附加已知专家编号。"""

        if states.shape[-1] != self.state_dim:
            raise ValueError(f"连续 Actor 状态末维必须为 {self.state_dim}。")
        local_end = self.num_experts * self.local_feature_count
        local = states[..., :local_end].reshape(
            *states.shape[:-1],
            self.num_experts,
            self.local_feature_count,
        )
        group = states[..., local_end:].unsqueeze(-2).expand(
            *states.shape[:-1],
            self.num_experts,
            self.group_feature_count,
        )
        features = [local, group]
        if self.include_expert_identity:
            # 编号独热向量不包含任何隐私或行为标签，只标识同一团队中的专家位置。
            leading = states.shape[:-1]
            identity = torch.eye(
                self.num_experts,
                dtype=states.dtype,
                device=states.device,
            ).reshape(*([1] * len(leading)), self.num_experts, self.num_experts)
            identity = identity.expand(*leading, self.num_experts, self.num_experts)
            features.append(identity)
        return torch.cat(features, dim=-1)

    def distribution(
        self,
        states: torch.Tensor,
        expert_mean_offsets: torch.Tensor | None = None,
    ) -> Normal:
        parameters = self.network(self.expert_features(states))
        mean = parameters[..., 0]
        offsets = (
            self.expert_mean_offsets
            if expert_mean_offsets is None
            else expert_mean_offsets
        )
        if offsets is not None:
            if offsets.shape != (self.num_experts,):
                raise ValueError("专家均值快参数必须与专家数量一致。")
            mean = mean + offsets
        log_std = torch.clamp(parameters[..., 1], min=-5.0, max=1.0)
        return Normal(mean, torch.exp(log_std))

    def active_expert_mask(self, states: torch.Tensor) -> torch.Tensor:
        """由状态中的理论调整量识别本轮真正影响环境的专家动作。"""

        if states.shape[-1] != self.state_dim:
            raise ValueError(f"连续 Actor 状态末维必须为 {self.state_dim}。")
        local = states[..., : self.num_experts * self.local_feature_count].reshape(
            *states.shape[:-1],
            self.num_experts,
            self.local_feature_count,
        )
        return local[..., 3] > 1.0e-12

    def _squash(self, raw_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unit_actions = torch.tanh(raw_actions)
        scale = (self.multiplier_high - self.multiplier_low) / 2.0
        multipliers = self.multiplier_low + scale * (unit_actions + 1.0)
        return multipliers, unit_actions

    def _inverse(self, multipliers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = (self.multiplier_high - self.multiplier_low) / 2.0
        unit_actions = (multipliers - self.multiplier_low) / scale - 1.0
        unit_actions = torch.clamp(unit_actions, min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
        raw_actions = 0.5 * (
            torch.log1p(unit_actions) - torch.log1p(-unit_actions)
        )
        return raw_actions, unit_actions

    def _per_expert_log_probability(
        self,
        distribution: Normal,
        raw_actions: torch.Tensor,
        unit_actions: torch.Tensor,
    ) -> torch.Tensor:
        scale = (self.multiplier_high - self.multiplier_low) / 2.0
        log_jacobian = torch.log(scale) + torch.log(
            torch.clamp(1.0 - unit_actions.square(), min=1.0e-6)
        )
        return distribution.log_prob(raw_actions) - log_jacobian

    def act(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_actions = distribution.mean if deterministic else distribution.sample()
        multipliers, unit_actions = self._squash(raw_actions)
        per_expert_log_probability = self._per_expert_log_probability(
            distribution,
            raw_actions,
            unit_actions,
        )
        active = self.active_expert_mask(states)
        active_float = active.to(per_expert_log_probability.dtype)
        active_count = torch.clamp(active_float.sum(dim=-1), min=1.0)
        return (
            multipliers,
            (per_expert_log_probability * active_float).sum(dim=-1),
            -(per_expert_log_probability * active_float).sum(dim=-1) / active_count,
        )

    def evaluate_actions_per_expert(
        self,
        states: torch.Tensor,
        multipliers: torch.Tensor,
        expert_mean_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回未聚合的逐专家对数概率与采样熵估计。"""

        distribution = self.distribution(states, expert_mean_offsets)
        raw_actions, unit_actions = self._inverse(multipliers)
        per_expert_log_probability = self._per_expert_log_probability(
            distribution,
            raw_actions,
            unit_actions,
        )
        return per_expert_log_probability, -per_expert_log_probability

    def evaluate_actions(
        self,
        states: torch.Tensor,
        multipliers: torch.Tensor,
        expert_mean_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_expert_log_probability, per_expert_entropy = (
            self.evaluate_actions_per_expert(
                states,
                multipliers,
                expert_mean_offsets,
            )
        )
        active_float = self.active_expert_mask(states).to(
            per_expert_log_probability.dtype
        )
        active_count = torch.clamp(active_float.sum(dim=-1), min=1.0)
        return (
            (per_expert_log_probability * active_float).sum(dim=-1),
            (per_expert_entropy * active_float).sum(dim=-1) / active_count,
        )


class ValueNetwork(nn.Module):
    """输出标量状态价值。"""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.network = _mlp(
            state_dim,
            hidden_sizes,
            1,
            activation,
            output_gain=1.0,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
