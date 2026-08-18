"""阶段 B 的仿真决策实例生成器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.model.fusion import compute_fusion_weights, fuse_opinions
from src.model.trust import (
    compute_ai_to_human_information_trust,
    sample_human_to_ai_trust,
)


FloatArray = NDArray[np.float64]


def _readonly(values: NDArray[np.floating]) -> FloatArray:
    """返回不可写副本，保护单次任务内应保持固定的变量。"""

    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class OpinionTaskData:
    """人类与 AI 原始意见及其生成诊断。"""

    reference: FloatArray
    human_opinions: FloatArray
    ai_opinions: FloatArray
    human_biases: FloatArray
    human_clipped_fraction: float
    ai_clipped_fraction: float


@dataclass(frozen=True)
class StageBInstance:
    """阶段 B 完整输出，所有数组在返回后均不可原地修改。"""

    task: OpinionTaskData
    trust_degrees: FloatArray
    distrust_degrees: FloatArray
    human_to_ai_trust: FloatArray
    ai_to_human_information_trust: FloatArray
    human_weights: FloatArray
    ai_weights: FloatArray
    initial_fused_opinions: FloatArray

    def to_serializable(self) -> dict[str, Any]:
        """转换为可由 JSON 直接保存的结构。"""

        return {
            "reference": self.task.reference.tolist(),
            "human_opinions": self.task.human_opinions.tolist(),
            "ai_opinions": self.task.ai_opinions.tolist(),
            "human_biases": self.task.human_biases.tolist(),
            "human_clipped_fraction": self.task.human_clipped_fraction,
            "ai_clipped_fraction": self.task.ai_clipped_fraction,
            "trust_degrees": self.trust_degrees.tolist(),
            "distrust_degrees": self.distrust_degrees.tolist(),
            "human_to_ai_trust": self.human_to_ai_trust.tolist(),
            "ai_to_human_information_trust": self.ai_to_human_information_trust.tolist(),
            "human_weights": self.human_weights.tolist(),
            "ai_weights": self.ai_weights.tolist(),
            "initial_fused_opinions": self.initial_fused_opinions.tolist(),
        }


def generate_reference_vector(
    num_issues: int,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> FloatArray:
    """生成只供仿真和离线评价使用的隐藏参考向量。"""

    if num_issues <= 0:
        raise ValueError("议题数量必须为正整数。")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("隐藏参考向量范围必须满足 0 <= low < high <= 1。")
    return rng.uniform(low, high, size=num_issues).astype(np.float64)


def generate_human_opinions(
    reference: NDArray[np.floating],
    num_experts: int,
    bias_max: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, float]:
    """依据专家级偏移和议题级噪声生成人类原始意见。"""

    reference_array = np.asarray(reference, dtype=np.float64)
    if reference_array.ndim != 1 or num_experts <= 0:
        raise ValueError("参考向量必须为一维，专家数量必须为正。")
    if bias_max < 0.0 or noise_std < 0.0:
        raise ValueError("偏移范围和噪声标准差不能为负。")

    biases = rng.uniform(-bias_max, bias_max, size=num_experts).astype(np.float64)
    noise = rng.normal(0.0, noise_std, size=(num_experts, reference_array.size))
    raw = reference_array[None, :] + biases[:, None] + noise
    clipped_fraction = float(np.mean((raw < 0.0) | (raw > 1.0)))
    return np.clip(raw, 0.0, 1.0), biases, clipped_fraction


def generate_ai_recommendations(
    reference: NDArray[np.floating],
    num_ais: int,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, float]:
    """使用独立噪声生成 5 个通常不同的 AI 建议向量。"""

    reference_array = np.asarray(reference, dtype=np.float64)
    if reference_array.ndim != 1 or num_ais <= 0:
        raise ValueError("参考向量必须为一维，AI 数量必须为正。")
    if noise_std < 0.0:
        raise ValueError("AI 建议噪声标准差不能为负。")

    noise = rng.normal(0.0, noise_std, size=(num_ais, reference_array.size))
    raw = reference_array[None, :] + noise
    clipped_fraction = float(np.mean((raw < 0.0) | (raw > 1.0)))
    return np.clip(raw, 0.0, 1.0), clipped_fraction


def generate_stage_b_instance(config: dict[str, Any], rng: np.random.Generator) -> StageBInstance:
    """按配置生成阶段 B 的完整、不可变初始实例。"""

    data = config["data"]
    num_experts = int(data["num_experts"])
    num_ais = int(data["num_ais"])
    if num_experts != num_ais:
        raise ValueError("当前阶段要求专家与 AI 一一配对。")

    reference = generate_reference_vector(
        int(data["num_issues"]),
        float(data["reference_low"]),
        float(data["reference_high"]),
        rng,
    )
    human, biases, human_clipped = generate_human_opinions(
        reference,
        num_experts,
        float(data["human_bias_max"]),
        float(data["human_noise_std"]),
        rng,
    )
    ai, ai_clipped = generate_ai_recommendations(
        reference,
        num_ais,
        float(data["ai_noise_std"]),
        rng,
    )

    trust, distrust, human_to_ai = sample_human_to_ai_trust(
        num_experts,
        data["human_trust_beta"],
        rng,
    )
    ai_to_human = compute_ai_to_human_information_trust(
        human,
        ai,
        variance_epsilon=float(data["correlation_variance_epsilon"]),
    )
    human_weights, ai_weights = compute_fusion_weights(
        human_to_ai,
        ai_to_human,
        epsilon=float(data["fusion_epsilon"]),
    )
    fused = fuse_opinions(human, ai, human_weights, ai_weights)

    task = OpinionTaskData(
        reference=_readonly(reference),
        human_opinions=_readonly(human),
        ai_opinions=_readonly(ai),
        human_biases=_readonly(biases),
        human_clipped_fraction=human_clipped,
        ai_clipped_fraction=ai_clipped,
    )
    return StageBInstance(
        task=task,
        trust_degrees=_readonly(trust),
        distrust_degrees=_readonly(distrust),
        human_to_ai_trust=_readonly(human_to_ai),
        ai_to_human_information_trust=_readonly(ai_to_human),
        human_weights=_readonly(human_weights),
        ai_weights=_readonly(ai_weights),
        initial_fused_opinions=_readonly(fused),
    )

