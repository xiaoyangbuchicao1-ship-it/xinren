"""双向信任的采样与计算。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _as_float_array(values: NDArray[np.floating] | list[float]) -> FloatArray:
    """转换为 float64 数组，统一数学模块的数值精度。"""

    return np.asarray(values, dtype=np.float64)


def compute_human_to_ai_trust(
    trust_degrees: NDArray[np.floating] | list[float],
    distrust_degrees: NDArray[np.floating] | list[float],
) -> FloatArray:
    """计算人对 AI 的综合信任 T^H=(t-d+1)/2。"""

    trust = _as_float_array(trust_degrees)
    distrust = _as_float_array(distrust_degrees)
    if trust.shape != distrust.shape:
        raise ValueError("信任与不信任数组形状必须一致。")
    if np.any((trust < 0.0) | (trust > 1.0)):
        raise ValueError("信任值必须位于 [0, 1]。")
    if np.any((distrust < 0.0) | (distrust > 1.0)):
        raise ValueError("不信任值必须位于 [0, 1]。")
    return np.clip((trust - distrust + 1.0) / 2.0, 0.0, 1.0)


def sample_human_to_ai_trust(
    num_experts: int,
    beta_parameters: tuple[float, float] | list[float],
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """按 Beta 分布生成完备信任表达 d=1-t。"""

    alpha, beta = (float(value) for value in beta_parameters)
    if num_experts <= 0:
        raise ValueError("专家数量必须为正整数。")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta 分布参数必须为正数。")

    trust = rng.beta(alpha, beta, size=num_experts).astype(np.float64)
    distrust = 1.0 - trust
    combined = compute_human_to_ai_trust(trust, distrust)
    return trust, distrust, combined


def compute_ai_to_human_information_trust(
    human_opinions: NDArray[np.floating],
    ai_opinions: NDArray[np.floating],
    variance_epsilon: float = 1.0e-12,
) -> FloatArray:
    """以逐对意见皮尔逊相关的非负部分表示 AI 对人的信息信任。

    当任一意见向量近似常数时，相关系数没有稳定含义，按论文约定返回 0。
    """

    human = _as_float_array(human_opinions)
    ai = _as_float_array(ai_opinions)
    if human.shape != ai.shape or human.ndim != 2:
        raise ValueError("人类意见与 AI 意见必须是形状一致的二维数组。")
    if variance_epsilon < 0.0:
        raise ValueError("方差边界不能为负。")

    information_trust = np.zeros(human.shape[0], dtype=np.float64)
    for index, (human_vector, ai_vector) in enumerate(zip(human, ai, strict=True)):
        human_centered = human_vector - human_vector.mean()
        ai_centered = ai_vector - ai_vector.mean()
        human_variance = float(np.mean(human_centered**2))
        ai_variance = float(np.mean(ai_centered**2))
        if human_variance <= variance_epsilon or ai_variance <= variance_epsilon:
            information_trust[index] = 0.0
            continue

        denominator = float(np.linalg.norm(human_centered) * np.linalg.norm(ai_centered))
        correlation = float(np.dot(human_centered, ai_centered) / denominator)
        information_trust[index] = max(0.0, float(np.clip(correlation, -1.0, 1.0)))

    return information_trust

