"""双向信任交叉赋权与初始意见融合。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def compute_fusion_weights(
    human_to_ai_trust: NDArray[np.floating],
    ai_to_human_information_trust: NDArray[np.floating],
    epsilon: float = 1.0e-6,
) -> tuple[FloatArray, FloatArray]:
    """计算人类意见权重与 AI 建议权重。

    采用交叉赋权：AI 对人的信息信任决定人类意见权重，
    人对 AI 的信任决定 AI 建议权重。
    """

    human_to_ai = np.asarray(human_to_ai_trust, dtype=np.float64)
    ai_to_human = np.asarray(ai_to_human_information_trust, dtype=np.float64)
    if human_to_ai.shape != ai_to_human.shape or human_to_ai.ndim != 1:
        raise ValueError("双向信任必须是形状一致的一维数组。")
    if np.any((human_to_ai < 0.0) | (human_to_ai > 1.0)):
        raise ValueError("人对 AI 的信任必须位于 [0, 1]。")
    if np.any((ai_to_human < 0.0) | (ai_to_human > 1.0)):
        raise ValueError("AI 对人的信息信任必须位于 [0, 1]。")
    if epsilon <= 0.0:
        raise ValueError("融合稳定项 epsilon 必须为正数。")

    denominator = human_to_ai + ai_to_human + 2.0 * epsilon
    human_weights = (ai_to_human + epsilon) / denominator
    ai_weights = (human_to_ai + epsilon) / denominator
    return human_weights, ai_weights


def fuse_opinions(
    human_opinions: NDArray[np.floating],
    ai_opinions: NDArray[np.floating],
    human_weights: NDArray[np.floating],
    ai_weights: NDArray[np.floating],
) -> FloatArray:
    """按专家—AI 配对权重生成初始综合意见。"""

    human = np.asarray(human_opinions, dtype=np.float64)
    ai = np.asarray(ai_opinions, dtype=np.float64)
    weight_h = np.asarray(human_weights, dtype=np.float64)
    weight_a = np.asarray(ai_weights, dtype=np.float64)

    if human.shape != ai.shape or human.ndim != 2:
        raise ValueError("人类意见与 AI 意见必须是形状一致的二维数组。")
    if weight_h.shape != (human.shape[0],) or weight_a.shape != (human.shape[0],):
        raise ValueError("每一对专家—AI 必须对应一个融合权重。")
    if not np.allclose(weight_h + weight_a, 1.0, atol=1.0e-10):
        raise ValueError("人类权重与 AI 权重之和必须为 1。")

    fused = weight_h[:, None] * human + weight_a[:, None] * ai
    return np.clip(fused, 0.0, 1.0)

