"""群体综合意见的共识测量。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _validate_opinions(opinions: NDArray[np.floating]) -> FloatArray:
    """验证并统一综合意见数组。"""

    values = np.asarray(opinions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("综合意见必须是至少包含 2 位专家和 1 个议题的二维数组。")
    if not np.all(np.isfinite(values)):
        raise ValueError("综合意见不能包含 NaN 或无穷值。")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("综合意见必须位于 [0, 1]。")
    return values


def pairwise_similarity(opinions: NDArray[np.floating]) -> FloatArray:
    """计算 S[k,s,j]=1-|p[k,j]-p[s,j]|。"""

    values = _validate_opinions(opinions)
    differences = np.abs(values[:, None, :] - values[None, :, :])
    return np.clip(1.0 - differences, 0.0, 1.0)


def element_consensus(
    opinions: NDArray[np.floating],
    similarity: NDArray[np.floating] | None = None,
) -> FloatArray:
    """计算每位专家在每个议题上的元素共识度 ACE。"""

    values = _validate_opinions(opinions)
    similarities = (
        pairwise_similarity(values)
        if similarity is None
        else np.asarray(similarity, dtype=np.float64)
    )
    expected_shape = (values.shape[0], values.shape[0], values.shape[1])
    if similarities.shape != expected_shape:
        raise ValueError(f"相似度数组形状必须为 {expected_shape}。")

    # 自相似度恒为 1，ACE 只平均其余 m-1 位专家。
    return (similarities.sum(axis=1) - 1.0) / (values.shape[0] - 1)


def overall_consensus(ace: NDArray[np.floating]) -> FloatArray:
    """对全部议题取平均，得到每位专家的整体共识度 ACD。"""

    values = np.asarray(ace, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("ACE 必须是二维数组。")
    if np.any((values < -1.0e-12) | (values > 1.0 + 1.0e-12)):
        raise ValueError("ACE 必须位于 [0, 1]。")
    return np.clip(values.mean(axis=1), 0.0, 1.0)


def identify_disagreement(
    ace: NDArray[np.floating],
    acd: NDArray[np.floating],
    threshold: float,
) -> tuple[BoolArray, BoolArray]:
    """返回未协调专家掩码 ECH 和未协调议题掩码 APS。"""

    element = np.asarray(ace, dtype=np.float64)
    overall = np.asarray(acd, dtype=np.float64)
    if element.ndim != 2 or overall.shape != (element.shape[0],):
        raise ValueError("ACE 与 ACD 的专家维度必须一致。")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("共识门槛必须位于 (0, 1]。")

    expert_mask = overall < float(threshold)
    issue_mask = element < float(threshold)
    # 已协调专家不进入理论优化，其议题掩码也清零。
    issue_mask = issue_mask & expert_mask[:, None]
    return expert_mask, issue_mask


def group_reference(opinions: NDArray[np.floating]) -> FloatArray:
    """计算当前综合意见的逐议题算术平均。"""

    return _validate_opinions(opinions).mean(axis=0)


@dataclass(frozen=True)
class ConsensusMetrics:
    """一次共识测量的完整结果。"""

    similarity: FloatArray
    ace: FloatArray
    acd: FloatArray
    expert_mask: BoolArray
    issue_mask: BoolArray
    reference: FloatArray
    threshold: float

    @property
    def success(self) -> bool:
        return bool(not np.any(self.expert_mask))

    @property
    def mean_acd(self) -> float:
        return float(self.acd.mean())

    @property
    def min_acd(self) -> float:
        return float(self.acd.min())


def evaluate_consensus(opinions: NDArray[np.floating], threshold: float) -> ConsensusMetrics:
    """一次性计算相似度、ACE、ACD、ECH、APS 和群体参考意见。"""

    values = _validate_opinions(opinions)
    similarity = pairwise_similarity(values)
    ace = element_consensus(values, similarity)
    acd = overall_consensus(ace)
    expert_mask, issue_mask = identify_disagreement(ace, acd, threshold)
    return ConsensusMetrics(
        similarity=similarity,
        ace=ace,
        acd=acd,
        expert_mask=expert_mask,
        issue_mask=issue_mask,
        reference=group_reference(values),
        threshold=float(threshold),
    )

