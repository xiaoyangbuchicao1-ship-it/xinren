"""专家异质反馈响应模型。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def suggestion_bin(
    suggested_adjustment: float,
    bins: Sequence[float] = (0.0, 0.3, 0.7, 1.0),
) -> int:
    """返回建议量档位：0=小幅、1=中等、2=大幅。"""

    boundaries = np.asarray(bins, dtype=np.float64)
    if boundaries.shape != (4,) or not np.all(np.diff(boundaries) > 0.0):
        raise ValueError("建议档位必须由 4 个严格递增边界构成。")
    if not np.isclose(boundaries[0], 0.0) or not np.isclose(boundaries[-1], 1.0):
        raise ValueError("建议档位必须覆盖闭区间 [0, 1]。")

    value = float(suggested_adjustment)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("建议调整比例必须位于 [0, 1]。")
    # side=right 使 0.3 归入中等档，0.7 归入大幅档。
    return int(np.searchsorted(boundaries[1:-1], value, side="right"))


def sample_response_types(
    num_experts: int,
    type_names: Sequence[str],
    probabilities: Sequence[float],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """为一个元任务采样任务内固定的隐藏响应类型。"""

    names = tuple(str(name) for name in type_names)
    probs = np.asarray(probabilities, dtype=np.float64)
    if num_experts <= 0 or not names or probs.shape != (len(names),):
        raise ValueError("响应类型名称、概率与专家数量不匹配。")
    if np.any(probs < 0.0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("响应类型概率必须非负且总和为 1。")
    sampled = rng.choice(np.asarray(names, dtype=object), size=num_experts, p=probs)
    return tuple(str(value) for value in sampled.tolist())


def sample_response_types_from_counts(
    type_names: Sequence[str],
    type_counts: Sequence[int],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """按固定类型人数构成生成一次随机专家位置排列。"""

    names = tuple(str(name) for name in type_names)
    counts = np.asarray(type_counts)
    if not names or len(set(names)) != len(names):
        raise ValueError("响应类型名称必须非空且不能重复。")
    if counts.ndim != 1 or counts.shape != (len(names),):
        raise ValueError("类型人数必须与响应类型名称一一对应。")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("类型人数必须是非负整数。")
    if int(counts.sum()) <= 0:
        raise ValueError("类型总人数必须为正整数。")
    expanded = np.repeat(np.asarray(names, dtype=object), counts.astype(np.int64))
    shuffled = rng.permutation(expanded)
    return tuple(str(value) for value in shuffled.tolist())


def sample_response_rate(
    response_type: str,
    suggested_adjustment: float,
    response_table: Mapping[str, Sequence[float]],
    noise_std: float,
    rng: np.random.Generator,
    bins: Sequence[float] = (0.0, 0.3, 0.7, 1.0),
    interpolation: str = "step",
) -> float:
    """按隐藏类型采样执行率；可选档位阶梯或档位中心线性插值。"""

    if response_type not in response_table:
        raise ValueError(f"未知响应类型：{response_type}")
    means = np.asarray(response_table[response_type], dtype=np.float64)
    if means.shape != (3,) or np.any((means < 0.0) | (means > 1.0)):
        raise ValueError("每种响应类型必须提供 3 个位于 [0, 1] 的均值。")
    if noise_std < 0.0:
        raise ValueError("响应噪声标准差不能为负。")

    if interpolation == "step":
        mean = float(means[suggestion_bin(suggested_adjustment, bins)])
    elif interpolation == "linear":
        boundaries = np.asarray(bins, dtype=np.float64)
        # 三个响应表值分别解释为小/中/大建议区间中心处的期望执行率。
        centers = 0.5 * (boundaries[:-1] + boundaries[1:])
        mean = float(
            np.interp(
                float(suggested_adjustment),
                centers,
                means,
                left=float(means[0]),
                right=float(means[-1]),
            )
        )
    else:
        raise ValueError("响应插值模式只支持step或linear。")
    noise = float(rng.normal(0.0, noise_std)) if noise_std > 0.0 else 0.0
    return float(np.clip(mean + noise, 0.0, 1.0))


def effective_adjustment(
    suggested_adjustment: NDArray[np.floating] | Sequence[float],
    response_rate: NDArray[np.floating] | Sequence[float],
) -> FloatArray:
    """计算实际生效调整比例 delta_eff=q*delta_rec。"""

    suggested = np.asarray(suggested_adjustment, dtype=np.float64)
    rates = np.asarray(response_rate, dtype=np.float64)
    if suggested.shape != rates.shape:
        raise ValueError("建议量与响应率数组形状必须一致。")
    if np.any((suggested < 0.0) | (suggested > 1.0)):
        raise ValueError("建议量必须位于 [0, 1]。")
    if np.any((rates < 0.0) | (rates > 1.0)):
        raise ValueError("响应率必须位于 [0, 1]。")
    return suggested * rates


def action_to_multiplier(
    actions: NDArray[np.integer] | Sequence[int],
    multipliers: Sequence[float],
) -> FloatArray:
    """将每位专家的离散动作索引转换为反馈倍率。"""

    indices = np.asarray(actions, dtype=np.int64)
    values = np.asarray(multipliers, dtype=np.float64)
    if indices.ndim != 1 or values.ndim != 1 or values.size < 1:
        raise ValueError("动作索引和倍率必须是一维数组。")
    if np.any(indices < 0) or np.any(indices >= values.size):
        raise ValueError("动作索引超出倍率表范围。")
    if np.any(values <= 0.0):
        raise ValueError("反馈倍率必须为正数。")
    return values[indices]
