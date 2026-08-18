"""联合最大和谐度理论调整求解。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.model.consensus import evaluate_consensus, group_reference


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def adjustment_distances(
    opinions: NDArray[np.floating],
    issue_mask: NDArray[np.bool_],
    reference: NDArray[np.floating] | None = None,
) -> FloatArray:
    """计算每位专家在未协调议题上的平均潜在调整距离 psi。"""

    values = np.asarray(opinions, dtype=np.float64)
    mask = np.asarray(issue_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("意见与 APS 掩码必须是形状一致的二维数组。")
    target = group_reference(values) if reference is None else np.asarray(reference, dtype=np.float64)
    if target.shape != (values.shape[1],):
        raise ValueError("群体参考意见维度错误。")

    distances = np.zeros(values.shape[0], dtype=np.float64)
    absolute = np.abs(values - target[None, :])
    for expert in range(values.shape[0]):
        selected = mask[expert]
        if np.any(selected):
            distances[expert] = float(absolute[expert, selected].mean())
    return distances


def apply_theoretical_adjustment(
    opinions: NDArray[np.floating],
    deltas: NDArray[np.floating],
    issue_mask: NDArray[np.bool_],
    reference: NDArray[np.floating] | None = None,
) -> FloatArray:
    """只在 APS 中按专家独立 delta 向共同参考意见移动。"""

    values = np.asarray(opinions, dtype=np.float64)
    coefficients = np.asarray(deltas, dtype=np.float64)
    mask = np.asarray(issue_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("意见与 APS 掩码必须是形状一致的二维数组。")
    if coefficients.shape != (values.shape[0],):
        raise ValueError("每位专家必须对应一个调整系数。")
    if np.any((coefficients < 0.0) | (coefficients > 1.0)):
        raise ValueError("理论调整系数必须位于 [0, 1]。")

    target = group_reference(values) if reference is None else np.asarray(reference, dtype=np.float64)
    if target.shape != (values.shape[1],):
        raise ValueError("群体参考意见维度错误。")

    moved = (1.0 - coefficients[:, None]) * values + coefficients[:, None] * target[None, :]
    return np.where(mask, moved, values)


def validate_solution(
    opinions: NDArray[np.floating],
    deltas: NDArray[np.floating],
    issue_mask: NDArray[np.bool_],
    threshold: float,
    tolerance: float,
    reference: NDArray[np.floating] | None = None,
) -> tuple[bool, FloatArray, float, FloatArray]:
    """显式复核候选解的全部专家共识约束。"""

    adjusted = apply_theoretical_adjustment(opinions, deltas, issue_mask, reference)
    acd = evaluate_consensus(adjusted, threshold).acd
    maximum_violation = float(max(0.0, threshold - float(acd.min())))
    feasible = bool(maximum_violation <= tolerance)
    return feasible, acd, maximum_violation, adjusted


@dataclass(frozen=True)
class HarmonyOptimizationResult:
    """SLSQP 联合求解的结构化输出。"""

    success: bool
    deltas: FloatArray
    adjusted_opinions: FloatArray
    adjusted_acd: FloatArray
    objective: float
    min_acd: float
    max_constraint_violation: float
    message: str
    iterations: int
    attempts: int
    solver_success: bool

    def to_serializable(self) -> dict[str, object]:
        return {
            "success": self.success,
            "deltas": self.deltas.tolist(),
            "adjusted_opinions": self.adjusted_opinions.tolist(),
            "adjusted_acd": self.adjusted_acd.tolist(),
            "objective": self.objective,
            "min_acd": self.min_acd,
            "max_constraint_violation": self.max_constraint_violation,
            "message": self.message,
            "iterations": self.iterations,
            "attempts": self.attempts,
            "solver_success": self.solver_success,
        }


def _initial_points(variable_count: int, restart_count: int) -> list[FloatArray]:
    """生成确定性初始点，确保固定种子外也可复现求解过程。"""

    if restart_count <= 0:
        raise ValueError("求解重启次数必须为正整数。")
    levels = np.linspace(0.0, 1.0, num=restart_count, dtype=np.float64)
    return [np.full(variable_count, level, dtype=np.float64) for level in levels]


def solve_harmony_adjustment(
    opinions: NDArray[np.floating],
    threshold: float,
    *,
    max_iterations: int = 500,
    ftol: float = 1.0e-9,
    constraint_tolerance: float = 1.0e-6,
    restarts: int = 3,
) -> HarmonyOptimizationResult:
    """联合求解所有未协调专家的理论最小调整比例。"""

    values = np.asarray(opinions, dtype=np.float64)
    metrics = evaluate_consensus(values, threshold)
    expert_indices = np.flatnonzero(metrics.expert_mask)
    full_zero = np.zeros(values.shape[0], dtype=np.float64)

    if expert_indices.size == 0:
        return HarmonyOptimizationResult(
            success=True,
            deltas=full_zero,
            adjusted_opinions=values.copy(),
            adjusted_acd=metrics.acd.copy(),
            objective=0.0,
            min_acd=float(metrics.acd.min()),
            max_constraint_violation=0.0,
            message="初始状态已经达到共识。",
            iterations=0,
            attempts=0,
            solver_success=True,
        )

    psi = adjustment_distances(values, metrics.issue_mask, metrics.reference)

    def expand(active_deltas: NDArray[np.floating]) -> FloatArray:
        complete = np.zeros(values.shape[0], dtype=np.float64)
        complete[expert_indices] = active_deltas
        return complete

    def objective(active_deltas: NDArray[np.floating]) -> float:
        return float(np.dot(active_deltas, psi[expert_indices]))

    def constraints(active_deltas: NDArray[np.floating]) -> FloatArray:
        complete = expand(active_deltas)
        adjusted = apply_theoretical_adjustment(
            values,
            complete,
            metrics.issue_mask,
            metrics.reference,
        )
        return evaluate_consensus(adjusted, threshold).acd - threshold

    best_result: HarmonyOptimizationResult | None = None
    best_failed: HarmonyOptimizationResult | None = None
    for attempt, initial in enumerate(_initial_points(expert_indices.size, restarts), start=1):
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * expert_indices.size,
            constraints=[{"type": "ineq", "fun": constraints}],
            options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
        )
        complete = expand(np.clip(result.x, 0.0, 1.0))
        feasible, adjusted_acd, violation, adjusted = validate_solution(
            values,
            complete,
            metrics.issue_mask,
            threshold,
            constraint_tolerance,
            metrics.reference,
        )
        verified_success = bool(result.success and feasible)
        candidate = HarmonyOptimizationResult(
            success=verified_success,
            deltas=complete,
            adjusted_opinions=adjusted,
            adjusted_acd=adjusted_acd,
            objective=float(objective(complete[expert_indices])),
            min_acd=float(adjusted_acd.min()),
            max_constraint_violation=violation,
            message=str(result.message),
            iterations=int(getattr(result, "nit", 0)),
            attempts=attempt,
            solver_success=bool(result.success),
        )

        if verified_success and (
            best_result is None or candidate.objective < best_result.objective
        ):
            best_result = candidate
        if best_failed is None or candidate.max_constraint_violation < best_failed.max_constraint_violation:
            best_failed = candidate

    if best_result is not None:
        return best_result
    assert best_failed is not None
    return HarmonyOptimizationResult(
        success=False,
        deltas=best_failed.deltas,
        adjusted_opinions=best_failed.adjusted_opinions,
        adjusted_acd=best_failed.adjusted_acd,
        objective=best_failed.objective,
        min_acd=best_failed.min_acd,
        max_constraint_violation=best_failed.max_constraint_violation,
        message=f"所有 SLSQP 重启均未通过求解器成功与显式约束复核；最佳消息：{best_failed.message}",
        iterations=best_failed.iterations,
        attempts=restarts,
        solver_success=False,
    )

