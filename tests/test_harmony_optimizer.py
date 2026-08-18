from __future__ import annotations

import numpy as np

from src.model.consensus import evaluate_consensus
from src.model.harmony_optimizer import (
    adjustment_distances,
    apply_theoretical_adjustment,
    solve_harmony_adjustment,
    validate_solution,
)


def test_theoretical_adjustment_respects_issue_mask_and_delta_boundaries() -> None:
    opinions = np.array([[0.0, 0.2], [1.0, 0.8]])
    issue_mask = np.array([[True, False], [True, False]])
    zero = apply_theoretical_adjustment(opinions, np.array([0.0, 0.0]), issue_mask)
    np.testing.assert_allclose(zero, opinions)

    full = apply_theoretical_adjustment(opinions, np.array([1.0, 1.0]), issue_mask)
    np.testing.assert_allclose(full[:, 0], [0.5, 0.5])
    np.testing.assert_allclose(full[:, 1], opinions[:, 1])


def test_adjustment_distance_uses_only_disagreement_issues() -> None:
    opinions = np.array([[0.0, 0.4], [1.0, 0.6]])
    issue_mask = np.array([[True, False], [True, False]])
    np.testing.assert_allclose(adjustment_distances(opinions, issue_mask), [0.5, 0.5])


def test_joint_optimizer_matches_two_expert_grid_optimum() -> None:
    opinions = np.array([[0.0], [1.0]])
    result = solve_harmony_adjustment(opinions, threshold=0.8, restarts=3)
    assert result.success
    assert result.max_constraint_violation <= 1.0e-6
    assert result.objective <= 0.800001
    assert result.deltas.sum() >= 1.6 - 1.0e-6

    # 0.01 网格穷举验证该问题的最优目标约为 0.8。
    grid_best = float("inf")
    for first in np.linspace(0.0, 1.0, 101):
        for second in np.linspace(0.0, 1.0, 101):
            deltas = np.array([first, second])
            mask = np.ones_like(opinions, dtype=bool)
            feasible, _, _, _ = validate_solution(
                opinions,
                deltas,
                mask,
                threshold=0.8,
                tolerance=1.0e-12,
            )
            if feasible:
                grid_best = min(grid_best, 0.5 * (first + second))
    assert abs(result.objective - grid_best) <= 1.0e-5


def test_already_coordinated_problem_returns_zero_adjustment() -> None:
    opinions = np.full((5, 5), 0.5)
    result = solve_harmony_adjustment(opinions, threshold=0.85)
    assert result.success
    np.testing.assert_allclose(result.deltas, 0.0)
    np.testing.assert_allclose(result.adjusted_acd, 1.0)
    assert result.iterations == 0


def test_fixed_problem_is_deterministic_and_all_constraints_hold() -> None:
    opinions = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.8, 0.7],
            [0.9, 0.8],
        ]
    )
    first = solve_harmony_adjustment(opinions, threshold=0.75, restarts=3)
    second = solve_harmony_adjustment(opinions, threshold=0.75, restarts=3)
    assert first.success and second.success
    np.testing.assert_allclose(first.deltas, second.deltas, atol=1.0e-8)
    assert np.min(evaluate_consensus(first.adjusted_opinions, 0.75).acd) >= 0.75 - 1.0e-6

