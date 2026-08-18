from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.data.task_generator import StageBInstance, generate_stage_b_instance
from src.env.consensus_env import (
    ConsensusFeedbackEnv,
    build_state,
    compute_deficit_reward,
    compute_factorized_deficit_reward,
    compute_reward,
    compute_factorized_reward,
    consensus_deficit,
    consensus_reached,
)
from src.env.response_model import effective_adjustment, sample_response_rate
from src.model.harmony_optimizer import adjustment_distances, apply_theoretical_adjustment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return deepcopy(load_config(PROJECT_ROOT / "configs" / "base.yaml"))


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _instance_with_opinions(config: dict[str, object], opinions: np.ndarray) -> StageBInstance:
    base = generate_stage_b_instance(config, np.random.default_rng(100))
    return replace(base, initial_fused_opinions=_readonly(opinions))


def _stress_opinions() -> np.ndarray:
    return np.repeat(np.linspace(0.0, 1.0, 5)[:, None], 5, axis=1)


def test_build_state_has_33_dimensions_and_correct_sentinel_positions() -> None:
    state = build_state(
        np.linspace(0.1, 0.5, 5),
        np.linspace(0.2, 0.6, 5),
        np.linspace(0.7, 0.9, 5),
        np.linspace(0.0, 0.4, 5),
        np.full(5, -1.0),
        np.full(5, -1.0),
        round_index=0,
        max_rounds=8,
    )
    assert state.shape == (33,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state[[4, 10, 16, 22, 28]], -1.0)
    np.testing.assert_allclose(state[[5, 11, 17, 23, 29]], -1.0)
    assert state[-1] == 0.0


def test_environment_one_step_matches_manual_update_and_preserves_fixed_inputs() -> None:
    config = _config()
    config["response"]["response_noise_std"] = 0.0
    instance = _instance_with_opinions(config, _stress_opinions())
    fixed_snapshots = {
        "human": instance.task.human_opinions.copy(),
        "ai": instance.task.ai_opinions.copy(),
        "human_trust": instance.human_to_ai_trust.copy(),
        "ai_trust": instance.ai_to_human_information_trust.copy(),
    }
    env = ConsensusFeedbackEnv(
        config,
        np.random.default_rng(2026),
        response_types=("flexible",) * 5,
    )
    state, reset_info = env.reset(instance)
    assert state.shape == (33,)
    assert not reset_info["initial_success"]
    assert reset_info["optimizer_success"]

    before = env.current_opinions.copy()
    metrics = env.metrics
    theoretical = env.optimization.deltas.copy()
    table = config["response"]["response_table"]
    rates = np.asarray(
        [
            sample_response_rate(
                "flexible",
                value,
                table,
                0.0,
                np.random.default_rng(0),
                config["response"]["suggestion_bins"],
            )
            if value > 1.0e-12
            else 0.0
            for value in theoretical
        ]
    )
    expected_effective = effective_adjustment(theoretical, rates)
    expected = apply_theoretical_adjustment(
        before,
        expected_effective,
        metrics.issue_mask,
        metrics.reference,
    )

    next_state, reward, terminated, truncated, info = env.step([2, 2, 2, 2, 2])
    np.testing.assert_allclose(env.current_opinions, expected)
    np.testing.assert_allclose(info["effective_deltas"], expected_effective)
    assert next_state.shape == (33,)
    active = np.asarray(info["theoretical_deltas"]) > 1.0e-12
    recommendation_features = next_state[:30].reshape(5, 6)[:, 5]
    np.testing.assert_allclose(
        recommendation_features[active],
        info["recommended_deltas"][active],
    )
    np.testing.assert_allclose(recommendation_features[~active], -1.0)
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    np.testing.assert_allclose(instance.task.human_opinions, fixed_snapshots["human"])
    np.testing.assert_allclose(instance.task.ai_opinions, fixed_snapshots["ai"])
    np.testing.assert_allclose(instance.human_to_ai_trust, fixed_snapshots["human_trust"])
    np.testing.assert_allclose(
        instance.ai_to_human_information_trust,
        fixed_snapshots["ai_trust"],
    )


def test_continuous_step_matches_discrete_multiplier_one() -> None:
    config = _config()
    config["response"]["response_noise_std"] = 0.0
    instance = _instance_with_opinions(config, _stress_opinions())
    response_types = ("normal",) * 5
    discrete = ConsensusFeedbackEnv(
        config,
        np.random.default_rng(77),
        response_types=response_types,
    )
    continuous = ConsensusFeedbackEnv(
        config,
        np.random.default_rng(77),
        response_types=response_types,
    )
    discrete.reset(instance)
    continuous.reset(instance)

    discrete_result = discrete.step([2, 2, 2, 2, 2])
    continuous_result = continuous.step_continuous([1.0] * 5)

    np.testing.assert_allclose(discrete.current_opinions, continuous.current_opinions)
    assert discrete_result[1:4] == continuous_result[1:4]
    np.testing.assert_allclose(
        discrete_result[4]["recommended_deltas"],
        continuous_result[4]["recommended_deltas"],
    )
    assert continuous_result[4]["action_mode"] == "continuous"
    with pytest.raises(ValueError, match="连续倍率"):
        continuous.step_continuous([0.49] * 5)


def test_direct_guidance_uses_observable_disagreement_without_static_solver() -> None:
    config = _config()
    config["guidance"] = {
        "mode": "direct",
        "action_bounds": [0.01, 0.99],
    }
    config["response"]["response_noise_std"] = 0.0
    config["reward"]["recommendation_cost_weight"] = 2.0
    instance = _instance_with_opinions(config, _stress_opinions())
    env = ConsensusFeedbackEnv(
        config,
        np.random.default_rng(88),
        response_types=("flexible",) * 5,
    )
    state, reset_info = env.reset(instance)

    assert env.optimization is None
    assert reset_info["optimizer_success"]
    assert reset_info["guidance_mode"] == "direct"
    np.testing.assert_allclose(reset_info["theoretical_deltas"], 0.0)
    initial_distance = adjustment_distances(
        env.current_opinions,
        env.metrics.issue_mask,
        env.metrics.reference,
    )
    active = env.metrics.expert_mask & (initial_distance > 1.0e-12)
    expected_signal = np.where(
        active,
        np.maximum(0.0, env.planning_threshold - env.metrics.acd),
        0.0,
    )
    np.testing.assert_allclose(
        state[:30].reshape(5, 6)[:, 3],
        expected_signal,
    )
    before = env.current_opinions.copy()
    previous_metrics = env.metrics
    rates = np.full(5, -1.0)
    for expert in np.flatnonzero(active):
        rates[expert] = sample_response_rate(
            "flexible",
            0.30,
            config["response"]["response_table"],
            0.0,
            np.random.default_rng(0),
            config["response"]["suggestion_bins"],
        )
    effective = np.zeros(5)
    effective[active] = effective_adjustment(
        np.full(int(active.sum()), 0.30),
        rates[active],
    )
    expected = apply_theoretical_adjustment(
        before,
        effective,
        previous_metrics.issue_mask,
        previous_metrics.reference,
    )

    next_state, _, _, _, info = env.step_continuous([0.30] * 5)
    np.testing.assert_allclose(env.current_opinions, expected)
    np.testing.assert_allclose(info["recommended_deltas"][active], 0.30)
    np.testing.assert_allclose(info["recommended_deltas"][~active], 0.0)
    np.testing.assert_array_equal(info["active_expert_mask"], active)
    np.testing.assert_allclose(info["theoretical_deltas"], 0.0)
    assert info["guidance_mode"] == "direct"
    expected_recommendation_cost = float(
        np.mean(-2.0 * info["recommended_deltas"] ** 2)
    )
    assert info["reward"]["recommendation_cost"] == pytest.approx(
        expected_recommendation_cost
    )
    assert np.mean(info["factorized_reward"]["recommendation_cost"]) == pytest.approx(
        expected_recommendation_cost
    )
    assert not info["optimizer_failed"]
    assert env.optimization is None
    np.testing.assert_allclose(
        next_state[:30].reshape(5, 6)[:, 3],
        env._direct_state_signal(),
    )
    with pytest.raises(ValueError, match="直接建议模式"):
        env.step([0, 0, 0, 0, 0])


@pytest.mark.parametrize(
    ("signal_mode", "include_deficit"),
    (("adjustment_distance", False), ("distance_deficit_sum", True)),
)
def test_direct_state_can_replace_redundant_deficit_with_adjustment_distance(
    signal_mode: str,
    include_deficit: bool,
) -> None:
    config = _config()
    config["guidance"] = {
        "mode": "direct",
        "action_bounds": [0.01, 0.99],
        "state_signal": signal_mode,
    }
    instance = _instance_with_opinions(config, _stress_opinions())
    env = ConsensusFeedbackEnv(config, np.random.default_rng(89))
    state, _ = env.reset(instance)
    distance = adjustment_distances(
        env.current_opinions,
        env.metrics.issue_mask,
        env.metrics.reference,
    )
    expected = distance
    if include_deficit:
        expected = np.clip(
            distance + np.maximum(0.0, env.planning_threshold - env.metrics.acd),
            0.0,
            1.0,
        )
    expected = np.where(
        env.metrics.expert_mask & (distance > 1.0e-12),
        expected,
        0.0,
    )
    np.testing.assert_allclose(state[:30].reshape(5, 6)[:, 3], expected)
    np.testing.assert_array_equal(
        expected > 0.0,
        env.metrics.expert_mask & (distance > 1.0e-12),
    )


def test_deficit_reward_audits_remaining_and_unexecuted_recommendation_costs() -> None:
    config = _config()
    config["guidance"] = {
        "mode": "direct",
        "action_bounds": [0.01, 0.99],
    }
    config["response"]["response_noise_std"] = 0.0
    config["reward"].update(
        {
            "mode": "deficit",
            "deficit_epsilon": 1.0e-8,
            "deficit_progress_weight": 1.0,
            "modification_cost_weight": 1.5,
            "round_cost": 0.01,
            "success_bonus": 0.25,
            "timeout_penalty": 0.25,
            "recommendation_cost_weight": 0.01,
            "remaining_deficit_cost_weight": 0.20,
            "unexecuted_recommendation_cost_weight": 0.30,
        }
    )
    instance = _instance_with_opinions(config, _stress_opinions())
    env = ConsensusFeedbackEnv(
        config,
        np.random.default_rng(90),
        response_types=("normal",) * 5,
    )
    _, reset_info = env.reset(instance)
    _, _, _, _, info = env.step_continuous([0.40] * 5)
    active = np.asarray(info["active_expert_mask"], dtype=bool)
    recommended = np.asarray(info["recommended_deltas"], dtype=np.float64)
    rates = np.asarray(info["response_rates"], dtype=np.float64)
    normalizer = max(float(reset_info["initial_consensus_deficit"]), 1.0e-8)
    expected_remaining = float(
        np.mean(
            -0.20
            * np.maximum(0.0, env.success_threshold - env.metrics.acd)
            / normalizer
        )
    )
    waste_terms = np.zeros(5, dtype=np.float64)
    waste_terms[active] = -0.30 * recommended[active] * (1.0 - rates[active])
    expected_waste = float(waste_terms.mean())
    assert info["reward"]["remaining_deficit_cost"] == pytest.approx(
        expected_remaining
    )
    assert info["reward"]["unexecuted_recommendation_cost"] == pytest.approx(
        expected_waste
    )
    assert np.mean(info["factorized_reward"]["remaining_deficit_cost"]) == pytest.approx(
        expected_remaining
    )
    assert np.mean(
        info["factorized_reward"]["unexecuted_recommendation_cost"]
    ) == pytest.approx(expected_waste)
    assert np.mean(info["factorized_reward"]["total"]) == pytest.approx(
        info["reward"]["total"]
    )


def test_reward_components_match_hand_calculation() -> None:
    before = np.zeros((2, 2))
    after = np.full((2, 2), 0.1)
    reward = compute_reward(
        before,
        after,
        previous_mean_acd=0.5,
        current_mean_acd=0.6,
        success=True,
        timeout=False,
        consensus_improvement_weight=10.0,
        modification_cost_weight=0.5,
        round_cost=0.05,
        success_bonus=2.0,
        timeout_penalty=2.0,
    )
    assert reward.consensus_improvement == pytest.approx(1.0)
    assert reward.mean_modification == pytest.approx(0.2)
    assert reward.modification_cost == pytest.approx(-0.1)
    assert reward.total == pytest.approx(2.85)


def test_factorized_reward_mean_exactly_recovers_group_reward() -> None:
    before = np.array([[0.0, 0.0], [0.4, 0.5]])
    after = np.array([[0.1, 0.0], [0.35, 0.45]])
    previous_acd = np.array([0.70, 0.80])
    current_acd = np.array([0.75, 0.82])
    parameters = {
        "success": True,
        "timeout": False,
        "consensus_improvement_weight": 10.0,
        "modification_cost_weight": 0.5,
        "round_cost": 0.05,
        "success_bonus": 2.0,
        "timeout_penalty": 2.0,
    }
    group = compute_reward(
        before,
        after,
        float(previous_acd.mean()),
        float(current_acd.mean()),
        **parameters,
    )
    factorized = compute_factorized_reward(
        before,
        after,
        previous_acd,
        current_acd,
        **parameters,
    )
    assert np.mean(factorized["total"]) == pytest.approx(group.total)
    np.testing.assert_allclose(
        factorized["consensus_improvement"],
        [0.5, 0.2],
    )
    np.testing.assert_allclose(factorized["modification_cost"], [-0.05, -0.05])


def test_consensus_deficit_reward_matches_hand_calculation() -> None:
    before = np.array([[0.0, 0.0], [0.4, 0.5]])
    after = np.array([[0.1, 0.0], [0.35, 0.45]])
    previous_acd = np.array([0.70, 0.90])
    current_acd = np.array([0.80, 0.92])
    initial_deficit = consensus_deficit(previous_acd, 0.925)
    assert initial_deficit == pytest.approx(0.125)

    parameters = {
        "initial_deficit": initial_deficit,
        "threshold": 0.925,
        "deficit_epsilon": 1.0e-8,
        "progress_weight": 1.0,
        "modification_cost_weight": 0.5,
        "round_cost": 0.01,
        "success_bonus": 0.25,
        "timeout_penalty": 0.25,
        "success": False,
        "timeout": False,
    }
    group = compute_deficit_reward(
        before,
        after,
        previous_acd,
        current_acd,
        **parameters,
    )
    factorized = compute_factorized_deficit_reward(
        before,
        after,
        previous_acd,
        current_acd,
        **parameters,
    )

    assert group.consensus_improvement == pytest.approx(0.48)
    assert group.modification_cost == pytest.approx(-0.05)
    assert group.total == pytest.approx(0.42)
    np.testing.assert_allclose(
        factorized["consensus_improvement"],
        [0.80, 0.16],
    )
    assert np.mean(factorized["total"]) == pytest.approx(group.total)


def test_consensus_success_uses_solver_consistent_numerical_tolerance() -> None:
    assert consensus_reached(np.array([0.8499995, 0.9]), 0.85, 1.0e-6)
    assert not consensus_reached(np.array([0.849998, 0.9]), 0.85, 1.0e-6)


def test_planning_margin_separates_success_and_theoretical_targets() -> None:
    config = _config()
    config["consensus"]["threshold"] = 0.84
    config["consensus"]["planning_margin"] = 0.06
    # 前四位意见为 0，末位为 0.15 时，末位专家 ACD 恰为 0.85。
    opinions = np.repeat(np.array([0.0, 0.0, 0.0, 0.0, 0.15])[:, None], 5, axis=1)
    instance = _instance_with_opinions(config, opinions)
    env = ConsensusFeedbackEnv(config, np.random.default_rng(2))
    _, info = env.reset(instance)
    assert env.success_threshold == pytest.approx(0.84)
    assert env.planning_threshold == pytest.approx(0.90)
    assert info["initial_success"]
    assert np.any(info["theoretical_deltas"] > 0.0)


def test_already_coordinated_instance_is_terminal_at_reset() -> None:
    config = _config()
    instance = _instance_with_opinions(config, np.full((5, 5), 0.5))
    env = ConsensusFeedbackEnv(config, np.random.default_rng(1))
    _, info = env.reset(instance)
    assert env.done
    assert info["initial_success"]
    np.testing.assert_allclose(info["theoretical_deltas"], 0.0)
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0, 0, 0])


def test_max_rounds_produces_truncation_without_false_success() -> None:
    config = _config()
    config["consensus"]["max_rounds"] = 1
    config["response"]["response_noise_std"] = 0.0
    instance = _instance_with_opinions(config, _stress_opinions())
    env = ConsensusFeedbackEnv(
        config,
        np.random.default_rng(3),
        response_types=("stubborn",) * 5,
    )
    env.reset(instance)
    _, _, terminated, truncated, info = env.step([0, 0, 0, 0, 0])
    assert not terminated
    assert truncated
    assert info["timeout"]
    assert not info["success"]
    assert info["reward"]["timeout_penalty"] == -2.0
