from __future__ import annotations

import torch

from src.agents.networks import ContinuousFactorizedActor, FactorizedActor, ValueNetwork


def test_factorized_actor_shapes_probabilities_and_joint_log_probability() -> None:
    torch.manual_seed(1)
    actor = FactorizedActor(33, 5, 4, hidden_sizes=(32, 32))
    states = torch.randn(7, 33)
    logits = actor(states)
    assert logits.shape == (7, 5, 4)
    probabilities = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(7, 5))

    actions, joint_log_probability, entropy = actor.act(states)
    distribution = actor.distribution(states)
    manual = distribution.log_prob(actions).sum(dim=-1)
    torch.testing.assert_close(joint_log_probability, manual)
    assert actions.shape == (7, 5)
    assert entropy.shape == (7,)


def test_deterministic_actor_and_value_network_outputs_are_finite() -> None:
    actor = FactorizedActor(33, 5, 4)
    critic = ValueNetwork(33)
    states = torch.zeros(3, 33)
    first, _, _ = actor.act(states, deterministic=True)
    second, _, _ = actor.act(states, deterministic=True)
    torch.testing.assert_close(first, second)
    values = critic(states)
    assert values.shape == (3,)
    assert torch.isfinite(values).all()


def test_theory_guided_action_prior_has_requested_probability() -> None:
    actor = FactorizedActor(33, 5, 4)
    actor.initialize_action_prior(2, 0.55)
    probabilities = actor.distribution(torch.zeros(1, 33)).probs
    torch.testing.assert_close(
        probabilities[0, :, 2],
        torch.full((5,), 0.55),
    )
    torch.testing.assert_close(
        probabilities[0, :, [0, 1, 3]],
        torch.full((5, 3), 0.15),
    )
    actions, _, _ = actor.act(torch.zeros(1, 33), deterministic=True)
    torch.testing.assert_close(actions, torch.full((1, 5), 2))


def test_actor_is_equivariant_to_expert_permutation() -> None:
    torch.manual_seed(9)
    actor = FactorizedActor(33, 5, 4)
    states = torch.randn(3, 33)
    local = states[:, :30].reshape(3, 5, 6)
    group = states[:, 30:]
    permutation = torch.tensor([2, 4, 0, 3, 1])
    permuted_states = torch.cat(
        [local[:, permutation].reshape(3, 30), group],
        dim=-1,
    )
    original_logits = actor(states)
    permuted_logits = actor(permuted_states)
    torch.testing.assert_close(permuted_logits, original_logits[:, permutation])


def test_candidate_features_encode_recommendation_and_suggestion_bin() -> None:
    actor = FactorizedActor(33, 5, 4)
    states = torch.zeros(1, 33)
    local = states[:, :30].reshape(1, 5, 6)
    local[0, 0, 3] = 0.4

    features = actor.candidate_features(states)
    assert features.shape == (1, 5, 4, 14)
    torch.testing.assert_close(
        features[0, 0, :, 9],
        torch.tensor([0.5, 0.75, 1.0, 1.25]),
    )
    torch.testing.assert_close(
        features[0, 0, :, 10],
        torch.tensor([0.2, 0.3, 0.4, 0.5]),
    )
    torch.testing.assert_close(
        features[0, 0, :, 11:],
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )


def test_continuous_actor_prior_bounds_and_log_probability_consistency() -> None:
    """连续 Actor 必须从倍率1.0出发，并保持动作概率计算自洽。"""

    torch.manual_seed(21)
    actor = ContinuousFactorizedActor(33, 5, hidden_sizes=(32, 32))
    actor.initialize_multiplier_prior(1.0, initial_log_std=-1.0)
    states = torch.randn(11, 33)

    deterministic, _, _ = actor.act(states, deterministic=True)
    torch.testing.assert_close(deterministic, torch.ones_like(deterministic))

    actions, sampled_log_probability, entropy = actor.act(states)
    evaluated_log_probability, evaluated_entropy = actor.evaluate_actions(states, actions)
    assert actions.shape == (11, 5)
    assert sampled_log_probability.shape == entropy.shape == (11,)
    assert torch.all((actions >= 0.5) & (actions <= 1.25))
    assert torch.isfinite(actions).all()
    assert torch.isfinite(sampled_log_probability).all()
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(
        evaluated_log_probability,
        sampled_log_probability,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(evaluated_entropy, entropy, rtol=1.0e-5, atol=1.0e-5)


def test_continuous_actor_residual_prior_is_centered_but_state_dependent() -> None:
    """残差先验应保留静态锚点，同时不能把所有状态写死为倍率1.0。"""

    torch.manual_seed(25)
    actor = ContinuousFactorizedActor(33, 5, hidden_sizes=(32, 32))
    actor.initialize_multiplier_residual_prior(
        preferred_multiplier=1.0,
        initial_log_std=-1.0,
        mean_head_gain=0.15,
    )
    states = torch.randn(64, 33)
    deterministic, log_probability, entropy = actor.act(states, deterministic=True)

    assert deterministic.shape == (64, 5)
    assert torch.isfinite(deterministic).all()
    assert torch.isfinite(log_probability).all()
    assert torch.isfinite(entropy).all()
    assert torch.all((deterministic >= 0.5) & (deterministic <= 1.25))
    assert float(deterministic.std().item()) > 1.0e-4
    assert float(abs(deterministic.mean().item() - 1.0)) < 0.08


def test_continuous_actor_joint_probability_ignores_inactive_experts() -> None:
    torch.manual_seed(24)
    actor = ContinuousFactorizedActor(33, 5, hidden_sizes=(16, 16))
    states = torch.zeros(3, 33)
    local = states[:, :30].reshape(3, 5, 6)
    local[:, 0, 3] = 0.2
    local[:, 3, 3] = 0.4
    actions, joint, entropy = actor.act(states)
    per_expert_log, per_expert_entropy = actor.evaluate_actions_per_expert(
        states,
        actions,
    )
    torch.testing.assert_close(joint, per_expert_log[:, [0, 3]].sum(dim=-1))
    torch.testing.assert_close(
        entropy,
        per_expert_entropy[:, [0, 3]].mean(dim=-1),
    )


def test_identified_actor_fast_offset_changes_only_one_expert_mean() -> None:
    actor = ContinuousFactorizedActor(
        33,
        5,
        hidden_sizes=(16, 16),
        include_expert_identity=True,
    )
    actor.initialize_multiplier_prior(1.0)
    states = torch.zeros(4, 33)
    before = actor.distribution(states).mean.detach().clone()
    with torch.no_grad():
        actor.expert_mean_offsets[2] = 0.4
    after = actor.distribution(states).mean.detach()
    expected = torch.zeros_like(after)
    expected[:, 2] = 0.4
    torch.testing.assert_close(after - before, expected)
    assert actor.fast_adaptation_parameters() == (actor.expert_mean_offsets,)


def test_continuous_actor_is_equivariant_to_expert_permutation() -> None:
    """交换专家位置只应交换对应倍率，不应改变策略规则。"""

    torch.manual_seed(22)
    actor = ContinuousFactorizedActor(33, 5)
    states = torch.randn(4, 33)
    local = states[:, :30].reshape(4, 5, 6)
    group = states[:, 30:]
    permutation = torch.tensor([4, 1, 3, 0, 2])
    permuted_states = torch.cat([local[:, permutation].reshape(4, 30), group], dim=-1)

    original_distribution = actor.distribution(states)
    permuted_distribution = actor.distribution(permuted_states)
    torch.testing.assert_close(
        permuted_distribution.mean,
        original_distribution.mean[:, permutation],
    )
    torch.testing.assert_close(
        permuted_distribution.stddev,
        original_distribution.stddev[:, permutation],
    )
