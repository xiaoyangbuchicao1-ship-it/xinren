from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.networks import ContinuousFactorizedActor, FactorizedActor, ValueNetwork
from src.agents.ppo import (
    ContinuousRolloutBuffer,
    PPOBatch,
    PPOTrainer,
    balanced_minibatch_indices,
    clipped_surrogate_loss,
    compute_discounted_returns,
    compute_gae,
)


def _trainer() -> PPOTrainer:
    return PPOTrainer(
        FactorizedActor(33, 5, 4, hidden_sizes=(16, 16)),
        ValueNetwork(33, hidden_sizes=(16, 16)),
        learning_rate=1.0e-3,
        clip_range=0.2,
        update_epochs=2,
        minibatch_size=8,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_gradient_norm=0.5,
        target_kl=1.0,
    )


def test_compute_gae_matches_two_step_hand_calculation() -> None:
    advantages, returns = compute_gae(
        np.array([1.0, 1.0]),
        np.array([0.5, 0.25]),
        np.array([False, True]),
        next_value=99.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages, [1.5, 0.75])
    np.testing.assert_allclose(returns, [2.0, 1.0])


def test_done_boundary_prevents_advantage_leak_between_episodes() -> None:
    advantages, _ = compute_gae(
        np.array([1.0, 10.0]),
        np.array([0.0, 0.0]),
        np.array([True, True]),
        next_value=0.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages, [1.0, 10.0])


def test_discounted_returns_stop_at_episode_boundaries() -> None:
    returns = compute_discounted_returns(
        np.array([1.0, 2.0, 10.0]),
        np.array([False, True, True]),
        gamma=0.5,
    )
    np.testing.assert_allclose(returns, [2.0, 2.0, 10.0])


def test_monte_carlo_batch_advantages_ignore_wrong_critic_values() -> None:
    buffer = ContinuousRolloutBuffer()
    state = np.zeros(33, dtype=np.float32)
    action = np.ones(5, dtype=np.float32)
    for reward, value, done in [(1.0, 100.0, False), (2.0, -100.0, True)]:
        buffer.add(state, action, reward, value, 0.0, done)
    batch = buffer.to_batch(
        torch.device("cpu"),
        gamma=0.5,
        gae_lambda=0.95,
        normalize_advantages=False,
        advantage_estimator="monte_carlo",
    )
    torch.testing.assert_close(batch.returns, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(batch.advantages, torch.tensor([2.0, 2.0]))


def test_clipped_surrogate_uses_clipping_boundary() -> None:
    new_log = torch.log(torch.tensor([2.0, 0.5]))
    old_log = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])
    loss, _, fraction = clipped_surrogate_loss(new_log, old_log, advantages, 0.2)
    # 两个样本都被裁剪：目标分别为 1.2 和 -0.8，负均值为 -0.2。
    assert loss.item() == pytest.approx(-0.2)
    assert fraction.item() == 1.0


def test_minibatches_are_balanced_without_tiny_tail() -> None:
    parts = balanced_minibatch_indices(np.arange(260), minibatch_size=256)
    sizes = [part.size for part in parts]
    assert sizes == [130, 130]
    np.testing.assert_array_equal(np.concatenate(parts), np.arange(260))


def test_minibatch_partition_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="非空一维"):
        balanced_minibatch_indices(np.array([]), minibatch_size=8)
    with pytest.raises(ValueError, match="正整数"):
        balanced_minibatch_indices(np.arange(8), minibatch_size=0)


def test_ppo_update_changes_parameters_and_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(3)
    trainer = _trainer()
    states = torch.randn(32, 33)
    with torch.no_grad():
        actions, old_log_probabilities, _ = trainer.actor.act(states)
    batch = PPOBatch(
        states=states,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        returns=torch.randn(32),
        advantages=torch.randn(32),
    )
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    metrics = trainer.update(batch, np.random.default_rng(5))
    assert all(np.isfinite(value) for value in metrics.to_serializable().values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, trainer.actor.parameters(), strict=True)
    )
    assert metrics.actor_gradient_norm > 0.0
    assert metrics.critic_gradient_norm > 0.0
    assert trainer.actor_optimizer is not trainer.critic_optimizer

    expected = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    path = trainer.save_checkpoint(tmp_path / "model.pt", {"update": 2})
    with torch.no_grad():
        for parameter in trainer.actor.parameters():
            parameter.add_(1.0)
    metadata = trainer.load_checkpoint(path)
    assert metadata == {"update": 2}
    for restored, wanted in zip(trainer.actor.parameters(), expected, strict=True):
        torch.testing.assert_close(restored, wanted)


def test_ppo_update_accepts_one_transition_for_one_shot_meta_adaptation() -> None:
    """一步即成功的支持 episode 也必须能完成 one-shot 内循环更新。"""

    torch.manual_seed(13)
    trainer = _trainer()
    states = torch.randn(1, 33)
    with torch.no_grad():
        actions, old_log_probabilities, _ = trainer.actor.act(states)
        values = trainer.critic(states)
    batch = PPOBatch(
        states=states,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        returns=values + 0.8,
        # 单样本时 RolloutBuffer 不做标准化，保留可学习的非零优势。
        advantages=torch.tensor([0.8]),
    )

    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    metrics = trainer.update(batch, np.random.default_rng(17))

    assert all(np.isfinite(value) for value in metrics.to_serializable().values())
    assert metrics.minibatches >= 1
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, trainer.actor.parameters(), strict=True)
    )


def test_continuous_rollout_and_ppo_update_preserve_float_actions() -> None:
    """连续倍率不能在缓存中被误转成离散整数索引。"""

    torch.manual_seed(23)
    actor = ContinuousFactorizedActor(33, 5, hidden_sizes=(16, 16))
    actor.initialize_multiplier_prior(1.0, initial_log_std=-0.8)
    trainer = PPOTrainer(
        actor,
        ValueNetwork(33, hidden_sizes=(16, 16)),
        learning_rate=1.0e-3,
        clip_range=0.2,
        update_epochs=2,
        minibatch_size=8,
        entropy_coefficient=1.0e-4,
        value_coefficient=0.5,
        max_gradient_norm=0.5,
        target_kl=1.0,
    )
    buffer = ContinuousRolloutBuffer()
    for index in range(16):
        state = np.random.default_rng(index).normal(size=33).astype(np.float32)
        action, log_probability, value = trainer.act(state)
        buffer.add(
            state,
            action,
            reward=float(index % 3) - 0.5,
            value=value,
            log_probability=log_probability,
            done=bool(index % 4 == 3),
        )
    batch = buffer.to_batch(
        torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert batch.actions.dtype == torch.float32
    assert batch.actions.shape == (16, 5)
    assert torch.all((batch.actions >= 0.5) & (batch.actions <= 1.25))

    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    metrics = trainer.update(batch, np.random.default_rng(24))
    assert all(np.isfinite(value) for value in metrics.to_serializable().values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, trainer.actor.parameters(), strict=True)
    )


def test_factorized_returns_stop_at_episode_boundaries() -> None:
    buffer = ContinuousRolloutBuffer()
    state = np.zeros(33, dtype=np.float32)
    action = np.ones(5, dtype=np.float32)
    for reward, done in [([1.0] * 5, True), ([10.0] * 5, True)]:
        buffer.add(
            state,
            action,
            reward=float(np.mean(reward)),
            value=0.0,
            log_probability=0.0,
            done=done,
            factorized_reward=np.asarray(reward),
        )
    returns = buffer.factorized_returns(torch.device("cpu"), gamma=1.0)
    torch.testing.assert_close(
        returns,
        torch.tensor([[1.0] * 5, [10.0] * 5]),
    )


def test_continuous_rollout_extend_preserves_episode_boundaries() -> None:
    first = ContinuousRolloutBuffer()
    second = ContinuousRolloutBuffer()
    state = np.zeros(33, dtype=np.float32)
    action = np.ones(5, dtype=np.float32)
    first.add(state, action, 1.0, 0.0, 0.0, True)
    second.add(state, action, 10.0, 0.0, 0.0, True)
    first.extend(second)
    batch = first.to_batch(
        torch.device("cpu"),
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_estimator="monte_carlo",
    )
    torch.testing.assert_close(batch.returns, torch.tensor([1.0, 10.0]))
