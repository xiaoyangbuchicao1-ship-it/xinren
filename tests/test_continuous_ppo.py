from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.config import load_config
from src.experiments.continuous_ppo import (
    collect_continuous_rollout,
    create_continuous_trainer,
    evaluate_continuous_policy_on_cases,
)
from src.experiments.train_ppo import make_validation_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_continuous_trainer_starts_from_static_multiplier_one() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(config, torch.device("cpu"))
    action, _, _ = trainer.act(np.zeros(33, dtype=np.float32), deterministic=True)
    np.testing.assert_allclose(action, 1.0, atol=1.0e-7)


def test_random_continuous_actor_is_reproducible_without_static_prior() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    torch.manual_seed(123)
    first = create_continuous_trainer(
        config,
        torch.device("cpu"),
        actor_initialization="random",
    )
    first_action, _, _ = first.act(
        np.zeros(33, dtype=np.float32),
        deterministic=True,
    )
    torch.manual_seed(123)
    second = create_continuous_trainer(
        config,
        torch.device("cpu"),
        actor_initialization="random",
    )
    second_action, _, _ = second.act(
        np.zeros(33, dtype=np.float32),
        deterministic=True,
    )
    np.testing.assert_allclose(first_action, second_action, atol=1.0e-7)
    assert not np.allclose(first_action, 1.0, atol=1.0e-7)


def test_identified_continuous_actor_adds_only_known_expert_index() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(
        config,
        torch.device("cpu"),
        include_expert_identity=True,
    )
    state = torch.zeros((2, 33), dtype=torch.float32)
    features = trainer.actor.expert_features(state)
    assert features.shape == (2, 5, 14)
    np.testing.assert_allclose(features[0, :, -5:].detach().numpy(), np.eye(5))


def test_fixed_continuous_policy_evaluation_is_reproducible() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    cases = make_validation_cases(
        config,
        3,
        task_seed=31,
        type_seed=32,
        response_seed=33,
    )
    selector = lambda _state, env: np.full(env.num_experts, 0.9, dtype=np.float64)
    first, first_episodes = evaluate_continuous_policy_on_cases(config, cases, selector)
    second, second_episodes = evaluate_continuous_policy_on_cases(config, cases, selector)
    assert first == second
    assert [item.to_serializable() for item in first_episodes] == [
        item.to_serializable() for item in second_episodes
    ]
    assert first["active_multiplier_mean"] == pytest.approx(0.9)


def test_continuous_rollout_has_complete_episode_boundaries() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(config, torch.device("cpu"), minibatch_size=16)
    buffer, summary = collect_continuous_rollout(
        trainer,
        config,
        task_rng=np.random.default_rng(41),
        type_rng=np.random.default_rng(42),
        response_seed_rng=np.random.default_rng(43),
        episode_target=3,
        fixed_response_composition=(1, 1, 3),
    )
    assert summary["episodes"] == 3
    assert len(buffer) == summary["transitions"]
    assert sum(buffer.dones) == 3
    assert len(buffer.factorized_rewards) == len(buffer)
    assert 0.5 <= summary["active_multiplier_min"]
    assert summary["active_multiplier_max"] <= 1.25
    assert 0.0 <= summary["active_response_rate_mean"] <= 1.0
    assert 0.0 <= summary["active_recommendation_mean"] <= 1.0
    assert sum(summary["suggestion_bin_counts"]) == summary["active_multiplier_count"]
    assert len(summary["response_rate_mean_by_bin"]) == 3
    batch = buffer.to_batch(torch.device("cpu"), gamma=0.99, gae_lambda=0.95)
    assert batch.actions.dtype == torch.float32
    assert batch.actions.shape == (len(buffer), 5)
    factorized_returns = buffer.factorized_returns(
        torch.device("cpu"),
        gamma=0.99,
    )
    assert factorized_returns.shape == (len(buffer), 5)
    assert torch.isfinite(factorized_returns).all()


def test_continuous_rollout_accepts_fixed_expert_profile() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(config, torch.device("cpu"), minibatch_size=16)
    _, summary = collect_continuous_rollout(
        trainer,
        config,
        task_rng=np.random.default_rng(44),
        type_rng=np.random.default_rng(45),
        response_seed_rng=np.random.default_rng(46),
        episode_target=2,
        fixed_response_types=("flexible", "normal", "stubborn", "stubborn", "stubborn"),
    )
    assert summary["episodes"] == 2
