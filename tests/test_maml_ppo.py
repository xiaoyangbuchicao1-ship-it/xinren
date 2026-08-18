from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.agents.maml_ppo import (
    FirstOrderMetaOptimizer,
    aggregate_fomaml_gradients,
    clone_task_trainer,
    compute_second_order_fast_query_gradients,
    compute_query_gradients,
    differentiable_fast_adaptation,
    gradient_pairwise_cosines,
    sample_meta_task,
)
from src.agents.ppo import PPOBatch
from src.common.config import load_config
from src.experiments.train_ppo import collect_rollout, create_trainer
from src.experiments.continuous_ppo import create_continuous_trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_batch(trainer: object, count: int = 12) -> PPOBatch:
    generator = torch.Generator(device="cpu").manual_seed(7)
    states = torch.randn(count, 33, generator=generator)
    with torch.no_grad():
        actions, old_log_probabilities, _ = trainer.actor.act(states)
        values = trainer.critic(states)
    advantages = torch.linspace(-1.0, 1.0, count)
    returns = values + torch.linspace(0.2, 1.0, count)
    return PPOBatch(
        states=states,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        returns=returns,
        advantages=advantages,
    )


def test_fixed_meta_task_rollout_collects_exact_complete_episodes() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_trainer(config, torch.device("cpu"), minibatch_size=32)
    fixed_types = ("stubborn",) * 5
    with patch(
        "src.experiments.train_ppo.sample_response_types",
        side_effect=AssertionError("固定元任务不应重新采样响应类型"),
    ):
        buffer, summary = collect_rollout(
            trainer,
            config,
            None,
            task_rng=np.random.default_rng(1),
            type_rng=np.random.default_rng(2),
            response_seed_rng=np.random.default_rng(3),
            episode_target=2,
            fixed_response_types=fixed_types,
        )
    assert summary["episodes"] == 2
    assert len(buffer) == summary["transitions"]
    assert buffer.dones.count(True) == 2


def test_task_clone_update_does_not_modify_meta_parameters() -> None:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml"))
    meta = create_trainer(config, torch.device("cpu"), minibatch_size=32)
    task = clone_task_trainer(meta, inner_learning_rate=3.0e-4)
    before = [parameter.detach().clone() for parameter in meta.actor.parameters()]
    batch = _synthetic_batch(task)
    task.update(batch, np.random.default_rng(5))
    assert any(
        not torch.equal(left, right)
        for left, right in zip(task.actor.parameters(), meta.actor.parameters())
    )
    for parameter, expected in zip(meta.actor.parameters(), before):
        torch.testing.assert_close(parameter, expected)


def test_task_clone_can_use_sgd_for_maml_inner_loop() -> None:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml"))
    meta = create_trainer(config, torch.device("cpu"), minibatch_size=32)
    task = clone_task_trainer(
        meta,
        inner_learning_rate=1.0e-2,
        inner_optimizer="sgd",
    )
    assert isinstance(task.actor_optimizer, torch.optim.SGD)
    assert isinstance(task.critic_optimizer, torch.optim.SGD)


def test_actor_only_inner_update_keeps_shared_critic_fixed() -> None:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml"))
    meta = create_trainer(config, torch.device("cpu"), minibatch_size=32)
    task = clone_task_trainer(
        meta,
        inner_learning_rate=1.0e-2,
        inner_optimizer="sgd",
    )
    actor_before = [parameter.detach().clone() for parameter in task.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in task.critic.parameters()]
    metrics = task.update_actor_only(_synthetic_batch(task), np.random.default_rng(6))
    assert metrics.critic_gradient_norm == 0.0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(actor_before, task.actor.parameters(), strict=True)
    )
    for previous, current in zip(
        critic_before,
        task.critic.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(previous, current)


def test_query_gradients_are_finite_and_meta_step_changes_parameters() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    meta = create_trainer(config, torch.device("cpu"), minibatch_size=32)
    first_task = clone_task_trainer(meta, inner_learning_rate=3.0e-4)
    second_task = clone_task_trainer(meta, inner_learning_rate=3.0e-4)
    first = compute_query_gradients(first_task, _synthetic_batch(first_task))
    second = compute_query_gradients(second_task, _synthetic_batch(second_task))
    assert len(first.actor_gradients) == len(tuple(meta.actor.parameters()))
    assert all(torch.isfinite(value).all() for value in first.actor_gradients)

    before = [parameter.detach().clone() for parameter in meta.actor.parameters()]
    optimizer = FirstOrderMetaOptimizer(meta, meta_learning_rate=1.0e-4)
    metrics = optimizer.step([first, second])
    assert metrics.task_count == 2
    assert np.isfinite(metrics.actor_gradient_norm)
    assert any(
        not torch.equal(parameter, expected)
        for parameter, expected in zip(meta.actor.parameters(), before)
    )


def test_differentiable_fast_step_matches_actual_sgd_and_has_meta_gradients() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    meta = create_continuous_trainer(
        config,
        torch.device("cpu"),
        include_expert_identity=True,
        minibatch_size=32,
        entropy_coefficient=1.0e-4,
    )
    states = torch.randn(12, 33)
    local = states[:, :30].reshape(12, 5, 6)
    local[..., 3] = 0.2
    with torch.no_grad():
        actions, old_log_probabilities, _ = meta.actor.act(states)
        values = meta.critic(states)
    batch = PPOBatch(
        states=states,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        returns=values + torch.linspace(0.2, 1.0, 12),
        advantages=torch.linspace(-1.0, 1.0, 12),
    )
    result = differentiable_fast_adaptation(
        meta,
        batch,
        inner_learning_rate=0.2,
    )
    assert result.adapted_offsets.requires_grad

    actual = clone_task_trainer(
        meta,
        inner_learning_rate=0.2,
        inner_update_epochs=1,
        inner_optimizer="sgd",
    )
    actual.actor_optimizer = torch.optim.SGD(
        actual.actor.fast_adaptation_parameters(),
        lr=0.2,
    )
    actual.update_actor_only(batch, np.random.default_rng(18))
    torch.testing.assert_close(
        result.adapted_offsets.detach(),
        actual.actor.expert_mean_offsets.detach(),
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    query = compute_second_order_fast_query_gradients(
        meta,
        result.adapted_offsets,
        batch,
    )
    assert len(query.actor_gradients) == len(tuple(meta.actor.parameters()))
    assert all(torch.isfinite(value).all() for value in query.actor_gradients)


def test_shared_fast_step_moves_all_expert_offsets_equally() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    meta = create_continuous_trainer(
        config,
        torch.device("cpu"),
        include_expert_identity=True,
        minibatch_size=32,
        entropy_coefficient=1.0e-4,
    )
    batch = _synthetic_batch(meta)
    before = meta.actor.expert_mean_offsets.detach().clone()
    result = differentiable_fast_adaptation(
        meta,
        batch,
        inner_learning_rate=0.05,
        shared_offset=True,
    )
    changes = result.adapted_offsets.detach() - before
    torch.testing.assert_close(changes, changes[0].expand_as(changes))


def test_positive_response_residual_lowers_shared_multiplier_offset() -> None:
    torch.manual_seed(34)
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(
        config,
        torch.device("cpu"),
        include_expert_identity=True,
        minibatch_size=32,
        entropy_coefficient=1.0e-4,
    )
    before = trainer.actor.expert_mean_offsets.detach().clone()
    result = differentiable_fast_adaptation(
        trainer,
        _synthetic_batch(trainer),
        inner_learning_rate=0.01,
        shared_offset=True,
        shared_auxiliary_signal=0.08,
        shared_auxiliary_coefficient=10.0,
        shared_policy_gradient_coefficient=0.0,
    )
    changes = result.adapted_offsets.detach() - before
    assert torch.all(changes < 0.0)
    torch.testing.assert_close(changes, changes[0].expand_as(changes))
    assert result.shared_auxiliary_gradient == pytest.approx(0.8)
    assert result.adapted_offsets.requires_grad


def test_gradient_aggregation_and_meta_task_sampling() -> None:
    first = (torch.tensor([1.0, 3.0]), torch.tensor(2.0))
    second = (torch.tensor([3.0, 5.0]), torch.tensor(4.0))
    averaged = aggregate_fomaml_gradients([first, second])
    torch.testing.assert_close(averaged[0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(averaged[1], torch.tensor(3.0))

    cosines = gradient_pairwise_cosines(
        [
            (torch.tensor([1.0, 0.0]),),
            (torch.tensor([1.0, 0.0]),),
            (torch.tensor([-1.0, 0.0]),),
        ]
    )
    assert cosines["mean"] == pytest.approx(-1.0 / 3.0)
    assert cosines["min"] == -1.0
    assert cosines["max"] == 1.0

    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    task = sample_meta_task(config, np.random.default_rng(9))
    assert len(task) == 5
    assert set(task).issubset({"flexible", "normal", "stubborn"})


def test_fast_only_shared_meta_update_keeps_actor_network_fixed() -> None:
    torch.manual_seed(35)
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(
        config,
        torch.device("cpu"),
        include_expert_identity=True,
        minibatch_size=32,
    )
    query = compute_query_gradients(trainer, _synthetic_batch(trainer))
    network_before = [
        parameter.detach().clone() for parameter in trainer.actor.network.parameters()
    ]
    offsets_before = trainer.actor.expert_mean_offsets.detach().clone()
    optimizer = FirstOrderMetaOptimizer(
        trainer,
        meta_learning_rate=1.0e-3,
        actor_fast_only=True,
        shared_actor_fast_update=True,
    )
    optimizer.step([query])
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            network_before,
            trainer.actor.network.parameters(),
            strict=True,
        )
    )
    changes = trainer.actor.expert_mean_offsets.detach() - offsets_before
    torch.testing.assert_close(changes, changes[0].expand_as(changes))
