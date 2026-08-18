from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.maml_ppo import FirstOrderMetaOptimizer
from src.common.config import load_config
from src.experiments.continuous_ppo import create_continuous_trainer
from src.experiments.group_receptiveness_maml import (
    adapt_continuous_to_group_receptiveness,
    estimate_receptiveness_residual,
    make_group_receptiveness_task_split,
    train_group_receptiveness_meta_iteration,
)
from src.experiments.response_function_maml import ResponseFunctionTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _trainer(config: dict[str, object]):
    return create_continuous_trainer(
        config,
        torch.device("cpu"),
        include_expert_identity=True,
        minibatch_size=16,
        entropy_coefficient=1.0e-4,
    )


def test_group_receptiveness_split_is_reproducible() -> None:
    first = make_group_receptiveness_task_split(split_seed=91)
    second = make_group_receptiveness_task_split(split_seed=91)
    assert first == second
    assert len(first.train) == 9
    assert len(first.validation) == 3
    assert len(first.test) == 3
    assert not set(first.train).intersection(first.test)


def test_response_residual_uses_known_type_mixture_and_suggestion_bins() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    probabilities = np.asarray(config["response"]["type_probabilities"])
    small_responses = np.asarray(
        [
            config["response"]["response_table"][name][0]
            for name in config["response"]["type_names"]
        ]
    )
    expected = float(probabilities @ small_responses)
    residual, baseline = estimate_receptiveness_residual(
        {
            "suggestion_bin_counts": [20, 0, 0],
            "active_response_rate_mean": expected + 0.05,
        },
        config,
    )
    assert baseline == pytest.approx(expected)
    assert residual == pytest.approx(0.05)


def test_group_adaptation_moves_five_offsets_by_one_shared_amount() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = _trainer(config)
    before = trainer.actor.expert_mean_offsets.detach().clone()
    adapted, metrics = adapt_continuous_to_group_receptiveness(
        trainer,
        config,
        ResponseFunctionTask(0.10),
        inner_steps=1,
        support_episodes=2,
        inner_learning_rate=0.02,
        support_seed=92,
    )
    changes = adapted.actor.expert_mean_offsets.detach() - before
    torch.testing.assert_close(changes, changes[0].expand_as(changes))
    torch.testing.assert_close(trainer.actor.expert_mean_offsets.detach(), before)
    assert metrics.shared_fast_offset is True
    assert np.isclose(metrics.shared_offset_change, float(changes.mean().item()))


def test_group_meta_iteration_updates_actor_and_pairs_scenarios() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = _trainer(config)
    optimizer = FirstOrderMetaOptimizer(trainer, meta_learning_rate=1.0e-4)
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    result = train_group_receptiveness_meta_iteration(
        trainer,
        optimizer,
        config,
        (ResponseFunctionTask(-0.10), ResponseFunctionTask(0.10)),
        support_episodes=1,
        query_episodes=1,
        inner_learning_rate=0.01,
        iteration_seed=93,
        paired_scenarios=True,
        outer_update_epochs=2,
    )
    assert result["environment_episodes"] == 4
    assert result["tasks"][0]["scenario_seeds"] == result["tasks"][1][
        "scenario_seeds"
    ]
    assert result["meta_update"]["epochs_completed"] == 2
    assert len(result["meta_update_epochs"]) == 2
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, trainer.actor.parameters(), strict=True)
    )
