from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.agents.maml_ppo import FirstOrderMetaOptimizer
from src.common.config import load_config
from src.experiments.continuous_ppo import create_continuous_trainer
from src.experiments.response_function_maml import (
    ResponseFunctionTask,
    config_for_response_function_task,
    evaluate_response_function_adaptation,
    make_response_function_ood_task_split,
    make_response_function_task_split,
    train_response_function_meta_iteration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_response_function_task_does_not_mutate_frozen_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    original = config["response"]["response_table"]["normal"][1]
    task_config = config_for_response_function_task(
        config,
        ResponseFunctionTask(0.10),
    )
    assert config["response"]["response_table"]["normal"][1] == original
    assert task_config["response"]["response_table"]["normal"][1] == pytest.approx(
        original + 0.10
    )


def test_response_function_task_split_is_disjoint_and_reproducible() -> None:
    first = make_response_function_task_split(split_seed=71)
    second = make_response_function_task_split(split_seed=71)
    assert first == second
    assert len(first.train) == 9
    assert len(first.validation) == 3
    assert len(first.test) == 3
    assert not set(first.train).intersection(first.validation)
    assert not set(first.train).intersection(first.test)
    assert not set(first.validation).intersection(first.test)
    for heldout in (first.validation, first.test):
        shifts = [task.receptiveness_shift for task in heldout]
        assert min(shifts) < -0.03
        assert max(shifts) > 0.03


def test_response_function_ood_split_is_outside_training_range() -> None:
    split = make_response_function_ood_task_split(split_seed=72)
    assert split.split_strategy == "range_ood"
    assert split.train_range == (-0.10, 0.10)
    train = {task.receptiveness_shift for task in split.train}
    validation = {task.receptiveness_shift for task in split.validation}
    test = {task.receptiveness_shift for task in split.test}
    assert train == {-0.10, -0.067, -0.033, 0.0, 0.033, 0.067, 0.10}
    assert validation == {-0.05, 0.05}
    assert test == {-0.20, -0.15, 0.15, 0.20}
    assert all(-0.10 <= value <= 0.10 for value in validation)
    assert all(value < -0.10 or value > 0.10 for value in test)


def test_response_function_zero_step_matches_initialization() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(config, torch.device("cpu"), minibatch_size=16)
    result = evaluate_response_function_adaptation(
        trainer,
        config,
        (ResponseFunctionTask(-0.05),),
        inner_steps=0,
        support_episodes=1,
        query_episodes=1,
        inner_learning_rate=1.0e-4,
        evaluation_seed=72,
    )
    assert result["zero_step"] == result["adapted"]
    assert result["adaptation_gain"]["mean_total_reward"] == 0.0
    assert result["adaptation_gain"]["mean_first_step_reward"] == 0.0


def test_response_function_meta_iteration_updates_actor() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(config, torch.device("cpu"), minibatch_size=16)
    optimizer = FirstOrderMetaOptimizer(trainer, meta_learning_rate=1.0e-4)
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    result = train_response_function_meta_iteration(
        trainer,
        optimizer,
        config,
        (ResponseFunctionTask(0.05),),
        support_episodes=1,
        query_episodes=1,
        inner_steps=1,
        inner_learning_rate=1.0e-4,
        iteration_seed=73,
    )
    assert result["environment_episodes"] == 2
    assert result["tasks"][0]["task"]["receptiveness_shift"] == pytest.approx(0.05)
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, trainer.actor.parameters(), strict=True)
    )
