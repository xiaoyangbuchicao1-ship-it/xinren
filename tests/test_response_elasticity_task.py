from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.agents.maml_ppo import FirstOrderMetaOptimizer
from src.common.config import load_config
from src.experiments.continuous_ppo import create_continuous_trainer
from src.experiments.response_elasticity_maml import (
    adapt_continuous_to_response_elasticity,
    sample_symmetric_response_elasticity_tasks,
)
from src.experiments.response_elasticity_task import (
    ResponseElasticityTask,
    config_for_response_elasticity_task,
    estimate_response_elasticity_signal,
    make_response_elasticity_ood_task_split,
    make_response_elasticity_task_split,
)
from src.experiments.response_function_maml import (
    train_response_function_meta_iteration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_response_elasticity_split_is_reproducible_and_disjoint() -> None:
    first = make_response_elasticity_task_split(split_seed=81)
    second = make_response_elasticity_task_split(split_seed=81)
    assert first == second
    assert len(first.train) == 9
    assert len(first.validation) == 3
    assert len(first.test) == 3
    assert not set(first.train).intersection(first.test)


def test_positive_sensitivity_steepens_response_curve_without_changing_middle() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    shifted = config_for_response_elasticity_task(
        config,
        ResponseElasticityTask(0.10),
    )
    for response_type in config["response"]["type_names"]:
        base = np.asarray(config["response"]["response_table"][response_type])
        task = np.asarray(shifted["response"]["response_table"][response_type])
        assert task[0] == min(1.0, base[0] + 0.10)
        assert task[1] == base[1]
        assert task[2] == max(0.0, base[2] - 0.10)
    assert shifted is not config


def test_elasticity_ood_split_is_outside_training_range() -> None:
    split = make_response_elasticity_ood_task_split(split_seed=82)
    assert split.split_strategy == "range_ood_moderate"
    assert split.train_range == (-0.20, 0.20)
    assert {task.magnitude_sensitivity for task in split.validation} == {
        -0.17,
        -0.10,
        0.10,
        0.17,
    }
    assert {task.magnitude_sensitivity for task in split.test} == {
        -0.25,
        -0.225,
        0.225,
        0.25,
    }


def test_wide_elasticity_split_preserves_one_dimension_and_mild_ood() -> None:
    split = make_response_elasticity_ood_task_split(
        split_seed=82,
        range_profile="wide",
    )
    assert split.split_strategy == "range_ood_wide"
    assert split.train_range == (-0.30, 0.30)
    assert {task.magnitude_sensitivity for task in split.validation} == {
        -0.25,
        -0.15,
        0.15,
        0.25,
    }
    assert {task.magnitude_sensitivity for task in split.test} == {
        -0.35,
        -0.325,
        0.325,
        0.35,
    }


def test_symmetric_elasticity_sampler_returns_balanced_pairs() -> None:
    split = make_response_elasticity_ood_task_split(range_profile="wide")
    first = sample_symmetric_response_elasticity_tasks(
        split.train,
        4,
        np.random.default_rng(8201),
    )
    second = sample_symmetric_response_elasticity_tasks(
        split.train,
        4,
        np.random.default_rng(8201),
    )
    assert first == second
    values = np.asarray([task.magnitude_sensitivity for task in first])
    assert np.isclose(values.sum(), 0.0)
    magnitudes, counts = np.unique(np.abs(values), return_counts=True)
    assert len(magnitudes) == 2
    np.testing.assert_array_equal(counts, np.asarray([2, 2]))


def test_elasticity_signal_recovers_synthetic_slope_shift() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    probabilities = np.asarray(config["response"]["type_probabilities"])
    table = np.asarray(
        [
            config["response"]["response_table"][name]
            for name in config["response"]["type_names"]
        ]
    )
    expected = probabilities @ table
    estimate, baseline_gap = estimate_response_elasticity_signal(
        {
            "suggestion_bin_counts": [20, 5, 20],
            "response_rate_mean_by_bin": [
                expected[0] + 0.08,
                expected[1],
                expected[2] - 0.08,
            ],
        },
        config,
    )
    assert np.isclose(estimate, 0.08)
    assert np.isclose(baseline_gap, expected[0] - expected[2])


def test_elasticity_wrapper_runs_shared_fast_adaptation() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    trainer = create_continuous_trainer(
        config,
        torch.device("cpu"),
        include_expert_identity=True,
        minibatch_size=16,
    )
    adapted, metrics = adapt_continuous_to_response_elasticity(
        trainer,
        config,
        ResponseElasticityTask(0.08),
        inner_steps=1,
        support_episodes=2,
        inner_learning_rate=0.01,
        support_seed=83,
    )
    changes = (
        adapted.actor.expert_mean_offsets.detach()
        - trainer.actor.expert_mean_offsets.detach()
    )
    torch.testing.assert_close(changes, changes[0].expand_as(changes))
    assert np.isfinite(metrics.estimated_receptiveness_residual)


def test_elasticity_task_runs_first_order_meta_iteration() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    config["guidance"] = {"mode": "direct", "action_bounds": [0.01, 0.99]}
    trainer = create_continuous_trainer(
        config,
        torch.device("cpu"),
        minibatch_size=16,
        preferred_multiplier=0.30,
    )
    optimizer = FirstOrderMetaOptimizer(trainer, meta_learning_rate=1.0e-4)
    result = train_response_function_meta_iteration(
        trainer,
        optimizer,
        config,
        (ResponseElasticityTask(0.05),),
        support_episodes=1,
        query_episodes=1,
        inner_steps=1,
        inner_learning_rate=3.0e-4,
        iteration_seed=84,
        task_config_factory=config_for_response_elasticity_task,
    )
    assert result["environment_episodes"] == 2
    assert result["tasks"][0]["task"]["magnitude_sensitivity"] == 0.05
