from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.baselines import FixedMultiplierPolicy
from src.common.config import load_config
from src.experiments.train_ppo import (
    collect_rollout,
    create_trainer,
    evaluate_policy_on_cases,
    evaluate_trainer,
    make_validation_cases,
    validation_selection_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_real_environment_rollout_update_and_fixed_validation_are_finite() -> None:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml"))
    # 测试只缩小 PPO 更新计算量，不修改环境。
    config["ppo"]["update_epochs"] = 1
    trainer = create_trainer(config, torch.device("cpu"), minibatch_size=16)
    buffer, summary = collect_rollout(
        trainer,
        config,
        16,
        task_rng=np.random.default_rng(1),
        type_rng=np.random.default_rng(2),
        response_seed_rng=np.random.default_rng(3),
    )
    assert len(buffer) >= 16
    assert summary["episodes"] > 0
    batch = buffer.to_batch(torch.device("cpu"), gamma=0.99, gae_lambda=0.95)
    metrics = trainer.update(batch, np.random.default_rng(4))
    assert np.isfinite(metrics.actor_loss)

    cases = make_validation_cases(
        config,
        3,
        task_seed=5,
        type_seed=6,
        response_seed=7,
    )
    first, _ = evaluate_trainer(trainer, config, cases, deterministic=True)
    second, _ = evaluate_trainer(trainer, config, cases, deterministic=True)
    assert first == second

    torch.manual_seed(1234)
    expected_after_evaluation = torch.rand(4)
    torch.manual_seed(1234)
    stochastic_first, _ = evaluate_trainer(
        trainer,
        config,
        cases,
        deterministic=False,
        action_seed=9988,
    )
    actual_after_evaluation = torch.rand(4)
    stochastic_second, _ = evaluate_trainer(
        trainer,
        config,
        cases,
        deterministic=False,
        action_seed=9988,
    )
    assert stochastic_first == stochastic_second
    torch.testing.assert_close(actual_after_evaluation, expected_after_evaluation)

    baseline, episodes = evaluate_policy_on_cases(
        config,
        cases,
        lambda _: FixedMultiplierPolicy(2),
    )
    assert baseline["episode_count"] == 3
    assert len(episodes) == 3
    assert baseline["optimizer_failure_rate"] == 0.0


def test_validation_selection_score_uses_deterministic_reward() -> None:
    assert validation_selection_score({"mean_total_reward": 1.25}) == 1.25
    with pytest.raises(FloatingPointError, match="有限值"):
        validation_selection_score({"mean_total_reward": float("nan")})


def test_rollout_accepts_one_fixed_response_composition() -> None:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml"))
    config["ppo"]["update_epochs"] = 1
    trainer = create_trainer(config, torch.device("cpu"), minibatch_size=16)
    buffer, summary = collect_rollout(
        trainer,
        config,
        None,
        task_rng=np.random.default_rng(21),
        type_rng=np.random.default_rng(22),
        response_seed_rng=np.random.default_rng(23),
        episode_target=2,
        fixed_response_composition=(2, 1, 2),
    )
    assert len(buffer) > 0
    assert summary["episodes"] == 2
    with pytest.raises(ValueError):
        collect_rollout(
            trainer,
            config,
            None,
            task_rng=np.random.default_rng(21),
            type_rng=np.random.default_rng(22),
            response_seed_rng=np.random.default_rng(23),
            episode_target=1,
            fixed_response_types=("normal",) * 5,
            fixed_response_composition=(2, 1, 2),
        )
