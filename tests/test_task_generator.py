from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.common.seed import make_numpy_rng
from src.data.task_generator import generate_stage_b_instance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_stage_b_instance_shape_range_and_reproducibility() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    first = generate_stage_b_instance(config, make_numpy_rng(12345))
    second = generate_stage_b_instance(config, make_numpy_rng(12345))

    assert first.task.reference.shape == (5,)
    assert first.task.human_opinions.shape == (5, 5)
    assert first.task.ai_opinions.shape == (5, 5)
    assert first.initial_fused_opinions.shape == (5, 5)
    for values in [
        first.task.reference,
        first.task.human_opinions,
        first.task.ai_opinions,
        first.initial_fused_opinions,
    ]:
        assert np.all((values >= 0.0) & (values <= 1.0))

    np.testing.assert_array_equal(first.task.human_opinions, second.task.human_opinions)
    np.testing.assert_array_equal(first.task.ai_opinions, second.task.ai_opinions)
    np.testing.assert_array_equal(first.initial_fused_opinions, second.initial_fused_opinions)


def test_five_ai_recommendations_are_not_identical() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    instance = generate_stage_b_instance(config, make_numpy_rng(2026))
    assert np.unique(instance.task.ai_opinions, axis=0).shape[0] == 5


def test_fixed_stage_b_arrays_are_read_only() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    instance = generate_stage_b_instance(config, make_numpy_rng(2026))
    with pytest.raises(ValueError):
        instance.task.human_opinions[0, 0] = 0.0
    with pytest.raises(ValueError):
        instance.task.ai_opinions[0, 0] = 0.0
    with pytest.raises(ValueError):
        instance.human_to_ai_trust[0] = 0.0

