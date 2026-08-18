from __future__ import annotations

import numpy as np
import pytest

from src.env.response_model import (
    action_to_multiplier,
    effective_adjustment,
    sample_response_rate,
    sample_response_types,
    sample_response_types_from_counts,
    suggestion_bin,
)


RESPONSE_TABLE = {
    "flexible": [0.95, 0.85, 0.75],
    "normal": [0.80, 0.60, 0.40],
    "stubborn": [0.50, 0.25, 0.10],
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0), (0.2999, 0), (0.3, 1), (0.6999, 1), (0.7, 2), (1.0, 2)],
)
def test_suggestion_bin_boundaries(value: float, expected: int) -> None:
    assert suggestion_bin(value) == expected


def test_response_table_means_and_effective_adjustment() -> None:
    rng = np.random.default_rng(7)
    assert sample_response_rate("flexible", 0.1, RESPONSE_TABLE, 0.0, rng) == 0.95
    assert sample_response_rate("normal", 0.4, RESPONSE_TABLE, 0.0, rng) == 0.60
    assert sample_response_rate("stubborn", 0.8, RESPONSE_TABLE, 0.0, rng) == 0.10
    np.testing.assert_allclose(
        effective_adjustment([0.2, 0.6], [0.5, 0.25]),
        [0.1, 0.15],
    )


def test_linear_response_interpolates_between_suggestion_bin_centers() -> None:
    rng = np.random.default_rng(8)
    assert sample_response_rate(
        "normal",
        0.15,
        RESPONSE_TABLE,
        0.0,
        rng,
        interpolation="linear",
    ) == pytest.approx(0.80)
    assert sample_response_rate(
        "normal",
        0.325,
        RESPONSE_TABLE,
        0.0,
        rng,
        interpolation="linear",
    ) == pytest.approx(0.70)
    assert sample_response_rate(
        "normal",
        0.50,
        RESPONSE_TABLE,
        0.0,
        rng,
        interpolation="linear",
    ) == pytest.approx(0.60)
    with pytest.raises(ValueError, match="插值模式"):
        sample_response_rate(
            "normal",
            0.5,
            RESPONSE_TABLE,
            0.0,
            rng,
            interpolation="unknown",
        )


def test_response_sampling_is_reproducible_and_clipped() -> None:
    first_rng = np.random.default_rng(2026)
    second_rng = np.random.default_rng(2026)
    first = [
        sample_response_rate("normal", 0.4, RESPONSE_TABLE, 0.5, first_rng)
        for _ in range(20)
    ]
    second = [
        sample_response_rate("normal", 0.4, RESPONSE_TABLE, 0.5, second_rng)
        for _ in range(20)
    ]
    np.testing.assert_allclose(first, second)
    assert np.all((np.asarray(first) >= 0.0) & (np.asarray(first) <= 1.0))


def test_type_sampling_and_action_mapping_are_reproducible() -> None:
    names = ["flexible", "normal", "stubborn"]
    probabilities = [0.4, 0.4, 0.2]
    first = sample_response_types(50, names, probabilities, np.random.default_rng(11))
    second = sample_response_types(50, names, probabilities, np.random.default_rng(11))
    assert first == second
    assert set(first).issubset(names)
    np.testing.assert_allclose(
        action_to_multiplier([0, 1, 2, 3], [0.5, 0.75, 1.0, 1.25]),
        [0.5, 0.75, 1.0, 1.25],
    )


def test_type_counts_are_preserved_while_positions_are_shuffled() -> None:
    names = ["flexible", "normal", "stubborn"]
    first = sample_response_types_from_counts(
        names,
        [2, 1, 2],
        np.random.default_rng(31),
    )
    second = sample_response_types_from_counts(
        names,
        [2, 1, 2],
        np.random.default_rng(31),
    )
    assert first == second
    assert first.count("flexible") == 2
    assert first.count("normal") == 1
    assert first.count("stubborn") == 2
    with pytest.raises(ValueError):
        sample_response_types_from_counts(names, [2, 1, 1.5], np.random.default_rng(1))


def test_invalid_suggestion_and_action_are_rejected() -> None:
    with pytest.raises(ValueError):
        suggestion_bin(1.01)
    with pytest.raises(ValueError):
        action_to_multiplier([4], [0.5, 0.75, 1.0, 1.25])
