from __future__ import annotations

import numpy as np

from src.model.fusion import compute_fusion_weights, fuse_opinions
from src.model.trust import (
    compute_ai_to_human_information_trust,
    compute_human_to_ai_trust,
)


def test_human_to_ai_trust_matches_formula() -> None:
    trust = compute_human_to_ai_trust([0.8, 0.1], [0.2, 0.9])
    np.testing.assert_allclose(trust, [0.8, 0.1])


def test_information_trust_handles_positive_negative_and_constant_vectors() -> None:
    human = np.array(
        [
            [0.1, 0.2, 0.4, 0.7, 0.9],
            [0.1, 0.2, 0.4, 0.7, 0.9],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]
    )
    ai = np.array(
        [
            [0.1, 0.2, 0.4, 0.7, 0.9],
            [0.9, 0.7, 0.4, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ]
    )
    information_trust = compute_ai_to_human_information_trust(human, ai)
    np.testing.assert_allclose(information_trust, [1.0, 0.0, 0.0], atol=1.0e-12)
    assert np.all(np.isfinite(information_trust))


def test_cross_weights_and_fusion_match_hand_calculation() -> None:
    human_to_ai = np.array([0.8])
    ai_to_human = np.array([0.2])
    human_weights, ai_weights = compute_fusion_weights(
        human_to_ai,
        ai_to_human,
        epsilon=1.0e-6,
    )
    np.testing.assert_allclose(human_weights + ai_weights, [1.0])
    assert ai_weights[0] > human_weights[0]

    human = np.array([[0.2, 0.7]])
    ai = np.array([[0.8, 0.5]])
    fused = fuse_opinions(human, ai, human_weights, ai_weights)
    expected = human_weights[:, None] * human + ai_weights[:, None] * ai
    np.testing.assert_allclose(fused, expected)
    assert np.all(fused >= np.minimum(human, ai))
    assert np.all(fused <= np.maximum(human, ai))

