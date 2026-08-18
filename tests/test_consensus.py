from __future__ import annotations

import numpy as np

from src.model.consensus import (
    element_consensus,
    evaluate_consensus,
    group_reference,
    identify_disagreement,
    overall_consensus,
    pairwise_similarity,
)


def test_similarity_is_symmetric_and_self_similarity_is_one() -> None:
    opinions = np.array([[0.1, 0.7], [0.4, 0.2], [0.9, 0.5]])
    similarity = pairwise_similarity(opinions)
    np.testing.assert_allclose(similarity, similarity.swapaxes(0, 1))
    np.testing.assert_allclose(np.diagonal(similarity, axis1=0, axis2=1), 1.0)
    assert np.all((similarity >= 0.0) & (similarity <= 1.0))


def test_identical_opinions_have_full_consensus() -> None:
    opinions = np.full((5, 5), 0.6)
    metrics = evaluate_consensus(opinions, threshold=0.85)
    np.testing.assert_allclose(metrics.ace, 1.0)
    np.testing.assert_allclose(metrics.acd, 1.0)
    assert metrics.success


def test_ace_acd_and_threshold_boundary_match_hand_calculation() -> None:
    opinions = np.array([[0.0], [0.5], [1.0]])
    similarity = pairwise_similarity(opinions)
    ace = element_consensus(opinions, similarity)
    acd = overall_consensus(ace)
    np.testing.assert_allclose(ace[:, 0], [0.25, 0.5, 0.25])
    np.testing.assert_allclose(acd, [0.25, 0.5, 0.25])

    expert_mask, issue_mask = identify_disagreement(ace, acd, threshold=0.5)
    np.testing.assert_array_equal(expert_mask, [True, False, True])
    np.testing.assert_array_equal(issue_mask[:, 0], [True, False, True])


def test_group_reference_is_column_mean() -> None:
    opinions = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])
    np.testing.assert_allclose(group_reference(opinions), [0.4, 0.6])

