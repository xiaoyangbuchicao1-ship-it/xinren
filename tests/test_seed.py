from __future__ import annotations

import random

import numpy as np
import torch

from src.common.seed import derive_seed_bundle, set_global_seed


def _sample_triplet(seed: int) -> tuple[float, float, float]:
    set_global_seed(seed)
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_seed_bundle_is_stable_and_independent() -> None:
    first = derive_seed_bundle(2026)
    second = derive_seed_bundle(2026)
    other = derive_seed_bundle(2027)
    assert first == second
    assert first != other
    assert len(set(first.to_dict().values())) == len(first.to_dict())


def test_global_seed_reproduces_all_libraries() -> None:
    assert _sample_triplet(1234) == _sample_triplet(1234)
    assert _sample_triplet(1234) != _sample_triplet(1235)

