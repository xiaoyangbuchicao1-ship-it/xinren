"""统一随机种子管理。"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedBundle:
    """从一个主种子派生实验各部分使用的独立种子。"""

    master: int
    python: int
    numpy: int
    torch: int
    task: int
    support: int
    query: int
    response: int
    evaluation: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def derive_seed_bundle(master_seed: int) -> SeedBundle:
    """使用 SeedSequence 稳定派生子种子，避免各随机流互相干扰。"""

    sequence = np.random.SeedSequence(int(master_seed))
    children = sequence.spawn(8)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return SeedBundle(
        master=int(master_seed),
        python=values[0],
        numpy=values[1],
        torch=values[2],
        task=values[3],
        support=values[4],
        query=values[5],
        response=values[6],
        evaluation=values[7],
    )


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """设置 Python、NumPy 和 PyTorch 全局种子。"""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_numpy_rng(seed: int) -> np.random.Generator:
    """返回局部 NumPy 生成器，业务代码优先使用局部生成器。"""

    return np.random.default_rng(int(seed))

