from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.common.config import config_hash, freeze_config, load_config, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_base_config_loads_and_hash_is_stable(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    assert config_hash(config) == config_hash(deepcopy(config))
    path, digest = freeze_config(config, tmp_path)
    frozen = load_config(path)
    assert frozen["experiment"]["config_hash"] == digest


def test_invalid_suggestion_bins_are_rejected() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    config["response"]["suggestion_bins"] = [0.0, 0.2, 0.4, 1.0]
    with pytest.raises(ValueError, match="0.3/0.4/0.3"):
        validate_config(config)


def test_invalid_planning_margin_is_rejected() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    config["consensus"]["threshold"] = 0.95
    config["consensus"]["planning_margin"] = 0.06
    with pytest.raises(ValueError, match="规划余量"):
        validate_config(config)
