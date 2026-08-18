"""扫描无弹性矫正时的支持集PPO内循环强度。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_group_receptiveness_maml import _trainer
from src.common.config import load_config
from src.experiments.response_elasticity_maml import (
    evaluate_response_elasticity_adaptation,
)
from src.experiments.response_elasticity_task import (
    make_response_elasticity_ood_task_split,
)


METRICS = ("success_rate", "mean_rounds", "mean_total_reward")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _saved_config(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("config_*.yaml"))
    if len(candidates) != 1:
        raise ValueError(f"{run_dir} 中应当恰好有一个 config_*.yaml。")
    return candidates[0]


def main() -> None:
    args = parse_arguments()
    rows: list[dict[str, Any]] = []
    for run_dir in (path.resolve() for path in args.run_dirs):
        arguments = _read_json(run_dir / "arguments.json")
        config = load_config(_saved_config(run_dir))
        split = make_response_elasticity_ood_task_split(
            split_seed=int(arguments["task_split_seed"]),
            range_profile=str(arguments["elasticity_range_profile"]),
        )
        for coefficient in args.coefficients:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = _trainer(
                config,
                device,
                actor_initialization=str(arguments["meta_actor_initialization"]),
                residual_head_gain=float(arguments["residual_head_gain"]),
                preferred_multiplier=float(arguments["direct_initial_recommendation"]),
            )
            trainer.load_checkpoint(run_dir / "best_elasticity_maml.pt")
            result = evaluate_response_elasticity_adaptation(
                trainer,
                config,
                split.test,
                inner_steps=1,
                support_episodes=int(arguments["support_episodes"]),
                query_episodes=int(arguments["test_query_episodes"]),
                inner_learning_rate=float(arguments["inner_learning_rate"]),
                evaluation_seed=int(arguments["test_case_seed"]),
                calibration_coefficient=0.0,
                policy_gradient_coefficient=float(coefficient),
            )
            row: dict[str, Any] = {
                "seed": int(arguments["seed"]),
                "policy_gradient_coefficient": float(coefficient),
                "effective_policy_step": (
                    float(arguments["inner_learning_rate"]) * float(coefficient)
                ),
            }
            for metric in METRICS:
                row[f"zero_{metric}"] = float(result["zero_step"][metric])
                row[f"adapted_{metric}"] = float(result["adapted"][metric])
                row[f"gain_{metric}"] = float(result["adaptation_gain"][metric])
            rows.append(row)

    aggregate: list[dict[str, Any]] = []
    for coefficient in args.coefficients:
        selected = [
            row
            for row in rows
            if np.isclose(row["policy_gradient_coefficient"], coefficient)
        ]
        record: dict[str, Any] = {
            "policy_gradient_coefficient": float(coefficient),
            "effective_policy_step": float(selected[0]["effective_policy_step"]),
            "seed_count": len(selected),
        }
        for key in selected[0]:
            if key in {
                "seed",
                "policy_gradient_coefficient",
                "effective_policy_step",
            }:
                continue
            values = np.asarray([float(row[key]) for row in selected])
            record[f"mean_{key}"] = float(values.mean())
            record[f"sample_std_{key}"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        aggregate.append(record)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ppo_only_step_sweep.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"per_seed": rows, "aggregate": aggregate},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "ppo_only_step_sweep.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(
        json.dumps(
            [
                {
                    "policy_gradient_coefficient": row[
                        "policy_gradient_coefficient"
                    ],
                    "effective_policy_step": row["effective_policy_step"],
                    "adapted_success_rate": row["mean_adapted_success_rate"],
                    "adapted_mean_rounds": row["mean_adapted_mean_rounds"],
                    "adapted_mean_total_reward": row[
                        "mean_adapted_mean_total_reward"
                    ],
                    "gain_mean_total_reward": row[
                        "mean_gain_mean_total_reward"
                    ],
                }
                for row in aggregate
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"结果已写入：{output_dir}")


if __name__ == "__main__":
    main()
