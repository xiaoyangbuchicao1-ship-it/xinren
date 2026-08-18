"""在相同元初始化和相同OOD案例上消融响应弹性梯度矫正项。"""

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
from src.common.encoding import write_json
from src.experiments.response_elasticity_maml import (
    evaluate_response_elasticity_adaptation,
)
from src.experiments.response_elasticity_task import (
    make_response_elasticity_ood_task_split,
    make_response_elasticity_task_split,
)


T_95_DF_2 = 4.302652729696142
METRICS = (
    "success_rate",
    "mean_rounds",
    "mean_total_reward",
    "mean_total_modification",
    "mean_final_min_acd",
    "active_multiplier_mean",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-episodes", type=int, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _saved_config(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("config_*.yaml"))
    if len(candidates) != 1:
        raise ValueError(f"{run_dir} 中应当恰好有一个 config_*.yaml。")
    return candidates[0]


def _evaluate_mode(
    run_dir: Path,
    *,
    calibration_coefficient: float,
    query_episodes_override: int | None,
) -> dict[str, Any]:
    arguments = _read_json(run_dir / "arguments.json")
    config = load_config(_saved_config(run_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = _trainer(
        config,
        device,
        actor_initialization=str(arguments["meta_actor_initialization"]),
        residual_head_gain=float(arguments["residual_head_gain"]),
        preferred_multiplier=float(arguments["direct_initial_recommendation"]),
    )
    trainer.load_checkpoint(run_dir / "best_elasticity_maml.pt")
    split = (
        make_response_elasticity_ood_task_split(
            split_seed=int(arguments["task_split_seed"]),
            range_profile=str(arguments["elasticity_range_profile"]),
        )
        if arguments["task_split_mode"] == "range_ood"
        else make_response_elasticity_task_split(
            split_seed=int(arguments["task_split_seed"])
        )
    )
    query_episodes = (
        int(query_episodes_override)
        if query_episodes_override is not None
        else int(arguments["test_query_episodes"])
    )
    result = evaluate_response_elasticity_adaptation(
        trainer,
        config,
        split.test,
        inner_steps=1,
        support_episodes=int(arguments["support_episodes"]),
        query_episodes=query_episodes,
        inner_learning_rate=float(arguments["inner_learning_rate"]),
        evaluation_seed=int(arguments["test_case_seed"]),
        calibration_coefficient=float(calibration_coefficient),
        policy_gradient_coefficient=float(arguments["policy_gradient_coefficient"]),
    )
    return {
        "seed": int(arguments["seed"]),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str((run_dir / "best_elasticity_maml.pt").resolve()),
        "mode": "with_correction" if calibration_coefficient else "ppo_only",
        "calibration_coefficient": float(calibration_coefficient),
        "policy_gradient_coefficient": float(
            arguments["policy_gradient_coefficient"]
        ),
        "support_episodes": int(arguments["support_episodes"]),
        "query_episodes_per_task": query_episodes,
        "evaluation_seed": int(arguments["test_case_seed"]),
        "evaluation": result,
    }


def _mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    half_width = (
        float(T_95_DF_2 * sample_std / np.sqrt(array.size))
        if array.size == 3
        else 0.0
    )
    return {
        "mean": mean,
        "sample_std": sample_std,
        "ci95_half_width": half_width,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def main() -> None:
    args = parse_arguments()
    run_dirs = [path.resolve() for path in args.run_dirs]
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for coefficient in (0.0, 1.0):
            records.append(
                _evaluate_mode(
                    run_dir,
                    calibration_coefficient=coefficient,
                    query_episodes_override=args.query_episodes,
                )
            )

    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), {})[str(record["mode"])] = record
    if any(set(modes) != {"ppo_only", "with_correction"} for modes in by_seed.values()):
        raise RuntimeError("每个种子都必须同时包含关闭和开启矫正的结果。")

    per_seed_rows: list[dict[str, Any]] = []
    for seed, modes in sorted(by_seed.items()):
        for mode_name, record in sorted(modes.items()):
            adapted = record["evaluation"]["adapted"]
            row = {"seed": seed, "mode": mode_name}
            row.update({metric: float(adapted[metric]) for metric in METRICS})
            per_seed_rows.append(row)

    deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    delta_rows: list[dict[str, Any]] = []
    for seed, modes in sorted(by_seed.items()):
        ppo = modes["ppo_only"]["evaluation"]["adapted"]
        corrected = modes["with_correction"]["evaluation"]["adapted"]
        row = {"seed": seed}
        for metric in METRICS:
            delta = float(corrected[metric]) - float(ppo[metric])
            row[f"delta_{metric}"] = delta
            deltas[metric].append(delta)
        delta_rows.append(row)

    summary = {
        "design": (
            "Paired deployment-time ablation: the same meta-trained checkpoint, "
            "support/query seeds, and OOD tasks are evaluated with lambda_cal=0 "
            "and lambda_cal=1. This isolates the one-step correction at inference "
            "but is not a full retraining ablation."
        ),
        "seed_count": len(by_seed),
        "records": records,
        "per_seed_adapted": per_seed_rows,
        "paired_delta_with_minus_without": delta_rows,
        "paired_delta_summary": {
            metric: _mean_ci(values) for metric, values in deltas.items()
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, output_dir / "ablation_results.json")
    with (output_dir / "per_seed_adapted.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (output_dir / "paired_deltas.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)
    print(json.dumps(summary["paired_delta_summary"], ensure_ascii=False, indent=2))
    print(f"结果已写入：{output_dir}")


if __name__ == "__main__":
    main()
