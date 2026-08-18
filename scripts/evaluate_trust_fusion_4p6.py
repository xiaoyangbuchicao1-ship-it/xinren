"""Paired OOD evaluation of trust-regulated, equal, and human-only fusion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.task_generator import StageBInstance
from src.experiments.continuous_ppo import (
    aggregate_continuous_episodes,
    create_continuous_trainer,
    evaluate_continuous_trainer,
)
from src.experiments.response_elasticity_maml import (
    adapt_continuous_to_response_elasticity,
)
from src.experiments.response_elasticity_task import (
    ResponseElasticityTask,
    config_for_response_elasticity_task,
)
from src.experiments.train_ppo import ValidationCase, make_validation_cases
from src.model.consensus import evaluate_consensus
from src.model.fusion import fuse_opinions
from src.common.config import load_config


MODES = ("bidirectional_trust", "equal_weight", "human_only")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def counterfactual_instance(instance: StageBInstance, mode: str) -> StageBInstance:
    if mode == "bidirectional_trust":
        return instance
    num_experts = instance.task.human_opinions.shape[0]
    if mode == "equal_weight":
        human_weights = np.full(num_experts, 0.5, dtype=np.float64)
        ai_weights = np.full(num_experts, 0.5, dtype=np.float64)
        opinions = fuse_opinions(
            instance.task.human_opinions,
            instance.task.ai_opinions,
            human_weights,
            ai_weights,
        )
    elif mode == "human_only":
        human_weights = np.ones(num_experts, dtype=np.float64)
        ai_weights = np.zeros(num_experts, dtype=np.float64)
        opinions = instance.task.human_opinions
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
    return replace(
        instance,
        human_weights=readonly(human_weights),
        ai_weights=readonly(ai_weights),
        initial_fused_opinions=readonly(opinions),
    )


def transform_cases(cases: list[ValidationCase], mode: str) -> list[ValidationCase]:
    return [
        replace(case, instance=counterfactual_instance(case.instance, mode))
        for case in cases
    ]


def initial_diagnostics(
    cases: list[ValidationCase],
    planning_threshold: float,
) -> dict[str, float]:
    min_acd: list[float] = []
    mean_acd: list[float] = []
    reference_mae: list[float] = []
    initial_success: list[float] = []
    for case in cases:
        opinions = np.asarray(case.instance.initial_fused_opinions, dtype=np.float64)
        metrics = evaluate_consensus(opinions, planning_threshold)
        min_acd.append(float(metrics.min_acd))
        mean_acd.append(float(metrics.mean_acd))
        reference = np.asarray(case.instance.task.reference, dtype=np.float64)
        reference_mae.append(float(np.mean(np.abs(opinions - reference[None, :]))))
        initial_success.append(float(metrics.min_acd >= planning_threshold))
    return {
        "mean_initial_min_acd": float(np.mean(min_acd)),
        "mean_initial_mean_acd": float(np.mean(mean_acd)),
        "mean_initial_reference_mae": float(np.mean(reference_mae)),
        "planning_threshold_initial_success_rate": float(np.mean(initial_success)),
    }


def trainer_for_run(
    config: dict[str, Any],
    run_arguments: dict[str, Any],
    device: torch.device,
):
    return create_continuous_trainer(
        config,
        device,
        learning_rate=3.0e-4,
        entropy_coefficient=1.0e-4,
        minibatch_size=64,
        include_expert_identity=True,
        preferred_multiplier=float(run_arguments["direct_initial_recommendation"]),
        actor_initialization=str(run_arguments["meta_actor_initialization"]),
        residual_head_gain=float(run_arguments["residual_head_gain"]),
    )


def paired_difference(
    full_episodes: list[Any],
    comparison_episodes: list[Any],
) -> dict[str, float]:
    if len(full_episodes) != len(comparison_episodes):
        raise ValueError("Paired episode lists must have equal length.")
    return {
        "success_rate_difference": float(
            np.mean(
                [
                    float(full.success) - float(other.success)
                    for full, other in zip(
                        full_episodes, comparison_episodes, strict=True
                    )
                ]
            )
        ),
        "mean_round_reduction": float(
            np.mean(
                [
                    float(other.rounds) - float(full.rounds)
                    for full, other in zip(
                        full_episodes, comparison_episodes, strict=True
                    )
                ]
            )
        ),
        "mean_reward_difference": float(
            np.mean(
                [
                    float(full.total_reward) - float(other.total_reward)
                    for full, other in zip(
                        full_episodes, comparison_episodes, strict=True
                    )
                ]
            )
        ),
        "mean_modification_difference": float(
            np.mean(
                [
                    float(full.total_modification) - float(other.total_modification)
                    for full, other in zip(
                        full_episodes, comparison_episodes, strict=True
                    )
                ]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    arguments = parser.parse_args()

    run_dir = arguments.run_dir.resolve()
    run_arguments = read_json(run_dir / "arguments.json")
    task_split = read_json(run_dir / "task_split.json")
    heldout = read_json(run_dir / "heldout_comparison.json")
    config_paths = sorted(run_dir.glob("config_*.yaml"))
    if len(config_paths) != 1:
        raise ValueError("The run directory must contain exactly one config file.")
    config = load_config(config_paths[0])

    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    initialization = trainer_for_run(config, run_arguments, device)
    initialization.load_checkpoint(run_dir / "best_elasticity_maml.pt")

    seed_rng = np.random.default_rng(int(run_arguments["test_case_seed"]))
    all_episodes: dict[str, list[Any]] = {mode: [] for mode in MODES}
    all_initial: dict[str, list[dict[str, float]]] = {mode: [] for mode in MODES}
    per_task: list[dict[str, Any]] = []
    planning_threshold = float(config["consensus"]["threshold"]) + float(
        config["consensus"]["planning_margin"]
    )

    for task_record in task_split["test"]:
        task = ResponseElasticityTask(float(task_record["magnitude_sensitivity"]))
        task_config = config_for_response_elasticity_task(config, task)
        task_seed, type_seed, response_seed, support_seed = (
            int(value)
            for value in seed_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=4,
                dtype=np.uint32,
            )
        )
        base_cases = make_validation_cases(
            task_config,
            int(run_arguments["test_query_episodes"]),
            task_seed=task_seed,
            type_seed=type_seed,
            response_seed=response_seed,
        )
        adapted, adaptation = adapt_continuous_to_response_elasticity(
            initialization,
            config,
            task,
            inner_steps=1,
            support_episodes=int(run_arguments["support_episodes"]),
            inner_learning_rate=float(run_arguments["inner_learning_rate"]),
            support_seed=support_seed,
            calibration_coefficient=float(run_arguments["calibration_coefficient"]),
            policy_gradient_coefficient=float(
                run_arguments["policy_gradient_coefficient"]
            ),
        )

        task_output: dict[str, Any] = {
            "task": task.to_serializable(),
            "support_adaptation": adaptation.to_serializable(),
            "fusion_modes": {},
        }
        for mode in MODES:
            cases = transform_cases(base_cases, mode)
            initial = initial_diagnostics(cases, planning_threshold)
            summary, episodes = evaluate_continuous_trainer(
                adapted,
                task_config,
                cases,
                deterministic=True,
            )
            task_output["fusion_modes"][mode] = {
                "initial": initial,
                "consensus": summary,
            }
            all_initial[mode].append(initial)
            all_episodes[mode].extend(episodes)
        per_task.append(task_output)

    aggregate: dict[str, Any] = {}
    for mode in MODES:
        initial_records = all_initial[mode]
        aggregate[mode] = {
            "initial": {
                key: float(np.mean([record[key] for record in initial_records]))
                for key in initial_records[0]
            },
            "consensus": aggregate_continuous_episodes(all_episodes[mode]),
        }

    formal = heldout["maml_initialization"]["adapted"]
    reproduced = aggregate["bidirectional_trust"]["consensus"]
    reproduction_check = {
        key: float(reproduced[key]) - float(formal[key])
        for key in (
            "success_rate",
            "mean_rounds",
            "mean_total_reward",
            "mean_total_modification",
            "mean_final_min_acd",
        )
    }
    result = {
        "source_run": str(run_dir),
        "device": str(device),
        "paired_protocol": {
            "task_values": [
                float(item["magnitude_sensitivity"])
                for item in task_split["test"]
            ],
            "query_episodes_per_task": int(run_arguments["test_query_episodes"]),
            "test_case_seed": int(run_arguments["test_case_seed"]),
            "same_response_types_and_noise_across_modes": True,
            "policy_trust_state_retained_across_modes": True,
            "intervention": "initial opinion fusion only",
        },
        "aggregate": aggregate,
        "bidirectional_minus_equal": paired_difference(
            all_episodes["bidirectional_trust"], all_episodes["equal_weight"]
        ),
        "bidirectional_minus_human_only": paired_difference(
            all_episodes["bidirectional_trust"], all_episodes["human_only"]
        ),
        "formal_result_reproduction_difference": reproduction_check,
        "per_task": per_task,
    }

    output_dir = run_dir / "analysis_4p6_trust_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_trust_fusion_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (output_dir / "paired_trust_fusion_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = (
            "fusion_mode",
            "mean_initial_min_acd",
            "mean_initial_reference_mae",
            "success_rate",
            "mean_rounds",
            "mean_total_reward",
            "mean_total_modification",
            "mean_final_min_acd",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode in MODES:
            initial = aggregate[mode]["initial"]
            consensus = aggregate[mode]["consensus"]
            writer.writerow(
                {
                    "fusion_mode": mode,
                    "mean_initial_min_acd": initial["mean_initial_min_acd"],
                    "mean_initial_reference_mae": initial[
                        "mean_initial_reference_mae"
                    ],
                    **{name: consensus[name] for name in fieldnames[3:]},
                }
            )

    compact = {
        "output": str(output_dir),
        "device": str(device),
        "aggregate": aggregate,
        "bidirectional_minus_equal": result["bidirectional_minus_equal"],
        "bidirectional_minus_human_only": result[
            "bidirectional_minus_human_only"
        ],
        "formal_result_reproduction_difference": reproduction_check,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
