"""Aggregate Section 4.5 reward-weight sensitivity runs on the formal OOD test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_run(run_dir: Path) -> dict[str, Any]:
    arguments = read_json(run_dir / "arguments.json")
    summary = read_json(run_dir / "summary.json")
    heldout = read_json(run_dir / "heldout_comparison.json")
    adapted = heldout["maml_initialization"]["adapted"]
    unexecuted_weight = float(arguments["unexecuted_recommendation_cost_weight"])
    unexecuted_component = float(
        adapted["mean_reward_components"]["unexecuted_recommendation_cost"]
    )
    if unexecuted_weight <= 0.0:
        cumulative_unexecuted = None
    else:
        cumulative_unexecuted = -unexecuted_component / unexecuted_weight
    return {
        "run_dir": str(run_dir.resolve()),
        "best_iteration": int(summary["best_iteration"]),
        "decision": summary["decision"],
        "modification_cost_weight": float(arguments["deficit_modification_cost"]),
        "unexecuted_cost_weight": unexecuted_weight,
        "episode_count": int(adapted["episode_count"]),
        "success_rate": float(adapted["success_rate"]),
        "mean_rounds": float(adapted["mean_rounds"]),
        "mean_total_modification": float(adapted["mean_total_modification"]),
        "mean_cumulative_unexecuted_recommendation": cumulative_unexecuted,
        "mean_final_min_acd": float(adapted["mean_final_min_acd"]),
        "mean_recommendation_magnitude": float(adapted["active_multiplier_mean"]),
        "within_configuration_mean_total_reward": float(adapted["mean_total_reward"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formal_run", type=Path)
    parser.add_argument("modification_low_run", type=Path)
    parser.add_argument("modification_high_run", type=Path)
    parser.add_argument("unexecuted_low_run", type=Path)
    parser.add_argument("unexecuted_high_run", type=Path)
    arguments = parser.parse_args()

    formal = extract_run(arguments.formal_run)
    modification = sorted(
        [
            extract_run(arguments.modification_low_run),
            formal,
            extract_run(arguments.modification_high_run),
        ],
        key=lambda item: item["modification_cost_weight"],
    )
    unexecuted = sorted(
        [
            extract_run(arguments.unexecuted_low_run),
            formal,
            extract_run(arguments.unexecuted_high_run),
        ],
        key=lambda item: item["unexecuted_cost_weight"],
    )

    fixed_grid_path = arguments.formal_run / "analysis_4p5" / "fixed_grid_ood_summary.json"
    fixed_grid = read_json(fixed_grid_path)
    result = {
        "comparison_protocol": {
            "task_values": fixed_grid["test_tasks"],
            "query_episodes_per_task": fixed_grid["query_episodes_per_task"],
            "test_case_seed": 51001,
            "one_factor_at_a_time": True,
            "cross_weight_reward_warning": (
                "Total returns are retained for auditing but are not compared across "
                "different reward definitions."
            ),
        },
        "fixed_feedback_grid": fixed_grid["fixed_aggregate"],
        "best_fixed_feedback": fixed_grid["best_fixed_by_mean_total_reward"],
        "formal_fomaml_ppo_adapted": fixed_grid["formal_fomaml_ppo_adapted"],
        "modification_cost_sensitivity": modification,
        "unexecuted_cost_sensitivity": unexecuted,
    }

    output_dir = arguments.formal_run / "analysis_4p5"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "section4_5_experiment_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    fieldnames = (
        "analysis",
        "parameter_value",
        "best_iteration",
        "success_rate",
        "mean_rounds",
        "mean_total_modification",
        "mean_cumulative_unexecuted_recommendation",
        "mean_final_min_acd",
        "mean_recommendation_magnitude",
        "run_dir",
    )
    with (output_dir / "reward_sensitivity_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for analysis, parameter, records in (
            ("modification_cost", "modification_cost_weight", modification),
            ("unexecuted_cost", "unexecuted_cost_weight", unexecuted),
        ):
            for record in records:
                writer.writerow(
                    {
                        "analysis": analysis,
                        "parameter_value": record[parameter],
                        **{name: record[name] for name in fieldnames[2:]},
                    }
                )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
