"""Aggregate the paper Section 4.5 fixed-feedback grid over all OOD tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = (
    "success_rate",
    "timeout_rate",
    "mean_rounds",
    "mean_final_min_acd",
    "mean_final_mean_acd",
    "mean_total_reward",
    "mean_total_modification",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sensitivity_label(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def weighted_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    counts = [int(record["episode_count"]) for record in records]
    total = sum(counts)
    if total <= 0:
        raise ValueError("The aggregate must contain at least one episode.")
    result: dict[str, float | int] = {"episode_count": total}
    for field in SUMMARY_FIELDS:
        result[field] = sum(
            float(record[field]) * count
            for record, count in zip(records, counts, strict=True)
        ) / total
    component_names = records[0]["mean_reward_components"].keys()
    result["mean_reward_components"] = {
        name: sum(
            float(record["mean_reward_components"][name]) * count
            for record, count in zip(records, counts, strict=True)
        )
        / total
        for name in component_names
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input-tag", default="paper4p5_seed51001")
    parser.add_argument(
        "--fixed-actions",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    arguments = parser.parse_args()

    run_dir = arguments.run_dir.resolve()
    task_split = read_json(run_dir / "task_split.json")
    heldout = read_json(run_dir / "heldout_comparison.json")
    tasks = [float(item["magnitude_sensitivity"]) for item in task_split["test"]]
    fixed_actions = list(dict.fromkeys(float(value) for value in arguments.fixed_actions))

    per_task: list[dict[str, Any]] = []
    fixed_records: dict[float, list[dict[str, Any]]] = {
        action: [] for action in fixed_actions
    }
    for task in tasks:
        directory = (
            run_dir
            / "micro_comparison"
            / f"sensitivity_{sensitivity_label(task)}_fixed_grid_{arguments.input_tag}"
        )
        payload = read_json(directory / "fixed_grid_vs_fomaml_micro.json")
        task_record: dict[str, Any] = {"magnitude_sensitivity": task}
        for action in fixed_actions:
            key = f"fixed_{str(action).replace('.', 'p')}"
            summary = payload["aggregate_64_cases"][key]
            fixed_records[action].append(summary)
            task_record[key] = summary
        per_task.append(task_record)

    fixed_aggregate = {
        f"fixed_{str(action).replace('.', 'p')}": weighted_summary(records)
        for action, records in fixed_records.items()
    }
    best_key = max(
        fixed_aggregate,
        key=lambda key: float(fixed_aggregate[key]["mean_total_reward"]),
    )
    best_fixed = fixed_aggregate[best_key]
    maml = heldout["maml_initialization"]["adapted"]
    result = {
        "source_run": str(run_dir),
        "test_tasks": tasks,
        "query_episodes_per_task": int(
            heldout["maml_initialization"]["query_episodes_per_task"]
        ),
        "fixed_actions": fixed_actions,
        "fixed_aggregate": fixed_aggregate,
        "best_fixed_by_mean_total_reward": {
            "key": best_key,
            **best_fixed,
        },
        "formal_fomaml_ppo_adapted": maml,
        "fomaml_minus_best_fixed": {
            "success_rate": float(maml["success_rate"])
            - float(best_fixed["success_rate"]),
            "mean_rounds": float(best_fixed["mean_rounds"])
            - float(maml["mean_rounds"]),
            "mean_total_reward": float(maml["mean_total_reward"])
            - float(best_fixed["mean_total_reward"]),
            "mean_total_modification": float(maml["mean_total_modification"])
            - float(best_fixed["mean_total_modification"]),
        },
        "per_task_fixed_results": per_task,
    }

    output_dir = run_dir / "analysis_4p5"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fixed_grid_ood_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (output_dir / "fixed_grid_ood_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "success_rate",
                "mean_rounds",
                "mean_total_reward",
                "mean_total_modification",
                "mean_final_min_acd",
            ),
        )
        writer.writeheader()
        for key, summary in fixed_aggregate.items():
            writer.writerow(
                {"method": key, **{field: summary[field] for field in writer.fieldnames[1:]}}
            )
        writer.writerow(
            {"method": "fomaml_ppo_adapted", **{field: maml[field] for field in writer.fieldnames[1:]}}
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
