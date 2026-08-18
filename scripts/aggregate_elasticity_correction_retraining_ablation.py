"""汇总有/无响应弹性梯度矫正的完整三种子重训练消融。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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
    parser.add_argument("--corrected", type=Path, nargs="+", required=True)
    parser.add_argument("--uncorrected", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract(run_dir: Path) -> dict[str, Any]:
    arguments = _read_json(run_dir / "arguments.json")
    summary = _read_json(run_dir / "summary.json")
    result = summary["heldout_comparison"]["maml_initialization"]
    return {
        "seed": int(arguments["seed"]),
        "calibration_coefficient": float(arguments["calibration_coefficient"]),
        "policy_gradient_coefficient": float(
            arguments["policy_gradient_coefficient"]
        ),
        "run_dir": str(run_dir.resolve()),
        "best_iteration": int(summary["best_iteration"]),
        "zero_step": result["zero_step"],
        "adapted": result["adapted"],
        "adaptation_gain": result["adaptation_gain"],
        "per_task": result["per_task"],
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
    corrected = {
        record["seed"]: record
        for record in (_extract(path.resolve()) for path in args.corrected)
    }
    uncorrected = {
        record["seed"]: record
        for record in (_extract(path.resolve()) for path in args.uncorrected)
    }
    if set(corrected) != set(uncorrected):
        raise ValueError("有矫正和无矫正运行的种子集合必须一致。")

    per_seed_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    aggregate_values: dict[str, list[float]] = {}
    for seed in sorted(corrected):
        modes = {
            "without_correction": uncorrected[seed],
            "with_correction": corrected[seed],
        }
        for mode, record in modes.items():
            row: dict[str, Any] = {
                "seed": seed,
                "mode": mode,
                "best_iteration": record["best_iteration"],
            }
            for stage in ("zero_step", "adapted"):
                for metric in METRICS:
                    row[f"{stage}_{metric}"] = float(record[stage][metric])
            for metric in (
                "success_rate",
                "mean_rounds",
                "mean_total_reward",
            ):
                row[f"gain_{metric}"] = float(record["adaptation_gain"][metric])
            per_seed_rows.append(row)

        delta: dict[str, Any] = {"seed": seed}
        for stage in ("zero_step", "adapted"):
            for metric in METRICS:
                key = f"{stage}_{metric}"
                value = float(corrected[seed][stage][metric]) - float(
                    uncorrected[seed][stage][metric]
                )
                delta[f"delta_{key}"] = value
                aggregate_values.setdefault(f"delta_{key}", []).append(value)
        for metric in ("success_rate", "mean_rounds", "mean_total_reward"):
            key = f"gain_{metric}"
            value = float(corrected[seed]["adaptation_gain"][metric]) - float(
                uncorrected[seed]["adaptation_gain"][metric]
            )
            delta[f"delta_{key}"] = value
            aggregate_values.setdefault(f"delta_{key}", []).append(value)
        delta_rows.append(delta)

    method_summary: dict[str, Any] = {}
    for mode, records in (
        ("without_correction", uncorrected),
        ("with_correction", corrected),
    ):
        method_summary[mode] = {}
        for stage in ("zero_step", "adapted"):
            method_summary[mode][stage] = {
                metric: _mean_ci(
                    [float(records[seed][stage][metric]) for seed in sorted(records)]
                )
                for metric in METRICS
            }
        method_summary[mode]["adaptation_gain"] = {
            metric: _mean_ci(
                [
                    float(records[seed]["adaptation_gain"][metric])
                    for seed in sorted(records)
                ]
            )
            for metric in ("success_rate", "mean_rounds", "mean_total_reward")
        }

    output = {
        "design": (
            "Full retraining ablation. All recorded hyperparameters and OOD "
            "evaluation seeds are held fixed except calibration_coefficient "
            "(lambda_cal=1 versus 0). Seeds are paired."
        ),
        "seeds": sorted(corrected),
        "corrected_runs": corrected,
        "uncorrected_runs": uncorrected,
        "method_summary": method_summary,
        "paired_delta_with_minus_without": delta_rows,
        "paired_delta_summary": {
            key: _mean_ci(values) for key, values in aggregate_values.items()
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "retraining_ablation_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    with (output_dir / "retraining_per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (output_dir / "retraining_paired_deltas.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)
    print(json.dumps(method_summary, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            output["paired_delta_summary"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"结果已写入：{output_dir}")


if __name__ == "__main__":
    main()
