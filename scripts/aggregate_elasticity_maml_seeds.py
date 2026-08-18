"""汇总多个响应弹性FOMAML-PPO种子，并生成可复现的论文图表与表格。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plot_style import configure_research_plot_style


ADAPTED_COLOR = "#D95F02"
ZERO_COLOR = "#1B9E77"
TRAIN_COLOR = "#0072B2"
SEED_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
T_95_DF_2 = 4.302652729696142

# Only parameters that affect the realized optimization/evaluation trajectory belong
# here.  The requested iteration count and early-stopping controls are intentionally
# excluded: runs are comparable when their recorded axes and completed trajectories
# match, even if one reached that length through early stopping.
EFFECTIVE_TRAINING_ARGUMENTS = (
    "task_mode",
    "guidance_mode",
    "direct_action_low",
    "direct_action_high",
    "direct_initial_recommendation",
    "direct_state_signal",
    "response_interpolation",
    "task_split_mode",
    "elasticity_range_profile",
    "balanced_elasticity_batches",
    "meta_batch_size",
    "support_episodes",
    "query_episodes",
    "validation_query_episodes",
    "test_query_episodes",
    "validation_interval",
    "inner_learning_rate",
    "meta_learning_rate",
    "outer_update_epochs",
    "second_order_fast",
    "calibration_coefficient",
    "policy_gradient_coefficient",
    "fast_only_meta",
    "shared_meta_offset",
    "fresh_meta_actor",
    "meta_actor_initialization",
    "residual_head_gain",
    "reward_mode",
    "deficit_progress_weight",
    "deficit_modification_cost",
    "deficit_round_cost",
    "deficit_success_bonus",
    "deficit_timeout_penalty",
    "deficit_epsilon",
    "recommendation_cost_weight",
    "remaining_deficit_cost_weight",
    "unexecuted_recommendation_cost_weight",
    "task_split_seed",
    "validation_case_seed",
    "test_case_seed",
)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    """保持序列长度不变的居中滑动平均。"""

    radius = window // 2
    return np.asarray(
        [
            np.mean(values[max(0, index - radius) : min(len(values), index + radius + 1)])
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def _mean_and_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """沿种子维计算均值及小样本Student-t 95%置信区间半宽。"""

    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("当前论文汇总固定要求三个形状一致的种子序列。")
    mean = np.mean(values, axis=0)
    standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, T_95_DF_2 * standard_error


def _save(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory / f"{stem}.png", dpi=320)
    figure.savefig(directory / f"{stem}.pdf")
    plt.close(figure)


def _extract_run(run_dir: Path) -> dict:
    arguments = _read_json(run_dir / "arguments.json")
    summary = _read_json(run_dir / "summary.json")
    training = _read_json(run_dir / "training.json")
    validation = _read_json(run_dir / "validation.json")
    heldout = summary["heldout_comparison"]
    maml = heldout["maml_initialization"]
    ordinary = heldout["ordinary_initialization"]

    training_query_return = np.asarray(
        [
            np.mean(
                [float(task["query_rollout"]["mean_episode_reward"]) for task in row["tasks"]]
            )
            for row in training
        ],
        dtype=np.float64,
    )
    validation_zero = np.asarray(
        [float(row["validation"]["zero_step"]["mean_total_reward"]) for row in validation],
        dtype=np.float64,
    )
    validation_adapted = np.asarray(
        [float(row["validation"]["adapted"]["mean_total_reward"]) for row in validation],
        dtype=np.float64,
    )
    best_iteration = int(summary["best_iteration"])
    best_row = next(row for row in validation if int(row["iteration"]) == best_iteration)

    return {
        "run_dir": run_dir,
        "seed": int(arguments["seed"]),
        "config_hash": str(summary["experiment_config_hash"]),
        "training_signature": {
            key: arguments.get(key) for key in EFFECTIVE_TRAINING_ARGUMENTS
        },
        "decision": str(summary["decision"]),
        "best_iteration": best_iteration,
        "best_episode": int(best_row["cumulative_environment_episodes"]),
        "best_validation_return": float(
            best_row["validation"]["adapted"]["mean_total_reward"]
        ),
        "final_validation_return": float(validation_adapted[-1]),
        "training_query_return_std": float(np.std(training_query_return)),
        "training_iteration": np.asarray([int(row["iteration"]) for row in training]),
        "training_query_return": training_query_return,
        "validation_iteration": np.asarray([int(row["iteration"]) for row in validation]),
        "validation_episode": np.asarray(
            [int(row["cumulative_environment_episodes"]) for row in validation]
        ),
        "validation_zero": validation_zero,
        "validation_adapted": validation_adapted,
        "ood_ordinary_adapted_return": float(ordinary["adapted"]["mean_total_reward"]),
        "ood_maml_zero_return": float(maml["zero_step"]["mean_total_reward"]),
        "ood_maml_adapted_return": float(maml["adapted"]["mean_total_reward"]),
        "ood_adaptation_gain": float(maml["adaptation_gain"]["mean_total_reward"]),
        "ood_maml_success_rate": float(maml["adapted"]["success_rate"]),
        "ood_maml_mean_rounds": float(maml["adapted"]["mean_rounds"]),
        "ood_maml_minus_ordinary": float(
            heldout["maml_minus_ordinary_after_adaptation"]["mean_total_reward"]
        ),
    }


def _validate_runs(runs: list[dict]) -> None:
    if len(runs) != 3:
        raise ValueError("当前论文汇总固定要求三个独立种子。")
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("输入目录包含重复随机种子。")
    if len({run["config_hash"] for run in runs}) != 1:
        raise ValueError("实验配置哈希不同，不能直接进行多种子汇总。")
    reference_signature = runs[0]["training_signature"]
    for run in runs[1:]:
        mismatches = [
            key
            for key in EFFECTIVE_TRAINING_ARGUMENTS
            if run["training_signature"][key] != reference_signature[key]
        ]
        if mismatches:
            raise ValueError(
                f"seed {run['seed']} 的有效训练参数不一致：{', '.join(mismatches)}"
            )
    for key in ("training_iteration", "validation_iteration", "validation_episode"):
        reference = runs[0][key]
        if any(not np.array_equal(reference, run[key]) for run in runs[1:]):
            raise ValueError(f"不同种子的{key}横轴不一致。")


def _plot_band(
    axis: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
) -> None:
    mean, ci = _mean_and_ci(values)
    axis.plot(x, mean, color=color, linewidth=2.0, label=label)
    axis.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.16, linewidth=0)


def _render_language(runs: list[dict], output_dir: Path, language: str) -> None:
    configure_research_plot_style(language)
    text = {
        "zh": {
            "meta_x": "元外循环次数",
            "episode_x": "累计训练 episode",
            "return_y": "平均总回报",
            "gain_y": r"一步适应回报增益 $\Delta J$",
            "training_title": "训练任务查询回报（三种子）",
            "validation_title": "留出任务适应前后回报（三种子）",
            "gain_title": "留出任务一步适应增益（三种子）",
            "ood_title": "分布外任务适应回报的配对比较",
            "training": "查询回报（11点滑动平均）",
            "adapted": "一步适应后",
            "zero": "适应前",
            "gain": "适应增益",
            "ordinary": "普通初始化\n一步适应后",
            "maml": "FOMAML-PPO\n一步适应后",
            "ci": "阴影：95%置信区间",
            "ood_delta_title": "FOMAML-PPO相对普通初始化的OOD回报提升",
            "delta_y": "配对回报差值",
            "mean_ci": "均值及95%置信区间",
        },
        "en": {
            "meta_x": "Meta-iteration",
            "episode_x": "Cumulative training episodes",
            "return_y": "Mean total return",
            "gain_y": r"One-step adaptation gain $\Delta J$",
            "training_title": "Training-task Query Return (Three Seeds)",
            "validation_title": "Held-out Return Before and After Adaptation (Three Seeds)",
            "gain_title": "One-step Adaptation Gain on Held-out Tasks (Three Seeds)",
            "ood_title": "Paired OOD Return after One-step Adaptation",
            "training": "Query return (11-point moving average)",
            "adapted": "After one-step adaptation",
            "zero": "Before adaptation",
            "gain": "Adaptation gain",
            "ordinary": "Ordinary init.\nafter adaptation",
            "maml": "FOMAML-PPO\nafter adaptation",
            "ci": "Shading: 95% confidence interval",
            "ood_delta_title": "OOD Return Improvement over Ordinary Initialization",
            "delta_y": "Paired return difference",
            "mean_ci": "Mean and 95% confidence interval",
        },
    }[language]
    figure_dir = output_dir / "figures" / language

    train_values = np.vstack(
        [_centered_mean(run["training_query_return"], 11) for run in runs]
    )
    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _plot_band(
        axis,
        runs[0]["training_iteration"],
        train_values,
        label=text["training"],
        color=TRAIN_COLOR,
    )
    axis.set(
        title=text["training_title"],
        xlabel=text["meta_x"],
        ylabel=text["return_y"],
    )
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, title=text["ci"])
    _save(figure, figure_dir, "01_training_query_return_mean_ci")

    validation_zero = np.vstack([run["validation_zero"] for run in runs])
    validation_adapted = np.vstack([run["validation_adapted"] for run in runs])
    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _plot_band(
        axis,
        runs[0]["validation_episode"],
        validation_adapted,
        label=text["adapted"],
        color=ADAPTED_COLOR,
    )
    _plot_band(
        axis,
        runs[0]["validation_episode"],
        validation_zero,
        label=text["zero"],
        color=ZERO_COLOR,
    )
    axis.set(
        title=text["validation_title"],
        xlabel=text["episode_x"],
        ylabel=text["return_y"],
    )
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, title=text["ci"])
    _save(figure, figure_dir, "02_validation_return_mean_ci")

    gain = validation_adapted - validation_zero
    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _plot_band(
        axis,
        runs[0]["validation_episode"],
        gain,
        label=text["gain"],
        color=ADAPTED_COLOR,
    )
    axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axis.set(
        title=text["gain_title"],
        xlabel=text["episode_x"],
        ylabel=text["gain_y"],
    )
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, title=text["ci"])
    _save(figure, figure_dir, "03_validation_adaptation_gain_mean_ci")

    ordinary = np.asarray([run["ood_ordinary_adapted_return"] for run in runs])
    maml = np.asarray([run["ood_maml_adapted_return"] for run in runs])
    figure, axis = plt.subplots(figsize=(5.4, 4.1))
    for index, run in enumerate(runs):
        color = SEED_COLORS[index % len(SEED_COLORS)]
        axis.plot([0, 1], [ordinary[index], maml[index]], color=color, alpha=0.75)
        axis.scatter(
            [0, 1],
            [ordinary[index], maml[index]],
            color=color,
            s=32,
            label=f"seed {run['seed']}",
            zorder=3,
        )
    axis.scatter(
        [0, 1],
        [np.mean(ordinary), np.mean(maml)],
        marker="D",
        color="black",
        s=44,
        label="mean" if language == "en" else "均值",
        zorder=4,
    )
    axis.set_xticks([0, 1], [text["ordinary"], text["maml"]])
    axis.set(title=text["ood_title"], ylabel=text["return_y"])
    axis.grid(axis="y", alpha=0.20)
    axis.legend(frameon=False, ncol=2)
    _save(figure, figure_dir, "04_ood_paired_return")

    delta = maml - ordinary
    delta_mean = float(np.mean(delta))
    delta_ci = float(T_95_DF_2 * np.std(delta, ddof=1) / np.sqrt(len(delta)))
    figure, axis = plt.subplots(figsize=(5.4, 4.1))
    positions = np.arange(len(runs), dtype=np.float64)
    for index, run in enumerate(runs):
        axis.scatter(
            positions[index],
            delta[index],
            color=SEED_COLORS[index % len(SEED_COLORS)],
            s=42,
            label=f"seed {run['seed']}",
            zorder=3,
        )
    mean_position = float(len(runs) + 0.25)
    axis.errorbar(
        mean_position,
        delta_mean,
        yerr=delta_ci,
        fmt="D",
        color="black",
        capsize=5,
        linewidth=1.6,
        label=text["mean_ci"],
        zorder=4,
    )
    axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axis.set_xticks(
        [*positions, mean_position],
        [*[str(run["seed"]) for run in runs], "mean" if language == "en" else "均值"],
    )
    axis.set(title=text["ood_delta_title"], xlabel="seed", ylabel=text["delta_y"])
    axis.grid(axis="y", alpha=0.20)
    axis.legend(frameon=False, ncol=2)
    _save(figure, figure_dir, "05_ood_improvement_mean_ci")


def _write_tables(runs: list[dict], output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "run_dir",
        "decision",
        "best_iteration",
        "best_episode",
        "best_validation_return",
        "final_validation_return",
        "training_query_return_std",
        "ood_ordinary_adapted_return",
        "ood_maml_zero_return",
        "ood_maml_adapted_return",
        "ood_adaptation_gain",
        "ood_maml_minus_ordinary",
        "ood_maml_success_rate",
        "ood_maml_mean_rounds",
    ]
    rows = [
        {
            field: str(run[field]) if field == "run_dir" else run[field]
            for field in fields
        }
        for run in runs
    ]
    with (table_dir / "per_seed_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_fields = [
        "best_validation_return",
        "final_validation_return",
        "training_query_return_std",
        *[field for field in fields if field.startswith("ood_")],
    ]
    metric_statistics = {}
    for field in metric_fields:
        values = np.asarray([run[field] for run in runs], dtype=np.float64)
        mean = float(np.mean(values))
        sample_std = float(np.std(values, ddof=1))
        ci95_half_width = float(T_95_DF_2 * sample_std / np.sqrt(len(values)))
        metric_statistics[field] = {
            "mean": mean,
            "sample_std": sample_std,
            "ci95_half_width": ci95_half_width,
            "ci95_low": mean - ci95_half_width,
            "ci95_high": mean + ci95_half_width,
        }
    aggregate = {
        "seed_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "config_hash": runs[0]["config_hash"],
        "all_decisions_go": all(run["decision"] == "GO" for run in runs),
        "metrics": metric_statistics,
    }
    with (table_dir / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    with (table_dir / "aggregate_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "mean",
                "sample_std",
                "ci95_half_width",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        for metric, statistics in aggregate["metrics"].items():
            writer.writerow({"metric": metric, **statistics})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()

    runs = [_extract_run(path.resolve()) for path in arguments.run_dirs]
    runs.sort(key=lambda run: run["seed"])
    _validate_runs(runs)
    output_dir = arguments.output_dir.resolve()
    _render_language(runs, output_dir, "zh")
    _render_language(runs, output_dir, "en")
    _write_tables(runs, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
