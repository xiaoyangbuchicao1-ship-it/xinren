"""比较固定动作先验与随机网络初始化的FOMAML-PPO训练结果。"""

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


FIXED_COLOR = "#0072B2"
RANDOM_COLOR = "#D55E00"
BEFORE_COLOR = "#A9A9A9"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _centered_mean(values: np.ndarray, window: int = 3) -> np.ndarray:
    """计算不改变点数的居中滑动平均。"""

    radius = window // 2
    return np.asarray(
        [
            np.mean(values[max(0, index - radius) : min(len(values), index + radius + 1)])
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def _load_run(run_dir: Path, name: str, color: str) -> dict:
    validation = _read_json(run_dir / "validation.json")
    summary = _read_json(run_dir / "summary.json")
    arguments = _read_json(run_dir / "arguments.json")
    best_iteration = int(summary["best_iteration"])
    best_row = next(row for row in validation if int(row["iteration"]) == best_iteration)
    heldout = summary["heldout_comparison"]
    adapted = np.asarray(
        [float(row["validation"]["adapted"]["mean_total_reward"]) for row in validation],
        dtype=np.float64,
    )
    return {
        "name": name,
        "color": color,
        "run_dir": run_dir,
        "seed": int(arguments["seed"]),
        "episodes": np.asarray(
            [int(row["cumulative_environment_episodes"]) for row in validation],
            dtype=np.int64,
        ),
        "adapted": adapted,
        "best_iteration": best_iteration,
        "best_episode": int(best_row["cumulative_environment_episodes"]),
        "best_return": float(best_row["validation"]["adapted"]["mean_total_reward"]),
        "initial_return": float(adapted[0]),
        "final_return": float(adapted[-1]),
        "heldout_before": float(
            heldout["before_meta_training"]["adapted"]["mean_total_reward"]
        ),
        "heldout_after": float(
            heldout["maml_initialization"]["adapted"]["mean_total_reward"]
        ),
        "heldout_success": float(
            heldout["maml_initialization"]["adapted"]["success_rate"]
        ),
        "heldout_rounds": float(
            heldout["maml_initialization"]["adapted"]["mean_rounds"]
        ),
        "ordinary_return": float(
            heldout["ordinary_initialization"]["adapted"]["mean_total_reward"]
        ),
    }


def _save(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory / f"{stem}.png", dpi=320)
    figure.savefig(directory / f"{stem}.pdf")
    plt.close(figure)


def _render_language(runs: list[dict], output_dir: Path, language: str) -> None:
    configure_research_plot_style(language)
    text = {
        "zh": {
            "fixed": "固定0.5先验",
            "random": "随机网络初始化（无先验）",
            "curve_title": "不同Actor初始化的留出任务适应后总回报",
            "episode_x": "累计元训练 episode",
            "return_y": "一步适应后平均总回报",
            "raw": "原始验证值",
            "smooth": "3点滑动平均",
            "best": "最佳检查点",
            "bar_title": "分布外任务：元训练前后的一步适应回报",
            "before": "元训练前",
            "after": "元训练后（最佳检查点）",
        },
        "en": {
            "fixed": "Fixed 0.5 prior",
            "random": "Random network initialization (no prior)",
            "curve_title": "Held-out Adapted Return under Different Actor Initializations",
            "episode_x": "Cumulative meta-training episodes",
            "return_y": "Mean total return after one-step adaptation",
            "raw": "Raw validation value",
            "smooth": "3-point moving average",
            "best": "Best checkpoint",
            "bar_title": "Out-of-distribution Adapted Return Before and After Meta-training",
            "before": "Before meta-training",
            "after": "After meta-training (best checkpoint)",
        },
    }[language]
    destination = output_dir / "figures" / language

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for index, run in enumerate(runs):
        label = text["fixed"] if index == 0 else text["random"]
        axis.plot(
            run["episodes"],
            run["adapted"],
            color=run["color"],
            linewidth=0.9,
            alpha=0.25,
        )
        axis.scatter(
            run["episodes"],
            run["adapted"],
            color=run["color"],
            s=12,
            alpha=0.35,
        )
        axis.plot(
            run["episodes"],
            _centered_mean(run["adapted"]),
            color=run["color"],
            linewidth=2.2,
            label=f"{label} — {text['smooth']}",
        )
        axis.scatter(
            [run["best_episode"]],
            [run["best_return"]],
            marker="*",
            s=90,
            color=run["color"],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    axis.set(
        title=text["curve_title"],
        xlabel=text["episode_x"],
        ylabel=text["return_y"],
    )
    axis.grid(alpha=0.20)
    axis.legend(frameon=False)
    _save(figure, destination, "01_adapted_validation_return")

    figure, axis = plt.subplots(figsize=(6.4, 4.1))
    x = np.arange(len(runs), dtype=np.float64)
    width = 0.33
    before = np.asarray([run["heldout_before"] for run in runs])
    after = np.asarray([run["heldout_after"] for run in runs])
    axis.bar(x - width / 2, before, width, color=BEFORE_COLOR, label=text["before"])
    axis.bar(
        x + width / 2,
        after,
        width,
        color=[run["color"] for run in runs],
        label=text["after"],
    )
    for position, (before_value, after_value) in enumerate(zip(before, after)):
        axis.text(
            position + width / 2,
            after_value + 0.00025,
            f"{after_value - before_value:+.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(x, [text["fixed"], text["random"]])
    axis.set(title=text["bar_title"], ylabel=text["return_y"])
    lower = min(float(before.min()), float(after.min())) - 0.004
    upper = max(float(before.max()), float(after.max())) + 0.004
    axis.set_ylim(lower, upper)
    axis.grid(axis="y", alpha=0.20)
    axis.legend(frameon=False)
    _save(figure, destination, "02_ood_before_after_meta_training")


def _write_summary(runs: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "initialization",
        "seed",
        "initial_validation_adapted_return",
        "best_validation_adapted_return",
        "best_meta_iteration",
        "best_cumulative_episode",
        "final_validation_adapted_return",
        "validation_initial_to_best_gain",
        "ood_before_meta_adapted_return",
        "ood_after_meta_adapted_return",
        "ood_meta_training_gain",
        "ood_success_rate",
        "ood_mean_rounds",
        "ordinary_ppo_ood_adapted_return",
        "ood_after_minus_ordinary_ppo",
        "source_directory",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "initialization": run["name"],
                    "seed": run["seed"],
                    "initial_validation_adapted_return": run["initial_return"],
                    "best_validation_adapted_return": run["best_return"],
                    "best_meta_iteration": run["best_iteration"],
                    "best_cumulative_episode": run["best_episode"],
                    "final_validation_adapted_return": run["final_return"],
                    "validation_initial_to_best_gain": run["best_return"]
                    - run["initial_return"],
                    "ood_before_meta_adapted_return": run["heldout_before"],
                    "ood_after_meta_adapted_return": run["heldout_after"],
                    "ood_meta_training_gain": run["heldout_after"] - run["heldout_before"],
                    "ood_success_rate": run["heldout_success"],
                    "ood_mean_rounds": run["heldout_rounds"],
                    "ordinary_ppo_ood_adapted_return": run["ordinary_return"],
                    "ood_after_minus_ordinary_ppo": run["heldout_after"]
                    - run["ordinary_return"],
                    "source_directory": str(run["run_dir"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_run", type=Path)
    parser.add_argument("random_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()

    runs = [
        _load_run(arguments.fixed_run.resolve(), "fixed_0.5", FIXED_COLOR),
        _load_run(arguments.random_run.resolve(), "random", RANDOM_COLOR),
    ]
    if runs[0]["seed"] != runs[1]["seed"]:
        raise ValueError("两次实验必须使用相同随机种子，才能比较初始化方式。")
    output_dir = arguments.output_dir.resolve()
    _write_summary(runs, output_dir)
    _render_language(runs, output_dir, "zh")
    _render_language(runs, output_dir, "en")
    print(output_dir)


if __name__ == "__main__":
    main()
