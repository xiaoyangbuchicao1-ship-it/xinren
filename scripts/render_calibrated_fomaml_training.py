"""从校准式FOMAML-PPO原始日志生成中英文论文训练图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plot_style import configure_research_plot_style


RAW_COLOR = "#9ECAE1"
TREND_COLOR = "#0868AC"
ADAPTED_COLOR = "#D95F02"
ZERO_COLOR = "#1B9E77"
ACTOR_COLOR = "#0072B2"
CRITIC_COLOR = "#CC79A7"
BEST_COLOR = "#6A3D9A"
THRESHOLD_COLOR = "#4D4D4D"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_run_config(run_dir: Path) -> dict:
    candidates = sorted(run_dir.glob("config_*.yaml"))
    if len(candidates) != 1:
        raise ValueError(
            f"运行目录必须包含且仅包含一个config_*.yaml，实际找到{len(candidates)}个。"
        )
    with candidates[0].open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    """计算保持长度不变的居中滑动平均，不制造额外训练点。"""

    radius = window // 2
    return np.asarray(
        [
            np.mean(values[max(0, index - radius) : min(len(values), index + radius + 1)])
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def _save(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory / f"{stem}.png", dpi=320)
    figure.savefig(directory / f"{stem}.pdf")
    plt.close(figure)


def _raw_and_trend(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    raw_label: str,
    trend_label: str,
    window: int,
    color: str = TREND_COLOR,
) -> None:
    axis.plot(x, y, color=RAW_COLOR, linewidth=0.8, alpha=0.42, label=raw_label)
    if len(y) <= 60:
        axis.scatter(x, y, color=RAW_COLOR, s=12, alpha=0.48, zorder=2)
    axis.plot(
        x,
        _centered_mean(y, window),
        color=color,
        linewidth=2.2,
        label=trend_label,
        zorder=3,
    )


def _mark_best(axis: plt.Axes, best_iteration: int, label: str) -> None:
    axis.axvline(
        best_iteration,
        color=BEST_COLOR,
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        alpha=0.90,
        label=label,
        zorder=1,
    )


def _finish_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, alpha=0.18, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.015)


def _plot_validation_return(
    axis: plt.Axes,
    x: np.ndarray,
    adapted: np.ndarray,
    zero: np.ndarray,
    *,
    adapted_label: str,
    zero_label: str,
) -> None:
    axis.plot(x, adapted, color=ADAPTED_COLOR, linewidth=0.8, alpha=0.24)
    axis.plot(x, zero, color=ZERO_COLOR, linewidth=0.8, alpha=0.24)
    axis.plot(
        x,
        _centered_mean(adapted, 3),
        color=ADAPTED_COLOR,
        linewidth=2.1,
        marker="o",
        markersize=3.4,
        markevery=2,
        label=adapted_label,
    )
    axis.plot(
        x,
        _centered_mean(zero, 3),
        color=ZERO_COLOR,
        linewidth=2.1,
        marker="s",
        markersize=3.2,
        markevery=2,
        label=zero_label,
    )


def _render_language(run_dir: Path, language: str) -> None:
    configure_research_plot_style(language)
    training = _read_json(run_dir / "training.json")
    validation = _read_json(run_dir / "validation.json")
    summary = _read_json(run_dir / "summary.json")
    config = _read_run_config(run_dir)
    output = run_dir / "paper_figures" / language
    best_iteration = int(summary["best_iteration"])
    gradient_clip = float(config["ppo"]["max_gradient_norm"])
    target_kl = float(config["ppo"]["target_kl"])

    text = {
        "zh": {
            "meta_x": "元外循环次数",
            "reward_y": "平均总回报",
            "gain_y": r"一步适应回报增益 $\Delta J$",
            "mse_y": "均方误差",
            "gradient_y": "裁剪前梯度范数",
            "kl_y": "近似 KL（对数尺度）",
            "failure_y": "失败率（%）",
            "overview_title": "校准式 FOMAML-PPO 训练概览（单种子）",
            "train_title": "训练任务查询回报（单种子诊断）",
            "gain_title": "留出任务一步适应增益",
            "return_title": "留出任务适应前后回报",
            "failure_title": "留出任务失败率（越低越好）",
            "critic_title": "Critic 查询损失",
            "actor_gradient_title": "Actor 元梯度范数",
            "kl_title": "元策略更新幅度",
            "raw": "逐次元更新原始值",
            "smooth11": "11点居中滑动平均",
            "smooth3": "3点滑动平均",
            "adapted": "一步适应后",
            "zero": "适应前",
            "best": f"最佳模型（第{best_iteration}次）",
            "gradient_clip": f"梯度裁剪阈值 {gradient_clip:g}",
            "target_kl": f"目标 KL {target_kl:g}",
        },
        "en": {
            "meta_x": "Meta-iteration",
            "reward_y": "Mean total return",
            "gain_y": r"One-step adaptation gain $\Delta J$",
            "mse_y": "Mean squared error",
            "gradient_y": "Pre-clipping gradient norm",
            "kl_y": "Approximate KL (log scale)",
            "failure_y": "Failure rate (%)",
            "overview_title": "Calibrated FOMAML-PPO Training Overview (Single Seed)",
            "train_title": "Training-task Query Return (Single-Seed Diagnostic)",
            "gain_title": "One-step Adaptation Gain on Held-out Tasks",
            "return_title": "Held-out Return Before and After Adaptation",
            "failure_title": "Held-out Failure Rate (Lower Is Better)",
            "critic_title": "Critic Query Loss",
            "actor_gradient_title": "Actor Meta-gradient Norm",
            "kl_title": "Meta-policy Update Magnitude",
            "raw": "Per-meta-update raw value",
            "smooth11": "11-point centered moving average",
            "smooth3": "3-point moving average",
            "adapted": "After one-step adaptation",
            "zero": "Before adaptation",
            "best": f"Best model (iteration {best_iteration})",
            "gradient_clip": f"Gradient clipping threshold {gradient_clip:g}",
            "target_kl": f"Target KL {target_kl:g}",
        },
    }[language]

    iterations = np.asarray([int(row["iteration"]) for row in training])
    query_reward = np.asarray(
        [
            np.mean(
                [float(task["query_rollout"]["mean_episode_reward"]) for task in row["tasks"]]
            )
            for row in training
        ],
        dtype=np.float64,
    )
    critic = np.asarray(
        [float(row["meta_update"]["mean_query_critic_loss"]) for row in training],
        dtype=np.float64,
    )
    actor_gradient = np.asarray(
        [float(row["meta_update"]["actor_gradient_norm"]) for row in training],
        dtype=np.float64,
    )
    approximate_kl = np.asarray(
        [float(row["meta_update"]["mean_query_approximate_kl"]) for row in training],
        dtype=np.float64,
    )
    approximate_kl = np.clip(approximate_kl, 1.0e-12, None)

    val_iteration = np.asarray([int(row["iteration"]) for row in validation])
    adapted = np.asarray(
        [float(row["validation"]["adapted"]["mean_total_reward"]) for row in validation]
    )
    zero = np.asarray(
        [float(row["validation"]["zero_step"]["mean_total_reward"]) for row in validation]
    )
    gain = adapted - zero
    adapted_failure = 100.0 * np.asarray(
        [1.0 - float(row["validation"]["adapted"]["success_rate"]) for row in validation]
    )
    zero_failure = 100.0 * np.asarray(
        [1.0 - float(row["validation"]["zero_step"]["success_rate"]) for row in validation]
    )

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    figure.suptitle(text["overview_title"], fontsize=14.0)
    _raw_and_trend(
        axes[0, 0],
        iterations,
        query_reward,
        raw_label=text["raw"],
        trend_label=text["smooth11"],
        window=11,
    )
    _mark_best(axes[0, 0], best_iteration, text["best"])
    axes[0, 0].set(
        title=text["train_title"], xlabel=text["meta_x"], ylabel=text["reward_y"]
    )
    _plot_validation_return(
        axes[0, 1],
        val_iteration,
        adapted,
        zero,
        adapted_label=text["adapted"],
        zero_label=text["zero"],
    )
    _mark_best(axes[0, 1], best_iteration, text["best"])
    axes[0, 1].set(
        title=text["return_title"], xlabel=text["meta_x"], ylabel=text["reward_y"]
    )
    _raw_and_trend(
        axes[1, 0],
        val_iteration,
        gain,
        raw_label=text["raw"],
        trend_label=text["smooth3"],
        window=3,
        color=ADAPTED_COLOR,
    )
    axes[1, 0].axhline(0.0, color=THRESHOLD_COLOR, linewidth=0.9, linestyle="--")
    _mark_best(axes[1, 0], best_iteration, text["best"])
    axes[1, 0].set(
        title=text["gain_title"], xlabel=text["meta_x"], ylabel=text["gain_y"]
    )
    _raw_and_trend(
        axes[1, 1],
        iterations,
        critic,
        raw_label=text["raw"],
        trend_label=text["smooth11"],
        window=11,
        color=CRITIC_COLOR,
    )
    _mark_best(axes[1, 1], best_iteration, text["best"])
    axes[1, 1].set(
        title=text["critic_title"], xlabel=text["meta_x"], ylabel=text["mse_y"]
    )
    for axis in axes.flat:
        _finish_axis(axis)
        axis.legend(frameon=False, fontsize=8.0)
    _save(figure, output, "00_training_overview")

    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _raw_and_trend(
        axis,
        iterations,
        query_reward,
        raw_label=text["raw"],
        trend_label=text["smooth11"],
        window=11,
    )
    _mark_best(axis, best_iteration, text["best"])
    axis.set(title=text["train_title"], xlabel=text["meta_x"], ylabel=text["reward_y"])
    _finish_axis(axis)
    axis.legend(frameon=False)
    _save(figure, output, "01_training_query_return")

    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _plot_validation_return(
        axis,
        val_iteration,
        adapted,
        zero,
        adapted_label=text["adapted"],
        zero_label=text["zero"],
    )
    _mark_best(axis, best_iteration, text["best"])
    axis.set(
        title=text["return_title"],
        xlabel=text["meta_x"],
        ylabel=text["reward_y"],
    )
    _finish_axis(axis)
    axis.legend(frameon=False)
    _save(figure, output, "02_validation_return")

    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _raw_and_trend(
        axis,
        val_iteration,
        gain,
        raw_label=text["raw"],
        trend_label=text["smooth3"],
        window=3,
        color=ADAPTED_COLOR,
    )
    axis.axhline(0.0, color=THRESHOLD_COLOR, linewidth=1.0, linestyle="--")
    _mark_best(axis, best_iteration, text["best"])
    axis.set(title=text["gain_title"], xlabel=text["meta_x"], ylabel=text["gain_y"])
    _finish_axis(axis)
    axis.legend(frameon=False)
    _save(figure, output, "03_validation_adaptation_gain")

    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _plot_validation_return(
        axis,
        val_iteration,
        adapted_failure,
        zero_failure,
        adapted_label=text["adapted"],
        zero_label=text["zero"],
    )
    _mark_best(axis, best_iteration, text["best"])
    axis.set(
        title=text["failure_title"],
        xlabel=text["meta_x"],
        ylabel=text["failure_y"],
    )
    axis.set_ylim(bottom=0.0)
    _finish_axis(axis)
    axis.legend(frameon=False)
    _save(figure, output, "04_validation_failure_rate")

    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _raw_and_trend(
        axis,
        iterations,
        critic,
        raw_label=text["raw"],
        trend_label=text["smooth11"],
        window=11,
        color=CRITIC_COLOR,
    )
    _mark_best(axis, best_iteration, text["best"])
    axis.set(title=text["critic_title"], xlabel=text["meta_x"], ylabel=text["mse_y"])
    _finish_axis(axis)
    axis.legend(frameon=False)
    _save(figure, output, "05_critic_loss")

    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _raw_and_trend(
        axis,
        iterations,
        actor_gradient,
        raw_label=text["raw"],
        trend_label=text["smooth11"],
        window=11,
        color=ACTOR_COLOR,
    )
    axis.axhline(
        gradient_clip,
        color=THRESHOLD_COLOR,
        linewidth=1.0,
        linestyle="--",
        label=text["gradient_clip"],
    )
    _mark_best(axis, best_iteration, text["best"])
    axis.set(
        title=text["actor_gradient_title"],
        xlabel=text["meta_x"],
        ylabel=text["gradient_y"],
    )
    axis.set_ylim(bottom=0.0)
    _finish_axis(axis)
    axis.legend(frameon=False)
    _save(figure, output, "06_actor_gradient_norm")

    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    _raw_and_trend(
        axis,
        iterations,
        approximate_kl,
        raw_label=text["raw"],
        trend_label=text["smooth11"],
        window=11,
        color=ACTOR_COLOR,
    )
    axis.axhline(
        target_kl,
        color=THRESHOLD_COLOR,
        linewidth=1.0,
        linestyle="--",
        label=text["target_kl"],
    )
    _mark_best(axis, best_iteration, text["best"])
    axis.set_yscale("log")
    axis.set(title=text["kl_title"], xlabel=text["meta_x"], ylabel=text["kl_y"])
    _finish_axis(axis)
    axis.legend(frameon=False)
    _save(figure, output, "07_policy_kl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    _render_language(run_dir, "zh")
    _render_language(run_dir, "en")
    print(run_dir / "paper_figures")


if __name__ == "__main__":
    main()
