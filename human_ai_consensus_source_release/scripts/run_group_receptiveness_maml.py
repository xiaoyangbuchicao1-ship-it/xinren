"""训练以群体响应规律为元任务的共享快参数FOMAML-PPO。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.maml_ppo import MetaGradientOptimizer
from src.analysis.plot_style import configure_plot_style
from src.common.config import config_hash, load_config
from src.common.encoding import configure_console_utf8, write_json
from src.common.logger import append_jsonl, create_run_directory
from src.common.seed import set_global_seed
from src.experiments.continuous_ppo import create_continuous_trainer
from src.experiments.group_receptiveness_maml import (
    evaluate_group_receptiveness_adaptation,
    make_group_receptiveness_task_split,
    sample_group_receptiveness_tasks,
    train_group_receptiveness_meta_iteration,
)
from src.experiments.response_elasticity_maml import (
    evaluate_response_elasticity_adaptation,
    sample_response_elasticity_tasks,
    sample_symmetric_response_elasticity_tasks,
    train_response_elasticity_meta_iteration,
)
from src.experiments.response_elasticity_task import (
    make_response_elasticity_ood_task_split,
    make_response_elasticity_task_split,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--task-mode",
        choices=("receptiveness", "elasticity"),
        default="receptiveness",
    )
    parser.add_argument(
        "--guidance-mode",
        choices=("static_optimizer", "direct"),
        default="static_optimizer",
        help="static_optimizer输出理论量倍率；direct直接输出建议比例。",
    )
    parser.add_argument("--direct-action-low", type=float, default=0.01)
    parser.add_argument("--direct-action-high", type=float, default=0.99)
    parser.add_argument("--direct-initial-recommendation", type=float, default=0.30)
    parser.add_argument(
        "--direct-state-signal",
        choices=("adjustment_distance", "consensus_deficit"),
        default="adjustment_distance",
    )
    parser.add_argument(
        "--response-interpolation",
        choices=("step", "linear"),
        default="linear",
    )
    parser.add_argument(
        "--task-split-mode",
        choices=("stratified", "range_ood"),
        default="range_ood",
    )
    parser.add_argument(
        "--elasticity-range-profile",
        choices=("moderate", "wide"),
        default="wide",
    )
    parser.add_argument(
        "--balanced-elasticity-batches",
        action="store_true",
        help="每个响应弹性元批次按绝对值成对包含正负任务。",
    )
    parser.add_argument("--meta-iterations", type=int, default=20)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--support-episodes", type=int, default=10)
    parser.add_argument("--query-episodes", type=int, default=10)
    parser.add_argument("--validation-query-episodes", type=int, default=24)
    parser.add_argument("--test-query-episodes", type=int, default=48)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="连续多少次验证未刷新最佳回报后停止；0表示禁用。",
    )
    parser.add_argument(
        "--early-stopping-min-iterations",
        type=int,
        default=0,
        help="达到该元外循环次数后才允许触发验证早停。",
    )
    parser.add_argument(
        "--inner-learning-rate",
        type=float,
        default=None,
        help="未指定时，直接响应弹性任务使用0.8，其余任务使用0.01。",
    )
    parser.add_argument("--meta-learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--outer-update-epochs",
        type=int,
        default=10,
        help="每批支持/查询数据的PPO裁剪外更新轮数。",
    )
    parser.add_argument(
        "--second-order-fast",
        action="store_true",
        help="显式启用二阶快参数梯度；默认使用用户选定的一阶FOMAML。",
    )
    parser.add_argument(
        "--calibration-coefficient",
        type=float,
        default=None,
        help="未指定时，直接响应弹性任务启用响应证据校准（1.0）。",
    )
    parser.add_argument(
        "--policy-gradient-coefficient",
        type=float,
        default=None,
        help="未指定时，直接响应弹性任务仅保留0.05的支持集PPO梯度。",
    )
    parser.add_argument(
        "--fast-only-meta",
        action="store_true",
        help="外循环只学习快偏置初始化，冻结预训练Actor特征网络。",
    )
    parser.add_argument(
        "--shared-meta-offset",
        action="store_true",
        help="把五个快偏置的外更新约束为同一个共享方向。",
    )
    parser.add_argument(
        "--fresh-meta-actor",
        action="store_true",
        help="元Actor从倍率1.0先验开始；普通PPO检查点仅作为独立基线。",
    )
    parser.add_argument(
        "--meta-actor-initialization",
        choices=("static", "residual", "random"),
        default="static",
        help=(
            "fresh-meta-actor启用时，选择精确静态先验、围绕先验的状态残差，"
            "或不注入建议量先验的标准随机网络。"
        ),
    )
    parser.add_argument("--residual-head-gain", type=float, default=0.15)
    parser.add_argument(
        "--reward-mode",
        choices=("legacy", "deficit"),
        default="legacy",
        help="legacy保留原奖励；deficit使用相对初始共识缺口的连续进展奖励。",
    )
    parser.add_argument("--deficit-progress-weight", type=float, default=1.0)
    parser.add_argument("--deficit-modification-cost", type=float, default=1.5)
    parser.add_argument("--deficit-round-cost", type=float, default=0.01)
    parser.add_argument("--deficit-success-bonus", type=float, default=0.25)
    parser.add_argument("--deficit-timeout-penalty", type=float, default=0.25)
    parser.add_argument("--deficit-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--recommendation-cost-weight", type=float, default=0.01)
    parser.add_argument(
        "--remaining-deficit-cost-weight",
        type=float,
        default=0.05,
        help=(
            "逐轮归一化剩余共识缺口的面积惩罚；0.05在不改变最优固定动作的"
            "前提下，为成功轨迹保留连续的收敛速度差异。"
        ),
    )
    parser.add_argument(
        "--unexecuted-recommendation-cost-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seed", type=int, default=7371)
    parser.add_argument("--task-split-seed", type=int, default=2026)
    parser.add_argument("--validation-case-seed", type=int, default=41001)
    parser.add_argument("--test-case-seed", type=int, default=51001)
    return parser.parse_args()


def _resolve_adaptation_hyperparameters(arguments: argparse.Namespace) -> None:
    """为响应弹性直接建议任务启用低方差的一步证据校准。"""

    calibrated_direct_elasticity = (
        arguments.task_mode == "elasticity" and arguments.guidance_mode == "direct"
    )
    if arguments.inner_learning_rate is None:
        arguments.inner_learning_rate = (
            0.8 if calibrated_direct_elasticity else 0.01
        )
    if arguments.calibration_coefficient is None:
        arguments.calibration_coefficient = (
            1.0 if calibrated_direct_elasticity else 0.0
        )
    if arguments.policy_gradient_coefficient is None:
        arguments.policy_gradient_coefficient = (
            0.05 if calibrated_direct_elasticity else 1.0
        )


def _trainer(
    config: dict[str, object],
    device: torch.device,
    *,
    actor_initialization: str = "static",
    residual_head_gain: float = 0.15,
    preferred_multiplier: float = 1.0,
):
    return create_continuous_trainer(
        config,
        device,
        learning_rate=3.0e-4,
        entropy_coefficient=1.0e-4,
        minibatch_size=64,
        include_expert_identity=True,
        preferred_multiplier=preferred_multiplier,
        actor_initialization=actor_initialization,
        residual_head_gain=residual_head_gain,
    )


def _save_figure(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory / f"{name}.png", dpi=220)
    figure.savefig(directory / f"{name}.pdf")
    plt.close(figure)


def _plot_series(
    axis: plt.Axes,
    x: list[int],
    values: list[float],
    *,
    label: str,
    color: str,
) -> None:
    axis.plot(x, values, color=color, alpha=0.35, linewidth=1.0)
    axis.scatter(x, values, color=color, s=18, alpha=0.75)
    if len(values) >= 3:
        smooth = np.convolve(values, np.ones(3) / 3.0, mode="valid")
        axis.plot(x[2:], smooth, color=color, linewidth=2.0, label=f"{label}（3点均值）")
    else:
        axis.plot(x, values, color=color, linewidth=2.0, label=label)


def _plot(
    training: list[dict[str, object]],
    validation: list[dict[str, object]],
    directory: Path,
    task_label: str,
) -> None:
    configure_plot_style()
    vx = [int(item["cumulative_environment_episodes"]) for item in validation]

    # 每次元外循环都有一个训练查询集均值，因此点数等于真实元更新次数。
    tx = [int(item["iteration"]) for item in training]
    training_query_reward = [
        float(
            np.mean(
                [
                    float(task["query_rollout"]["mean_episode_reward"])
                    for task in item["tasks"]
                ]
            )
        )
        for item in training
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    _plot_series(
        axis,
        tx,
        training_query_reward,
        label="训练任务查询回报",
        color="#009E73",
    )
    axis.set(
        title=f"{task_label}元训练：查询总回报",
        xlabel="元外循环次数",
        ylabel="平均总回报",
    )
    axis.grid(alpha=0.22)
    axis.legend()
    _save_figure(figure, directory, "00_training_query_reward")

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    adapted_reward = [
        float(item["validation"]["adapted"]["mean_total_reward"])
        for item in validation
    ]
    zero_reward = [
        float(item["validation"]["zero_step"]["mean_total_reward"])
        for item in validation
    ]
    _plot_series(axis, vx, adapted_reward, label="适应后", color="#D55E00")
    axis.plot(vx, zero_reward, color="#0072B2", marker="o", label="零步适应")
    axis.set(
        title=f"留出{task_label}任务：查询总回报",
        xlabel="累计训练 episode",
        ylabel="平均总回报",
    )
    axis.grid(alpha=0.22)
    axis.legend()
    _save_figure(figure, directory, "01_validation_reward")

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    adapted_success = [
        float(item["validation"]["adapted"]["success_rate"])
        for item in validation
    ]
    zero_success = [
        float(item["validation"]["zero_step"]["success_rate"])
        for item in validation
    ]
    _plot_series(axis, vx, adapted_success, label="适应后", color="#D55E00")
    axis.plot(vx, zero_success, color="#0072B2", marker="o", label="零步适应")
    axis.set(
        title=f"留出{task_label}任务：共识成功率",
        xlabel="累计训练 episode",
        ylabel="成功率",
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.22)
    axis.legend()
    _save_figure(figure, directory, "02_validation_success")

    epoch_records = [
        item.get("meta_update_epochs", [item["meta_update"]]) for item in training
    ]
    common_epoch_count = min(len(records) for records in epoch_records)
    epoch_x = np.arange(1, common_epoch_count + 1)

    def epoch_values(name: str) -> np.ndarray:
        return np.asarray(
            [
                [float(record[name]) for record in records[:common_epoch_count]]
                for records in epoch_records
            ],
            dtype=np.float64,
        )

    def plot_epoch_band(
        axis: plt.Axes,
        values: np.ndarray,
        *,
        label: str,
        color: str,
    ) -> None:
        mean = values.mean(axis=0)
        std = (
            values.std(axis=0, ddof=1)
            if values.shape[0] > 1
            else np.zeros_like(mean)
        )
        axis.plot(epoch_x, mean, marker="o", color=color, linewidth=2.0, label=label)
        axis.fill_between(epoch_x, mean - std, mean + std, color=color, alpha=0.16)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    plot_epoch_band(
        axes[0],
        epoch_values("mean_query_actor_loss"),
        label="跨元更新均值±标准差",
        color="#009E73",
    )
    plot_epoch_band(
        axes[1],
        epoch_values("mean_query_critic_loss"),
        label="跨元更新均值±标准差",
        color="#CC79A7",
    )
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].set(title="查询集Actor目标", xlabel="单批PPO外更新 epoch", ylabel="损失")
    axes[1].set(title="查询集Critic均方误差", xlabel="单批PPO外更新 epoch", ylabel="损失")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend()
    _save_figure(figure, directory, "03_actor_critic_losses")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    plot_epoch_band(
        axes[0],
        epoch_values("mean_query_approximate_kl"),
        label="近似KL",
        color="#E69F00",
    )
    plot_epoch_band(
        axes[1],
        epoch_values("mean_query_clip_fraction"),
        label="裁剪比例",
        color="#56B4E9",
    )
    axes[0].axhline(0.03, color="#222222", linewidth=0.8, linestyle="--", label="早停阈值")
    axes[0].set(title="旧策略偏移诊断", xlabel="单批PPO外更新 epoch", ylabel="近似KL")
    axes[1].set(title="PPO裁剪诊断", xlabel="单批PPO外更新 epoch", ylabel="裁剪比例")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend()
    _save_figure(figure, directory, "04_meta_diagnostics")


def _evaluate(
    trainer,
    config: dict[str, object],
    tasks,
    *,
    support_episodes: int,
    query_episodes: int,
    inner_learning_rate: float,
    evaluation_seed: int,
    calibration_coefficient: float,
    policy_gradient_coefficient: float,
    evaluator: Callable[..., dict[str, object]],
) -> dict[str, object]:
    return evaluator(
        trainer,
        config,
        tasks,
        inner_steps=1,
        support_episodes=support_episodes,
        query_episodes=query_episodes,
        inner_learning_rate=inner_learning_rate,
        evaluation_seed=evaluation_seed,
        calibration_coefficient=calibration_coefficient,
        policy_gradient_coefficient=policy_gradient_coefficient,
    )


def main() -> None:
    configure_console_utf8()
    arguments = parse_arguments()
    _resolve_adaptation_hyperparameters(arguments)
    checkpoint = arguments.checkpoint.resolve()
    counts = (
        arguments.meta_iterations,
        arguments.meta_batch_size,
        arguments.support_episodes,
        arguments.query_episodes,
        arguments.validation_query_episodes,
        arguments.test_query_episodes,
        arguments.validation_interval,
        arguments.outer_update_epochs,
    )
    if arguments.guidance_mode == "static_optimizer" and not checkpoint.is_file():
        raise FileNotFoundError(f"找不到逐专家连续PPO检查点：{checkpoint}")
    if any(value <= 0 for value in counts):
        raise ValueError("全部训练、支持、查询和验证计数必须为正整数。")
    if (
        arguments.early_stopping_patience < 0
        or arguments.early_stopping_min_iterations < 0
    ):
        raise ValueError("早停耐心与最小训练次数不能为负。")
    if arguments.inner_learning_rate <= 0.0 or arguments.meta_learning_rate <= 0.0:
        raise ValueError("内外循环学习率必须为正。")
    if (
        arguments.calibration_coefficient < 0.0
        or arguments.policy_gradient_coefficient < 0.0
    ):
        raise ValueError("响应证据与PPO梯度系数不能为负。")
    if not 0.0 < arguments.residual_head_gain <= 1.0:
        raise ValueError("残差均值头增益必须位于(0, 1]。")
    if not (
        0.0
        < arguments.direct_action_low
        < arguments.direct_initial_recommendation
        < arguments.direct_action_high
        <= 1.0
    ):
        raise ValueError("直接建议下界、初始量和上界必须依次位于(0,1]。")
    if arguments.guidance_mode == "direct" and not arguments.fresh_meta_actor:
        raise ValueError("直接建议demo必须启用fresh-meta-actor，不能加载倍率策略。")
    if arguments.balanced_elasticity_batches and arguments.task_mode != "elasticity":
        raise ValueError("对称弹性元批次只适用于elasticity任务模式。")
    deficit_values = (
        arguments.deficit_progress_weight,
        arguments.deficit_modification_cost,
        arguments.deficit_round_cost,
        arguments.deficit_success_bonus,
        arguments.deficit_timeout_penalty,
        arguments.deficit_epsilon,
    )
    if any(value <= 0.0 for value in deficit_values):
        raise ValueError("连续缺口奖励的全部系数必须为正。")
    auxiliary_costs = (
        arguments.recommendation_cost_weight,
        arguments.remaining_deficit_cost_weight,
        arguments.unexecuted_recommendation_cost_weight,
    )
    if any(value < 0.0 for value in auxiliary_costs):
        raise ValueError("三个建议相关奖励代价不能为负。")

    set_global_seed(arguments.seed)
    config = load_config(PROJECT_ROOT / "configs" / "frozen_v1.yaml")
    base_digest = config_hash(config)
    if base_digest != "3583d0aeb2fa":
        raise RuntimeError(
            f"训练必须以冻结环境3583d0aeb2fa为基础，当前为{base_digest}。"
        )
    if arguments.reward_mode == "deficit":
        config["reward"] = dict(config["reward"])
        config["reward"].update(
            {
                "mode": "deficit",
                "deficit_epsilon": arguments.deficit_epsilon,
                "deficit_progress_weight": arguments.deficit_progress_weight,
                "modification_cost_weight": arguments.deficit_modification_cost,
                "round_cost": arguments.deficit_round_cost,
                "success_bonus": arguments.deficit_success_bonus,
                "timeout_penalty": arguments.deficit_timeout_penalty,
                "recommendation_cost_weight": (
                    arguments.recommendation_cost_weight
                ),
                "remaining_deficit_cost_weight": (
                    arguments.remaining_deficit_cost_weight
                ),
                "unexecuted_recommendation_cost_weight": (
                    arguments.unexecuted_recommendation_cost_weight
                ),
            }
        )
    config["response"] = dict(config["response"])
    config["response"]["interpolation"] = arguments.response_interpolation
    if arguments.guidance_mode == "direct":
        config["guidance"] = {
            "mode": "direct",
            "action_bounds": [
                arguments.direct_action_low,
                arguments.direct_action_high,
            ],
            "state_signal": arguments.direct_state_signal,
        }
    experiment_digest = config_hash(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arguments.task_mode == "elasticity":
        split = (
            make_response_elasticity_ood_task_split(
                split_seed=arguments.task_split_seed,
                range_profile=arguments.elasticity_range_profile,
            )
            if arguments.task_split_mode == "range_ood"
            else make_response_elasticity_task_split(
                split_seed=arguments.task_split_seed
            )
        )
        evaluate_adaptation = evaluate_response_elasticity_adaptation
        sample_tasks = (
            sample_symmetric_response_elasticity_tasks
            if arguments.balanced_elasticity_batches
            else sample_response_elasticity_tasks
        )
        train_meta_iteration = train_response_elasticity_meta_iteration
        task_label = "响应弹性"
    else:
        split = make_group_receptiveness_task_split(
            split_seed=arguments.task_split_seed
        )
        evaluate_adaptation = evaluate_group_receptiveness_adaptation
        sample_tasks = sample_group_receptiveness_tasks
        train_meta_iteration = train_group_receptiveness_meta_iteration
        task_label = "群体接纳度"
    if arguments.meta_batch_size > len(split.train):
        raise ValueError("元批次大小超过当前训练任务数。")

    preferred_multiplier = (
        arguments.direct_initial_recommendation
        if arguments.guidance_mode == "direct"
        else 1.0
    )
    ordinary = _trainer(
        config,
        device,
        preferred_multiplier=preferred_multiplier,
    )
    if arguments.guidance_mode == "static_optimizer":
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if "expert_mean_offsets" not in checkpoint_payload["actor"]:
            raise ValueError("群体共享快适应要求带专家编号的连续Actor检查点。")
        ordinary.load_checkpoint(checkpoint)
    meta = _trainer(
        config,
        device,
        actor_initialization=arguments.meta_actor_initialization,
        residual_head_gain=arguments.residual_head_gain,
        preferred_multiplier=preferred_multiplier,
    )
    if not arguments.fresh_meta_actor:
        meta.actor.load_state_dict(ordinary.actor.state_dict())
    # 旧奖励下的Critic不能作为新奖励的价值先验，否则初始价值标尺不一致。
    meta_critic_initialization = "random"
    if (
        arguments.reward_mode == "legacy"
        and arguments.guidance_mode == "static_optimizer"
    ):
        meta.critic.load_state_dict(ordinary.critic.state_dict())
        meta_critic_initialization = "ordinary_ppo_checkpoint"

    run_dir = create_run_directory(
        config,
        PROJECT_ROOT,
        stage=f"{arguments.task_mode}_maml",
    )
    argument_record = dict(vars(arguments))
    argument_record["checkpoint"] = str(checkpoint)
    write_json(argument_record, run_dir / "arguments.json")
    write_json(split.to_serializable(), run_dir / "task_split.json")
    checkpoint_stem = f"{arguments.task_mode}_maml"
    initial_checkpoint = meta.save_checkpoint(
        run_dir / f"initial_{checkpoint_stem}.pt",
        {"role": "before_meta_training", "source_checkpoint": str(checkpoint)},
    )

    initial_validation = _evaluate(
        meta,
        config,
        split.validation,
        support_episodes=arguments.support_episodes,
        query_episodes=arguments.validation_query_episodes,
        inner_learning_rate=arguments.inner_learning_rate,
        evaluation_seed=arguments.validation_case_seed,
        calibration_coefficient=arguments.calibration_coefficient,
        policy_gradient_coefficient=arguments.policy_gradient_coefficient,
        evaluator=evaluate_adaptation,
    )
    validation_records = [
        {
            "iteration": 0,
            "cumulative_environment_steps": 0,
            "cumulative_environment_episodes": 0,
            "validation": initial_validation,
        }
    ]
    best_reward = float(initial_validation["adapted"]["mean_total_reward"])
    best_iteration = 0
    best_checkpoint = meta.save_checkpoint(
        run_dir / f"best_{checkpoint_stem}.pt",
        {"best_iteration": 0, "validation": initial_validation},
    )
    latest_checkpoint = run_dir / f"latest_{checkpoint_stem}.pt"
    optimizer = MetaGradientOptimizer(
        meta,
        meta_learning_rate=arguments.meta_learning_rate,
        actor_fast_only=arguments.fast_only_meta,
        shared_actor_fast_update=arguments.shared_meta_offset,
    )
    train_rng = np.random.default_rng(arguments.seed + 200)
    training_records: list[dict[str, object]] = []
    cumulative_steps = 0
    cumulative_episodes = 0
    completed_iterations = 0
    validations_without_improvement = 0
    stopped_early = False
    for iteration in range(1, arguments.meta_iterations + 1):
        tasks = sample_tasks(
            split.train,
            arguments.meta_batch_size,
            train_rng,
        )
        record = train_meta_iteration(
            meta,
            optimizer,
            config,
            tasks,
            support_episodes=arguments.support_episodes,
            query_episodes=arguments.query_episodes,
            inner_learning_rate=arguments.inner_learning_rate,
            iteration_seed=int(
                train_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
            ),
            paired_scenarios=True,
            calibration_coefficient=arguments.calibration_coefficient,
            policy_gradient_coefficient=arguments.policy_gradient_coefficient,
            outer_update_epochs=arguments.outer_update_epochs,
            second_order_fast=arguments.second_order_fast,
        )
        cumulative_steps += int(record["environment_steps"])
        cumulative_episodes += int(record["environment_episodes"])
        record["iteration"] = iteration
        record["cumulative_environment_steps"] = cumulative_steps
        record["cumulative_environment_episodes"] = cumulative_episodes
        record["diagnostics"] = {
            "mean_inner_actor_gradient_norm": float(
                np.mean(
                    [
                        float(task["inner_update"]["actor_gradient_norm"])
                        for task in record["tasks"]
                    ]
                )
            ),
            "mean_absolute_shared_offset_change": float(
                np.mean(
                    [
                        abs(float(task["shared_offset_change"]))
                        for task in record["tasks"]
                    ]
                )
            ),
        }
        training_records.append(record)
        append_jsonl(run_dir / "training.jsonl", record)
        completed_iterations = iteration

        if (
            iteration % arguments.validation_interval == 0
            or iteration == arguments.meta_iterations
        ):
            validation = _evaluate(
                meta,
                config,
                split.validation,
                support_episodes=arguments.support_episodes,
                query_episodes=arguments.validation_query_episodes,
                inner_learning_rate=arguments.inner_learning_rate,
                evaluation_seed=arguments.validation_case_seed,
                calibration_coefficient=arguments.calibration_coefficient,
                policy_gradient_coefficient=arguments.policy_gradient_coefficient,
                evaluator=evaluate_adaptation,
            )
            validation_record = {
                "iteration": iteration,
                "cumulative_environment_steps": cumulative_steps,
                "cumulative_environment_episodes": cumulative_episodes,
                "validation": validation,
            }
            validation_records.append(validation_record)
            append_jsonl(run_dir / "validation.jsonl", validation_record)
            reward = float(validation["adapted"]["mean_total_reward"])
            if reward > best_reward:
                best_reward = reward
                best_iteration = iteration
                validations_without_improvement = 0
                meta.save_checkpoint(
                    best_checkpoint,
                    {"best_iteration": iteration, "validation": validation},
                )
            else:
                validations_without_improvement += 1
            meta.save_checkpoint(
                latest_checkpoint,
                {"latest_iteration": iteration, "validation": validation},
            )
            print(
                f"{task_label}MAML {iteration}/{arguments.meta_iterations}: "
                f"适应后回报={reward:.5f}, "
                f"适应增益={validation['adaptation_gain']['mean_total_reward']:+.5f}, "
                f"成功率={validation['adapted']['success_rate']:.3f}",
                flush=True,
            )
            if (
                arguments.early_stopping_patience > 0
                and iteration >= arguments.early_stopping_min_iterations
                and validations_without_improvement
                >= arguments.early_stopping_patience
                and iteration < arguments.meta_iterations
            ):
                stopped_early = True
                print(
                    f"验证回报连续{validations_without_improvement}次未刷新最佳值，"
                    f"在第{iteration}次元外循环早停。",
                    flush=True,
                )
                break

    final_checkpoint = meta.save_checkpoint(
        run_dir / f"final_{checkpoint_stem}.pt",
        {
            "final_iteration": completed_iterations,
            "training_environment_steps": cumulative_steps,
            "training_environment_episodes": cumulative_episodes,
        },
    )
    best = _trainer(
        config,
        device,
        actor_initialization=arguments.meta_actor_initialization,
        residual_head_gain=arguments.residual_head_gain,
        preferred_multiplier=preferred_multiplier,
    )
    best.load_checkpoint(best_checkpoint)
    before_meta = _trainer(
        config,
        device,
        actor_initialization=arguments.meta_actor_initialization,
        residual_head_gain=arguments.residual_head_gain,
        preferred_multiplier=preferred_multiplier,
    )
    before_meta.load_checkpoint(initial_checkpoint)
    ordinary_test = _evaluate(
        ordinary,
        config,
        split.test,
        support_episodes=arguments.support_episodes,
        query_episodes=arguments.test_query_episodes,
        inner_learning_rate=arguments.inner_learning_rate,
        evaluation_seed=arguments.test_case_seed,
        calibration_coefficient=arguments.calibration_coefficient,
        policy_gradient_coefficient=arguments.policy_gradient_coefficient,
        evaluator=evaluate_adaptation,
    )
    best_test = _evaluate(
        best,
        config,
        split.test,
        support_episodes=arguments.support_episodes,
        query_episodes=arguments.test_query_episodes,
        inner_learning_rate=arguments.inner_learning_rate,
        evaluation_seed=arguments.test_case_seed,
        calibration_coefficient=arguments.calibration_coefficient,
        policy_gradient_coefficient=arguments.policy_gradient_coefficient,
        evaluator=evaluate_adaptation,
    )
    before_meta_test = _evaluate(
        before_meta,
        config,
        split.test,
        support_episodes=arguments.support_episodes,
        query_episodes=arguments.test_query_episodes,
        inner_learning_rate=arguments.inner_learning_rate,
        evaluation_seed=arguments.test_case_seed,
        calibration_coefficient=arguments.calibration_coefficient,
        policy_gradient_coefficient=arguments.policy_gradient_coefficient,
        evaluator=evaluate_adaptation,
    )
    comparison = {
        "ordinary_initialization": ordinary_test,
        "before_meta_training": before_meta_test,
        "maml_initialization": best_test,
        "maml_minus_ordinary_after_adaptation": {
            "mean_total_reward": float(best_test["adapted"]["mean_total_reward"])
            - float(ordinary_test["adapted"]["mean_total_reward"]),
            "success_rate": float(best_test["adapted"]["success_rate"])
            - float(ordinary_test["adapted"]["success_rate"]),
            "mean_rounds": float(ordinary_test["adapted"]["mean_rounds"])
            - float(best_test["adapted"]["mean_rounds"]),
        },
        "maml_minus_ordinary_without_adaptation": {
            "mean_total_reward": float(best_test["adapted"]["mean_total_reward"])
            - float(ordinary_test["zero_step"]["mean_total_reward"]),
            "success_rate": float(best_test["adapted"]["success_rate"])
            - float(ordinary_test["zero_step"]["success_rate"]),
        },
        "maml_minus_before_meta_after_adaptation": {
            "mean_total_reward": float(best_test["adapted"]["mean_total_reward"])
            - float(before_meta_test["adapted"]["mean_total_reward"]),
            "success_rate": float(best_test["adapted"]["success_rate"])
            - float(before_meta_test["adapted"]["success_rate"]),
        },
    }
    gates = {
        "meta_adaptation_reward_gain_positive": (
            float(best_test["adaptation_gain"]["mean_total_reward"]) > 0.0
        ),
        "meta_adapted_above_ordinary_adapted": (
            float(
                comparison["maml_minus_ordinary_after_adaptation"][
                    "mean_total_reward"
                ]
            )
            > 0.0
        ),
        "meta_adapted_above_ordinary_unadapted": (
            float(
                comparison["maml_minus_ordinary_without_adaptation"][
                    "mean_total_reward"
                ]
            )
            > 0.0
        ),
        "meta_training_improves_adapted_reward": (
            float(
                comparison["maml_minus_before_meta_after_adaptation"][
                    "mean_total_reward"
                ]
            )
            > 0.0
        ),
        "optimizer_failure_zero": (
            float(best_test["adapted"]["optimizer_failure_rate"]) == 0.0
        ),
    }
    summary = {
        "run_dir": str(run_dir),
        "base_environment_hash": base_digest,
        "experiment_config_hash": experiment_digest,
        "reward_mode": arguments.reward_mode,
        "guidance_mode": arguments.guidance_mode,
        "meta_actor_initialization": arguments.meta_actor_initialization,
        "meta_critic_initialization": meta_critic_initialization,
        "device": str(device),
        "task_mode": arguments.task_mode,
        "meta_gradient_order": (
            "second" if arguments.second_order_fast else "first"
        ),
        "best_iteration": best_iteration,
        "requested_meta_iterations": arguments.meta_iterations,
        "completed_meta_iterations": completed_iterations,
        "early_stopping": {
            "enabled": arguments.early_stopping_patience > 0,
            "patience": arguments.early_stopping_patience,
            "minimum_iterations": arguments.early_stopping_min_iterations,
            "stopped_early": stopped_early,
            "validations_without_improvement": validations_without_improvement,
        },
        "initial_checkpoint": str(initial_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "training_environment_steps": cumulative_steps,
        "training_environment_episodes": cumulative_episodes,
        "heldout_comparison": comparison,
        "gates": gates,
        "decision": "GO" if all(gates.values()) else "REVIEW",
    }
    write_json(training_records, run_dir / "training.json")
    write_json(validation_records, run_dir / "validation.json")
    write_json(comparison, run_dir / "heldout_comparison.json")
    write_json(summary, run_dir / "summary.json")
    _plot(
        training_records,
        validation_records,
        run_dir / "figures",
        task_label,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
