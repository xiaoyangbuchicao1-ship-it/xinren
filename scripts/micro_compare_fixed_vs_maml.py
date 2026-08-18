"""在完全相同的OOD案例上逐轮比较固定建议量与适应后的FOMAML-PPO。"""

from __future__ import annotations

import argparse
import csv
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

from src.analysis.plot_style import configure_plot_style
from src.common.config import load_config
from src.common.encoding import configure_console_utf8, write_json
from src.env.consensus_env import ConsensusFeedbackEnv
from src.experiments.continuous_ppo import (
    create_continuous_trainer,
    evaluate_continuous_policy_on_cases,
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


ActionSelector = Callable[[np.ndarray, ConsensusFeedbackEnv], np.ndarray]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_seeds(
    test_tasks: list[dict[str, float]],
    target: float,
    evaluation_seed: int,
) -> tuple[int, int, int, int]:
    """重放正式测试的种子流，保证微观案例与最终OOD测试一致。"""

    rng = np.random.default_rng(evaluation_seed)
    for record in test_tasks:
        values = tuple(
            int(value)
            for value in rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=4,
                dtype=np.uint32,
            )
        )
        if np.isclose(float(record["magnitude_sensitivity"]), target):
            return values
    raise ValueError(f"测试任务中不存在响应弹性 {target}。")


def _trace_episode(
    config: dict[str, Any],
    case: ValidationCase,
    selector: ActionSelector,
) -> dict[str, Any]:
    """保留每一轮的动作、响应、实际调整和共识变化。"""

    env = ConsensusFeedbackEnv(
        config,
        np.random.default_rng(case.response_seed),
        response_types=case.response_types,
    )
    state, reset_info = env.reset(case.instance)
    initial_acd = env.metrics.acd.copy()
    initial_opinions = env.current_opinions.copy()
    rounds: list[dict[str, Any]] = []
    total_reward = 0.0

    while not env.done:
        before_acd = env.metrics.acd.copy()
        before_opinions = env.current_opinions.copy()
        actions = np.asarray(selector(state, env), dtype=np.float64)
        state, reward, _, _, info = env.step_continuous(actions)
        active = np.asarray(info["active_expert_mask"], dtype=bool)
        recommended = np.asarray(info["recommended_deltas"], dtype=np.float64)
        response_rates = np.asarray(info["response_rates"], dtype=np.float64)
        effective = np.asarray(info["effective_deltas"], dtype=np.float64)
        total_reward += float(reward)
        rounds.append(
            {
                "round": int(info["round"]),
                "active_experts_1_based": (np.flatnonzero(active) + 1).tolist(),
                "actions": actions.tolist(),
                "recommended_deltas": recommended.tolist(),
                "response_rates": response_rates.tolist(),
                "effective_deltas": effective.tolist(),
                "active_recommendation_mean": (
                    float(recommended[active].mean()) if np.any(active) else 0.0
                ),
                "active_response_rate_mean": (
                    float(response_rates[active].mean()) if np.any(active) else 0.0
                ),
                "active_effective_delta_mean": (
                    float(effective[active].mean()) if np.any(active) else 0.0
                ),
                "before_acd": before_acd.tolist(),
                "after_acd": env.metrics.acd.tolist(),
                "before_min_acd": float(before_acd.min()),
                "after_min_acd": float(env.metrics.min_acd),
                "before_mean_acd": float(before_acd.mean()),
                "after_mean_acd": float(env.metrics.mean_acd),
                "before_fused_opinions": before_opinions.tolist(),
                "after_fused_opinions": env.current_opinions.tolist(),
                "reward": float(reward),
                "reward_breakdown": dict(info["reward"]),
                "success": bool(info["success"]),
                "timeout": bool(info["timeout"]),
            }
        )

    return {
        "initial_success": bool(reset_info["initial_success"]),
        "response_types": list(case.response_types),
        "initial_acd": initial_acd.tolist(),
        "initial_min_acd": float(initial_acd.min()),
        "initial_mean_acd": float(initial_acd.mean()),
        "initial_fused_opinions": initial_opinions.tolist(),
        "success": bool(env.success),
        "round_count": int(env.round_index),
        "total_reward": total_reward,
        "final_acd": env.metrics.acd.tolist(),
        "final_min_acd": float(env.metrics.min_acd),
        "final_mean_acd": float(env.metrics.mean_acd),
        "rounds": rounds,
    }


def _write_round_csv(
    path: Path,
    traces: dict[str, dict[str, Any]],
    sensitivity: float,
    case_index: int,
) -> None:
    fields = [
        "sensitivity",
        "case_index",
        "strategy",
        "round",
        "expert",
        "active",
        "action",
        "recommended_delta",
        "response_rate",
        "effective_delta",
        "before_acd",
        "after_acd",
        "round_reward",
        "success",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for strategy, trace in traces.items():
            for round_record in trace["rounds"]:
                active = set(round_record["active_experts_1_based"])
                for expert in range(1, len(round_record["actions"]) + 1):
                    index = expert - 1
                    writer.writerow(
                        {
                            "sensitivity": sensitivity,
                            "case_index": case_index,
                            "strategy": strategy,
                            "round": round_record["round"],
                            "expert": expert,
                            "active": expert in active,
                            "action": round_record["actions"][index],
                            "recommended_delta": round_record["recommended_deltas"][index],
                            "response_rate": round_record["response_rates"][index],
                            "effective_delta": round_record["effective_deltas"][index],
                            "before_acd": round_record["before_acd"][index],
                            "after_acd": round_record["after_acd"][index],
                            "round_reward": round_record["reward"],
                            "success": round_record["success"],
                        }
                    )


def _plot(path: Path, traces: dict[str, dict[str, Any]]) -> None:
    configure_plot_style()
    fixed_colors = ["#E69F00", "#D55E00", "#CC79A7", "#7A5195"]
    fixed_names = [name for name in traces if name.startswith("固定建议量")]
    colors = {
        name: fixed_colors[index % len(fixed_colors)]
        for index, name in enumerate(fixed_names)
    }
    colors["FOMAML-PPO一步适应"] = "#0072B2"
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    for strategy, trace in traces.items():
        color = colors[strategy]
        x_acd = [0] + [int(item["round"]) for item in trace["rounds"]]
        y_acd = [float(trace["initial_min_acd"])] + [
            float(item["after_min_acd"]) for item in trace["rounds"]
        ]
        axes[0].plot(x_acd, y_acd, marker="o", color=color, label=strategy)
        x_action = [int(item["round"]) for item in trace["rounds"]]
        y_action = [
            float(item["active_recommendation_mean"]) for item in trace["rounds"]
        ]
        axes[1].plot(x_action, y_action, marker="o", color=color, label=strategy)
    axes[0].set(title="最小专家共识度逐轮变化", xlabel="协商轮次", ylabel="最小ACD")
    axes[1].set(title="有效专家平均建议量", xlabel="协商轮次", ylabel="建议比例")
    for axis in axes:
        axis.grid(alpha=0.20)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    configure_console_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sensitivity", type=float, default=-0.35)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help="覆盖正式测试种子，用指定根种子复现独立的微观案例集合。",
    )
    parser.add_argument("--fixed-action", type=float, default=0.5)
    parser.add_argument(
        "--fixed-actions",
        type=float,
        nargs="+",
        default=None,
        help="一次比较多个固定建议量；设置后覆盖--fixed-action。",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="附加到输出目录名的可选标签，用于保留不同评估协议的结果。",
    )
    arguments = parser.parse_args()

    fixed_actions = (
        list(arguments.fixed_actions)
        if arguments.fixed_actions is not None
        else [float(arguments.fixed_action)]
    )
    if not fixed_actions or any(action < 0.0 or action > 1.0 for action in fixed_actions):
        raise ValueError("固定建议量必须位于[0, 1]区间。")
    # 保持用户给定顺序，同时删除重复值。
    fixed_actions = list(dict.fromkeys(float(action) for action in fixed_actions))

    run_dir = arguments.run_dir.resolve()
    run_arguments = _read_json(run_dir / "arguments.json")
    task_split = _read_json(run_dir / "task_split.json")
    config_paths = sorted(run_dir.glob("config_*.yaml"))
    if len(config_paths) != 1:
        raise ValueError("运行目录必须且只能包含一个冻结实验配置文件。")
    config = load_config(config_paths[0])

    test_seed = int(
        run_arguments["test_case_seed"]
        if arguments.evaluation_seed is None
        else arguments.evaluation_seed
    )
    task_seed, type_seed, response_seed, support_seed = _task_seeds(
        list(task_split["test"]),
        arguments.sensitivity,
        test_seed,
    )
    task = ResponseElasticityTask(float(arguments.sensitivity))
    task_config = config_for_response_elasticity_task(config, task)
    cases = make_validation_cases(
        task_config,
        int(run_arguments["test_query_episodes"]),
        task_seed=task_seed,
        type_seed=type_seed,
        response_seed=response_seed,
    )

    # 微观审计显式使用CPU，避免与仍在运行的GPU训练争抢资源。
    trainer = create_continuous_trainer(
        config,
        torch.device("cpu"),
        learning_rate=3.0e-4,
        entropy_coefficient=1.0e-4,
        minibatch_size=64,
        include_expert_identity=True,
        preferred_multiplier=float(run_arguments["direct_initial_recommendation"]),
        actor_initialization=str(run_arguments["meta_actor_initialization"]),
        residual_head_gain=float(run_arguments["residual_head_gain"]),
    )
    trainer.load_checkpoint(run_dir / "best_elasticity_maml.pt")
    adapted, adaptation = adapt_continuous_to_response_elasticity(
        trainer,
        config,
        task,
        inner_steps=1,
        support_episodes=int(run_arguments["support_episodes"]),
        inner_learning_rate=float(run_arguments["inner_learning_rate"]),
        support_seed=support_seed,
        calibration_coefficient=float(run_arguments["calibration_coefficient"]),
        policy_gradient_coefficient=float(run_arguments["policy_gradient_coefficient"]),
    )

    fixed_results: dict[float, tuple[dict[str, Any], list[Any]]] = {}
    for fixed_action in fixed_actions:
        fixed_results[fixed_action] = evaluate_continuous_policy_on_cases(
            task_config,
            cases,
            lambda _state, _env, action=fixed_action: np.full(
                int(config["data"]["num_experts"]),
                action,
                dtype=np.float64,
            ),
        )
    maml_summary, maml_episodes = evaluate_continuous_trainer(
        adapted,
        task_config,
        cases,
        deterministic=True,
    )

    # 用每个案例上表现最好的固定策略构造强基线，再据此选择中位代表案例。
    fixed_reward_matrix = np.asarray(
        [
            [episode.total_reward for episode in fixed_results[action][1]]
            for action in fixed_actions
        ],
        dtype=np.float64,
    )
    best_fixed_rewards_per_case = fixed_reward_matrix.max(axis=0)
    gains = np.asarray(
        [
            maml.total_reward - best_fixed_reward
            for best_fixed_reward, maml in zip(
                best_fixed_rewards_per_case,
                maml_episodes,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    median_gain = float(np.median(gains))
    case_index = int(np.argmin(np.abs(gains - median_gain)))
    selected_case = cases[case_index]

    maml_selector: ActionSelector = lambda state, _env: adapted.act(
        state,
        deterministic=True,
    )[0]
    traces: dict[str, dict[str, Any]] = {}
    for fixed_action in fixed_actions:
        fixed_selector: ActionSelector = lambda _state, _env, action=fixed_action: np.full(
            int(config["data"]["num_experts"]),
            action,
            dtype=np.float64,
        )
        traces[f"固定建议量{fixed_action:g}"] = _trace_episode(
            task_config,
            selected_case,
            fixed_selector,
        )
    traces["FOMAML-PPO一步适应"] = _trace_episode(
        task_config,
        selected_case,
        maml_selector,
    )

    sensitivity_label = str(arguments.sensitivity).replace("-", "m").replace(".", "p")
    comparison_label = (
        "fixed_grid" if len(fixed_actions) > 1 else f"fixed_{fixed_actions[0]:g}"
    ).replace(".", "p")
    output_tag = arguments.output_tag.strip()
    if output_tag:
        safe_tag = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in output_tag
        )
        comparison_label = f"{comparison_label}_{safe_tag}"
    output = (
        run_dir
        / "micro_comparison"
        / f"sensitivity_{sensitivity_label}_{comparison_label}"
    )
    output.mkdir(parents=True, exist_ok=True)
    fixed_summaries = {
        f"fixed_{str(action).replace('.', 'p')}": fixed_results[action][0]
        for action in fixed_actions
    }
    best_fixed_action = max(
        fixed_actions,
        key=lambda action: float(fixed_results[action][0]["mean_total_reward"]),
    )
    best_fixed_summary = fixed_results[best_fixed_action][0]
    result = {
        "selection_rule": (
            "从正式OOD查询案例中，按FOMAML-PPO回报减去该案例四种固定策略的最高回报，"
            "选择配对增益最接近中位数的案例"
        ),
        "task": task.to_serializable(),
        "test_case_seed": test_seed,
        "query_case_count": len(cases),
        "fixed_actions": fixed_actions,
        "selected_case_index_0_based": case_index,
        "selected_case_number_1_based": case_index + 1,
        "paired_reward_gain_vs_per_case_best_fixed_distribution": {
            "mean": float(gains.mean()),
            "median": median_gain,
            "std": float(gains.std()),
            "minimum": float(gains.min()),
            "maximum": float(gains.max()),
            "selected": float(gains[case_index]),
        },
        "support_adaptation": adaptation.to_serializable(),
        "aggregate_64_cases": {
            **fixed_summaries,
            "fomaml_ppo_adapted": maml_summary,
            "best_fixed_by_mean_reward": {
                "action": best_fixed_action,
                "mean_total_reward": best_fixed_summary["mean_total_reward"],
                "success_rate": best_fixed_summary["success_rate"],
                "mean_rounds": best_fixed_summary["mean_rounds"],
            },
            "fomaml_gain_vs_best_fixed_by_mean_reward": {
                "mean_reward_gain": float(
                    maml_summary["mean_total_reward"]
                    - best_fixed_summary["mean_total_reward"]
                ),
                "success_rate_gain": float(
                    maml_summary["success_rate"] - best_fixed_summary["success_rate"]
                ),
                "mean_round_reduction": float(
                    best_fixed_summary["mean_rounds"] - maml_summary["mean_rounds"]
                ),
            },
        },
        "selected_case_input": selected_case.instance.to_serializable(),
        "selected_case_traces": traces,
    }
    write_json(result, output / "fixed_grid_vs_fomaml_micro.json")
    _write_round_csv(
        output / "fixed_grid_vs_fomaml_rounds.csv",
        traces,
        float(arguments.sensitivity),
        case_index,
    )
    _plot(output / "fixed_grid_vs_fomaml_trace.png", traces)

    compact = {
        "output": str(output),
        "task": task.to_serializable(),
        "selected_case_number_1_based": case_index + 1,
        "aggregate_64_cases": result["aggregate_64_cases"],
        "selected_case": {
            strategy: {
                "success": trace["success"],
                "round_count": trace["round_count"],
                "total_reward": trace["total_reward"],
                "final_min_acd": trace["final_min_acd"],
            }
            for strategy, trace in traces.items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
