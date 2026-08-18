"""普通 PPO 的真实环境轨迹收集、验证与训练循环。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.networks import FactorizedActor, ValueNetwork
from src.agents.ppo import PPOTrainer, RolloutBuffer
from src.analysis.plot_style import configure_plot_style
from src.common.encoding import write_json
from src.common.logger import append_jsonl
from src.data.task_generator import StageBInstance, generate_stage_b_instance
from src.env.consensus_env import ConsensusFeedbackEnv
from src.env.response_model import (
    sample_response_types,
    sample_response_types_from_counts,
)
from src.experiments.evaluate_policies import EpisodeEvaluation, aggregate_episodes
from src.experiments.evaluate_policies import run_policy_episode


@dataclass(frozen=True)
class ValidationCase:
    """固定验证意见、隐藏响应类型和响应噪声种子。"""

    instance: StageBInstance
    response_types: tuple[str, ...]
    response_seed: int


def make_validation_cases(
    config: dict[str, Any],
    count: int,
    *,
    task_seed: int,
    type_seed: int,
    response_seed: int,
) -> list[ValidationCase]:
    """一次生成后反复复用，保证不同更新轮次的验证任务完全一致。"""

    task_rng = np.random.default_rng(task_seed)
    type_rng = np.random.default_rng(type_seed)
    response_seed_rng = np.random.default_rng(response_seed)
    response = config["response"]
    cases = []
    for _ in range(count):
        cases.append(
            ValidationCase(
                instance=generate_stage_b_instance(config, task_rng),
                response_types=sample_response_types(
                    int(config["data"]["num_experts"]),
                    response["type_names"],
                    response["type_probabilities"],
                    type_rng,
                ),
                response_seed=int(
                    response_seed_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
                ),
            )
        )
    return cases


def collect_rollout(
    trainer: PPOTrainer,
    config: dict[str, Any],
    transition_target: int | None = None,
    *,
    task_rng: np.random.Generator,
    type_rng: np.random.Generator,
    response_seed_rng: np.random.Generator,
    episode_target: int | None = None,
    fixed_response_types: tuple[str, ...] | None = None,
    fixed_response_composition: tuple[int, ...] | None = None,
) -> tuple[RolloutBuffer, dict[str, object]]:
    """按转移数或完整回合数收集 on-policy 轨迹。"""

    if (transition_target is None) == (episode_target is None):
        raise ValueError("必须且只能指定转移目标或回合目标之一。")
    if transition_target is not None and transition_target <= 0:
        raise ValueError("目标转移数必须为正整数。")
    if episode_target is not None and episode_target <= 0:
        raise ValueError("目标回合数必须为正整数。")
    if fixed_response_types is not None and len(fixed_response_types) != int(
        config["data"]["num_experts"]
    ):
        raise ValueError("固定响应类型数量必须与专家数一致。")
    if fixed_response_types is not None and fixed_response_composition is not None:
        raise ValueError("固定响应类型和固定人数构成不能同时指定。")
    if fixed_response_composition is not None:
        response_type_count = len(config["response"]["type_names"])
        if (
            len(fixed_response_composition) != response_type_count
            or any(
                not isinstance(value, (int, np.integer)) or int(value) < 0
                for value in fixed_response_composition
            )
            or sum(int(value) for value in fixed_response_composition)
            != int(config["data"]["num_experts"])
        ):
            raise ValueError("固定人数构成必须是与类型数一致、总和等于专家数的非负整数。")
    buffer = RolloutBuffer()
    episode_rewards: list[float] = []
    episode_successes: list[bool] = []
    episode_rounds: list[int] = []
    action_counts = np.zeros(len(config["response"]["multipliers"]), dtype=np.int64)
    response = config["response"]

    def collection_complete() -> bool:
        if transition_target is not None:
            return len(buffer) >= transition_target
        assert episode_target is not None
        return len(episode_rewards) >= episode_target

    while not collection_complete():
        instance = generate_stage_b_instance(config, task_rng)
        response_types = (
            fixed_response_types
            if fixed_response_types is not None
            else sample_response_types_from_counts(
                response["type_names"],
                fixed_response_composition,
                type_rng,
            )
            if fixed_response_composition is not None
            else sample_response_types(
                int(config["data"]["num_experts"]),
                response["type_names"],
                response["type_probabilities"],
                type_rng,
            )
        )
        env = ConsensusFeedbackEnv(
            config,
            np.random.default_rng(
                int(response_seed_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            ),
            response_types=response_types,
        )
        state, _ = env.reset(instance)
        episode_reward = 0.0
        while not env.done:
            action, log_probability, value = trainer.act(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            buffer.add(state, action, reward, value, log_probability, done)
            active = np.asarray(info["theoretical_deltas"]) > 1.0e-12
            action_counts += np.bincount(action[active], minlength=action_counts.size)
            episode_reward += reward
            state = next_state
        episode_rewards.append(float(episode_reward))
        episode_successes.append(env.success)
        episode_rounds.append(env.round_index)

    total_actions = int(action_counts.sum())
    summary = {
        "transitions": len(buffer),
        "episodes": len(episode_rewards),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "success_rate": float(np.mean(episode_successes)),
        "mean_rounds": float(np.mean(episode_rounds)),
        "active_action_counts": action_counts.tolist(),
        "active_action_proportions": (
            (action_counts / total_actions).tolist()
            if total_actions
            else np.zeros(action_counts.size).tolist()
        ),
    }
    return buffer, summary


def _evaluate_trainer_once(
    trainer: PPOTrainer,
    config: dict[str, Any],
    cases: list[ValidationCase],
    *,
    deterministic: bool,
) -> tuple[dict[str, object], list[EpisodeEvaluation]]:
    """在固定验证回合上执行一次评价。"""

    episodes: list[EpisodeEvaluation] = []
    for case in cases:
        env = ConsensusFeedbackEnv(
            config,
            np.random.default_rng(case.response_seed),
            response_types=case.response_types,
        )
        state, reset_info = env.reset(case.instance)
        total_reward = 0.0
        total_modification = 0.0
        component_totals = {
            "consensus_improvement": 0.0,
            "modification_cost": 0.0,
            "round_cost": 0.0,
            "success_bonus": 0.0,
            "timeout_penalty": 0.0,
        }
        action_counts = np.zeros(env.action_count, dtype=np.int64)
        timeout = False
        optimizer_failed = not bool(reset_info["optimizer_success"])
        while not env.done:
            action, _, _ = trainer.act(state, deterministic=deterministic)
            state, reward, _, _, info = env.step(action)
            total_reward += float(reward)
            total_modification += float(info["reward"]["mean_modification"])
            for key in component_totals:
                component_totals[key] += float(info["reward"][key])
            active = np.asarray(info["theoretical_deltas"]) > 1.0e-12
            action_counts += np.bincount(action[active], minlength=env.action_count)
            timeout = bool(info["timeout"])
            optimizer_failed = bool(info["optimizer_failed"])
        episodes.append(
            EpisodeEvaluation(
                initial_success=bool(reset_info["initial_success"]),
                success=env.success,
                timeout=timeout,
                optimizer_failed=optimizer_failed,
                rounds=env.round_index,
                total_reward=total_reward,
                total_modification=total_modification,
                total_consensus_improvement=component_totals["consensus_improvement"],
                total_modification_cost=component_totals["modification_cost"],
                total_round_cost=component_totals["round_cost"],
                total_success_bonus=component_totals["success_bonus"],
                total_timeout_penalty=component_totals["timeout_penalty"],
                active_action_counts=tuple(int(value) for value in action_counts),
            )
        )
    return aggregate_episodes(episodes), episodes


def evaluate_trainer(
    trainer: PPOTrainer,
    config: dict[str, Any],
    cases: list[ValidationCase],
    *,
    deterministic: bool,
    action_seed: int | None = None,
) -> tuple[dict[str, object], list[EpisodeEvaluation]]:
    """评价当前策略，并可隔离随机动作采样流以保证重复验证可比。"""

    if deterministic or action_seed is None:
        return _evaluate_trainer_once(
            trainer,
            config,
            cases,
            deterministic=deterministic,
        )
    cuda_devices: list[int] = []
    if trainer.device.type == "cuda":
        cuda_devices.append(
            torch.cuda.current_device()
            if trainer.device.index is None
            else trainer.device.index
        )
    # fork_rng 在退出时恢复训练随机状态，验证采样不会改变后续 on-policy 轨迹。
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(action_seed))
        return _evaluate_trainer_once(
            trainer,
            config,
            cases,
            deterministic=False,
        )


def validation_selection_score(deterministic_validation: dict[str, object]) -> float:
    """中心策略最终确定性执行，因此检查点按确定性验证回报选择。"""

    score = float(deterministic_validation["mean_total_reward"])
    if not np.isfinite(score):
        raise FloatingPointError("确定性验证回报不是有限值。")
    return score


def evaluate_policy_on_cases(
    config: dict[str, Any],
    cases: list[ValidationCase],
    policy_factory: Callable[[int], object],
) -> tuple[dict[str, object], list[EpisodeEvaluation]]:
    """在与学习策略完全相同的固定任务上评价一个非学习策略。"""

    episodes: list[EpisodeEvaluation] = []
    for index, case in enumerate(cases):
        env = ConsensusFeedbackEnv(
            config,
            np.random.default_rng(case.response_seed),
            response_types=case.response_types,
        )
        episodes.append(run_policy_episode(env, case.instance, policy_factory(index)))
    return aggregate_episodes(episodes), episodes


def create_trainer(
    config: dict[str, Any],
    device: torch.device,
    *,
    learning_rate: float | None = None,
    entropy_coefficient: float | None = None,
    minibatch_size: int | None = None,
    preferred_action_index: int | None = None,
    preferred_action_probability: float = 0.255,
) -> PPOTrainer:
    """依据冻结配置与少量显式训练覆盖项创建 PPO。"""

    ppo = config["ppo"]
    state_dim = int(config["data"]["num_experts"]) * 6 + 3
    actor = FactorizedActor(
        state_dim,
        int(config["data"]["num_experts"]),
        len(config["response"]["multipliers"]),
        ppo["hidden_sizes"],
        ppo["activation"],
        multipliers=config["response"]["multipliers"],
        suggestion_bins=config["response"]["suggestion_bins"],
    ).to(device)
    if preferred_action_index is not None:
        actor.initialize_action_prior(
            preferred_action_index,
            preferred_action_probability,
        )
    critic = ValueNetwork(state_dim, ppo["hidden_sizes"], ppo["activation"]).to(device)
    return PPOTrainer(
        actor,
        critic,
        learning_rate=(
            float(ppo["learning_rate"]) if learning_rate is None else learning_rate
        ),
        clip_range=float(ppo["clip_range"]),
        update_epochs=int(ppo["update_epochs"]),
        minibatch_size=(
            int(ppo["minibatch_size"]) if minibatch_size is None else minibatch_size
        ),
        entropy_coefficient=(
            float(ppo["entropy_coefficient"])
            if entropy_coefficient is None
            else entropy_coefficient
        ),
        value_coefficient=float(ppo["value_coefficient"]),
        max_gradient_norm=float(ppo["max_gradient_norm"]),
        target_kl=float(ppo["target_kl"]),
    )


def plot_training_curves(
    training_records: list[dict[str, object]],
    validation_records: list[dict[str, object]],
    directory: Path,
) -> None:
    """生成回报、成功率、损失和 PPO 数值诊断曲线。"""

    configure_plot_style()
    directory.mkdir(parents=True, exist_ok=True)
    updates = [int(record["update"]) for record in training_records]
    validation_updates = [int(record["update"]) for record in validation_records]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    axes[0, 0].plot(
        updates,
        [record["rollout"]["mean_episode_reward"] for record in training_records],
        label="训练回报",
    )
    axes[0, 0].plot(
        validation_updates,
        [record["validation"]["mean_total_reward"] for record in validation_records],
        marker="o",
        label="确定性验证回报",
    )
    if all("stochastic_validation" in record for record in validation_records):
        axes[0, 0].plot(
            validation_updates,
            [
                record["stochastic_validation"]["mean_total_reward"]
                for record in validation_records
            ],
            marker="s",
            label="随机验证回报",
        )
    axes[0, 0].set(title="回报曲线", xlabel="更新轮次")
    axes[0, 0].legend()

    axes[0, 1].plot(
        updates,
        [record["rollout"]["success_rate"] for record in training_records],
        label="训练成功率",
    )
    axes[0, 1].plot(
        validation_updates,
        [record["validation"]["success_rate"] for record in validation_records],
        marker="o",
        label="确定性验证成功率",
    )
    if all("stochastic_validation" in record for record in validation_records):
        axes[0, 1].plot(
            validation_updates,
            [
                record["stochastic_validation"]["success_rate"]
                for record in validation_records
            ],
            marker="s",
            label="随机验证成功率",
        )
    axes[0, 1].set(title="成功率曲线", ylim=(0.0, 1.0), xlabel="更新轮次")
    axes[0, 1].legend()

    axes[1, 0].plot(updates, [record["update_metrics"]["actor_loss"] for record in training_records], label="Actor")
    axes[1, 0].plot(updates, [record["update_metrics"]["critic_loss"] for record in training_records], label="Critic")
    axes[1, 0].set(title="损失曲线", xlabel="更新轮次")
    axes[1, 0].legend()

    axes[1, 1].plot(updates, [record["update_metrics"]["entropy"] for record in training_records], label="熵")
    axes[1, 1].plot(updates, [record["update_metrics"]["approximate_kl"] for record in training_records], label="近似 KL")
    axes[1, 1].plot(updates, [record["update_metrics"]["clip_fraction"] for record in training_records], label="裁剪比例")
    axes[1, 1].set(title="PPO 数值诊断", xlabel="更新轮次")
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(directory / "training_curves.png", dpi=180)
    figure.savefig(directory / "training_curves.pdf")
    plt.close(figure)


def train_ppo(
    config: dict[str, Any],
    trainer: PPOTrainer,
    run_dir: Path,
    *,
    updates: int,
    rollout_steps: int,
    validation_interval: int,
    validation_cases: list[ValidationCase],
    final_cases: list[ValidationCase],
    gamma: float,
    gae_lambda: float,
    task_seed: int,
    type_seed: int,
    response_seed: int,
    update_seed: int,
    validation_action_seed: int | None = None,
) -> dict[str, object]:
    """执行普通 PPO 训练，按确定性验证回报选模并保留随机诊断。"""

    task_rng = np.random.default_rng(task_seed)
    type_rng = np.random.default_rng(type_seed)
    response_seed_rng = np.random.default_rng(response_seed)
    update_rng = np.random.default_rng(update_seed)
    training_records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []
    checkpoint = run_dir / "best_model.pt"

    initial_validation, _ = evaluate_trainer(
        trainer,
        config,
        validation_cases,
        deterministic=True,
    )
    initial_stochastic_validation, _ = evaluate_trainer(
        trainer,
        config,
        validation_cases,
        deterministic=False,
        action_seed=validation_action_seed,
    )
    initial_selection_score = validation_selection_score(initial_validation)
    validation_records.append(
        {
            "update": 0,
            "validation": initial_validation,
            "stochastic_validation": initial_stochastic_validation,
            "selection_score": initial_selection_score,
        }
    )
    append_jsonl(run_dir / "validation.jsonl", validation_records[-1])
    best_reward = initial_selection_score
    best_update = 0
    trainer.save_checkpoint(
        checkpoint,
        {
            "best_update": best_update,
            "selection_metric": "deterministic_mean_total_reward",
            "selection_score": initial_selection_score,
            "validation": initial_validation,
            "stochastic_validation": initial_stochastic_validation,
        },
    )

    for update in range(1, updates + 1):
        buffer, rollout_summary = collect_rollout(
            trainer,
            config,
            rollout_steps,
            task_rng=task_rng,
            type_rng=type_rng,
            response_seed_rng=response_seed_rng,
        )
        batch = buffer.to_batch(
            trainer.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            next_value=0.0,
        )
        update_metrics = trainer.update(batch, update_rng)
        record = {
            "update": update,
            "rollout": rollout_summary,
            "update_metrics": update_metrics.to_serializable(),
        }
        training_records.append(record)
        append_jsonl(run_dir / "training.jsonl", record)

        if update % validation_interval == 0 or update == updates:
            validation, _ = evaluate_trainer(
                trainer,
                config,
                validation_cases,
                deterministic=True,
            )
            stochastic_validation, _ = evaluate_trainer(
                trainer,
                config,
                validation_cases,
                deterministic=False,
                action_seed=validation_action_seed,
            )
            selection_score = validation_selection_score(validation)
            validation_record = {
                "update": update,
                "validation": validation,
                "stochastic_validation": stochastic_validation,
                "selection_score": selection_score,
            }
            validation_records.append(validation_record)
            append_jsonl(run_dir / "validation.jsonl", validation_record)
            if selection_score > best_reward:
                best_reward = selection_score
                best_update = update
                trainer.save_checkpoint(
                    checkpoint,
                    {
                        "best_update": best_update,
                        "selection_metric": "deterministic_mean_total_reward",
                        "selection_score": selection_score,
                        "validation": validation,
                        "stochastic_validation": stochastic_validation,
                    },
                )
        print(
            f"PPO update {update}/{updates}: "
            f"reward={rollout_summary['mean_episode_reward']:.4f}, "
            f"success={rollout_summary['success_rate']:.3f}, "
            f"entropy={update_metrics.entropy:.4f}, "
            f"kl={update_metrics.approximate_kl:.6f}",
            flush=True,
        )

    checkpoint_metadata = trainer.load_checkpoint(checkpoint)
    final_deterministic, deterministic_episodes = evaluate_trainer(
        trainer,
        config,
        final_cases,
        deterministic=True,
    )
    final_stochastic, stochastic_episodes = evaluate_trainer(
        trainer,
        config,
        final_cases,
        deterministic=False,
        action_seed=(
            None
            if validation_action_seed is None
            else int(validation_action_seed) + 1
        ),
    )
    plot_training_curves(training_records, validation_records, run_dir / "figures")
    write_json(training_records, run_dir / "training_records.json")
    write_json(validation_records, run_dir / "validation_records.json")
    write_json(
        [episode.to_serializable() for episode in deterministic_episodes],
        run_dir / "final_deterministic_episodes.json",
    )
    write_json(
        [episode.to_serializable() for episode in stochastic_episodes],
        run_dir / "final_stochastic_episodes.json",
    )
    return {
        "best_update": best_update,
        "best_validation_reward": best_reward,
        "selection_metric": "deterministic_mean_total_reward",
        "initial_validation": initial_validation,
        "initial_stochastic_validation": initial_stochastic_validation,
        "checkpoint_metadata": checkpoint_metadata,
        "total_transitions": int(
            sum(int(record["rollout"]["transitions"]) for record in training_records)
        ),
        "total_episodes": int(
            sum(int(record["rollout"]["episodes"]) for record in training_records)
        ),
        "final_deterministic": final_deterministic,
        "final_stochastic": final_stochastic,
        "training_records": training_records,
        "validation_records": validation_records,
    }
