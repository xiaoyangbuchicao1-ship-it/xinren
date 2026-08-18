# 校准式 FOMAML-PPO 人机共识实验

该目录是从原实验工程中抽出的干净版本，只保留当前论文方案的训练、绘图、环境源码和直接相关测试，不包含历史输出、SAC、GRU及失败尝试。

## 1. 环境与测试

```powershell
conda activate pytorch_gpu
cd D:\projects\codex\钱\human_ai_consensus\new
python -m pip install -r requirements.txt
python -m pytest -q
```

## 2. 最小训练检查

直接建议模式不会读取位置参数中的检查点，`unused.pt`只是为了兼容统一命令格式。

```powershell
python scripts\run_group_receptiveness_maml.py unused.pt `
  --task-mode elasticity `
  --guidance-mode direct `
  --fresh-meta-actor `
  --direct-initial-recommendation 0.5 `
  --meta-iterations 2 `
  --meta-batch-size 2 `
  --support-episodes 2 `
  --query-episodes 2 `
  --validation-query-episodes 4 `
  --test-query-episodes 4 `
  --validation-interval 1 `
  --outer-update-epochs 1 `
  --reward-mode deficit `
  --seed 8201
```

## 3. 正式训练

```powershell
python scripts\run_group_receptiveness_maml.py unused.pt `
  --task-mode elasticity `
  --guidance-mode direct `
  --fresh-meta-actor `
  --direct-initial-recommendation 0.5 `
  --meta-iterations 200 `
  --meta-batch-size 4 `
  --support-episodes 5 `
  --query-episodes 5 `
  --validation-query-episodes 24 `
  --test-query-episodes 64 `
  --validation-interval 4 `
  --outer-update-epochs 1 `
  --meta-learning-rate 5e-5 `
  --reward-mode deficit `
  --seed 8201
```

建议分别运行种子 `8201`、`8202`、`8203`。程序默认使用：

- 一阶 FOMAML 外更新；
- 宽范围响应弹性训练及双侧 OOD 测试；
- 内学习率 `0.8`；
- 响应证据系数 `1.0`；
- 支持集 PPO 梯度系数 `0.05`；
- 初始直接建议量 `0.5`。

每次训练会在 `outputs` 下建立独立目录，并保存原始 JSON、最佳/最终权重、留出测试结果和基础图表。正式使用时应选择 `best_elasticity_maml.pt`，而不是默认使用最终权重。

## 4. 生成中英文论文图

将下面的 `<run_dir>` 替换成某次训练生成的目录：

```powershell
python scripts\render_calibrated_fomaml_training.py <run_dir>
```

输出位置：

```text
<run_dir>/paper_figures/zh
<run_dir>/paper_figures/en
```

每幅图同时保存为独立的 PNG 和 PDF；浅色线为原始数据，深色线为明确标注的滑动平均，不修改原始实验记录。

## 5. 核心文件

- `configs/frozen_v1.yaml`：冻结环境与基础参数；
- `scripts/run_group_receptiveness_maml.py`：训练、验证、OOD测试与权重保存；
- `scripts/render_calibrated_fomaml_training.py`：中英文论文图；
- `src/experiments/response_elasticity_task.py`：响应弹性任务与支持集证据估计；
- `src/experiments/group_receptiveness_maml.py`：共享快参数适应和一阶元更新；
- `src/agents/maml_ppo.py`：PPO损失、快适应和元梯度聚合；
- `tests/`：与当前方案直接对应的自动化测试。
