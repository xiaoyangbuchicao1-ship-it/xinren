# Human-AI Consensus with Dual-Loop Meta-Learning and Trust Regulation

## 中文说明

本代码包包含论文模型的必要源代码，可用于重新训练基于信任调节和一阶双循环元强化学习的人机群体共识模型。代码包不包含预训练权重、训练输出、论文分析脚本、单元测试、缓存文件或临时文件。

模型使用信任决策空间表示人类对AI的主观信任，使用人机意见之间的Pearson相关系数表示AI对人类意见的信息信任。信任调节后的综合意见进入多轮共识过程，FOMAML-PPO策略根据不同群体的响应弹性生成意见调整建议。

### 目录结构

```text
configs/frozen_v1.yaml                 冻结的环境和基础参数
scripts/run_group_receptiveness_maml.py  FOMAML-PPO训练入口
src/agents/                            PPO网络、更新与元梯度
src/common/                            配置、日志与随机种子
src/data/                              决策任务生成
src/env/                               共识环境与异质响应模型
src/experiments/                       元训练、内循环适应与训练期验证
src/model/                             信任、意见融合、共识度与和谐度模型
requirements.txt                       Python依赖
```

`src/experiments/evaluate_policies.py`提供训练流程调用的回合统计函数，并不是独立测试套件。项目原有的`tests/`目录和论文专用测试、对比及敏感性分析脚本均未包含在本代码包中。

### 环境安装

建议使用Python 3.10或更高版本。PyTorch应根据本机CUDA版本安装。

```powershell
conda activate pytorch_gpu
python -m pip install -r requirements.txt
```

### 正式训练命令

直接建议模式不会读取位置参数中的模型文件，命令中的`unused.pt`只是为兼容训练入口的统一参数格式，不需要创建该文件。

```powershell
python scripts\run_group_receptiveness_maml.py unused.pt `
  --task-mode elasticity `
  --guidance-mode direct `
  --fresh-meta-actor `
  --direct-initial-recommendation 0.5 `
  --task-split-mode range_ood `
  --elasticity-range-profile wide `
  --balanced-elasticity-batches `
  --meta-iterations 200 `
  --meta-batch-size 4 `
  --support-episodes 5 `
  --query-episodes 10 `
  --validation-query-episodes 48 `
  --test-query-episodes 64 `
  --validation-interval 5 `
  --inner-learning-rate 0.8 `
  --meta-learning-rate 2e-5 `
  --outer-update-epochs 3 `
  --calibration-coefficient 1.0 `
  --policy-gradient-coefficient 0.05 `
  --reward-mode deficit `
  --deficit-progress-weight 1.0 `
  --deficit-modification-cost 1.5 `
  --deficit-round-cost 0.01 `
  --deficit-success-bonus 0.25 `
  --deficit-timeout-penalty 0.25 `
  --recommendation-cost-weight 0.01 `
  --remaining-deficit-cost-weight 0.05 `
  --unexecuted-recommendation-cost-weight 0.15 `
  --seed 8203 `
  --task-split-seed 2026 `
  --validation-case-seed 41001 `
  --test-case-seed 51001
```

程序会在新建的`outputs/`目录中保存训练日志和模型权重。该目录已加入`.gitignore`，不会被默认提交。

## English Description

This package contains the source code required to retrain the human-AI group consensus model based on trust regulation and first-order dual-loop meta-reinforcement learning. Pretrained checkpoints, generated outputs, paper-specific analysis scripts, automated tests, caches, and temporary files are excluded.

The Trust Decision Making Space represents humans' subjective trust in AI, and the Pearson correlation between human and AI opinions represents AI's information trust in human opinions. The trust-regulated integrated opinions enter a multi-round consensus process, where a FOMAML-PPO policy generates feedback recommendations under heterogeneous response-elasticity tasks.

The formal command above reproduces the training configuration associated with seed 8203. Generated checkpoints and logs are written to `outputs/`, which is ignored by Git.
