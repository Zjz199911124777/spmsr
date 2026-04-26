# SinSR Stage1 Effective 说明文档

本文档用于说明你当前新增的 `stage1` 方案（`SinSR_ProMax_stage1_effective.yaml`）与：

1. 之前的 `main1` 第一阶段（`SinSR_ProMax_stage1.yaml`）  
2. 官方原始 SinSR（以 `UNetModelSwin` + 官方训练范式为代表）

之间的核心区别、设计动机和使用方式。

---

## 1. 背景与目标

你当前观察到的现象是：

- 旧版 `main1 stage1` 在同类测试口径下，指标（例如 MUSIQ）平台期偏低；
- `stage2` 能显著抬升结果，说明“阶段二有效”，但“阶段一底座质量不足”；
- 因此本次改动目标是：**先把 stage1 训练稳定性和上限拉起来**，再进入 stage2 微调。

本次新增配置文件：

- `configs/SinSR_ProMax_stage1_effective.yaml`

本次代码改动文件：

- `trainer.py`

---

## 2. 新 Stage1 的模型结构（与旧 Stage1 的区别）

### 2.1 学生主干规模

- **旧 stage1**：`model.params.base_params.model_channels = 192`
- **新 stage1 effective**：`model_channels = 160`

设计意图：先回归到更接近 teacher/官方主干的规模，降低异构蒸馏难度和优化不稳定性。

### 2.2 ProMax 插件分支策略

新 stage1 里，先关闭复杂外挂模块，聚焦主干蒸馏：

- `enable_dynamic_lora: False`
- `enable_detail_branch: False`
- `semantic_condition_scale: 0.0`
- `semantic_spatial_scale: 0.0`
- `semantic_detail_gate: 0.0`

对比旧 stage1：旧配置会保留 detail/semantic 等轻量增强，这些模块在 stage1 早期会引入额外优化目标，可能稀释主蒸馏收敛。

---

## 3. 训练目标与优化策略差异（新 vs 旧 stage1）

### 3.1 Loss 组合

新 stage1 采用“强主目标、弱辅目标”：

- 保留主蒸馏：`lambda_distill = 1.0`
- 辅助项降权：
  - `lambda_xT = 0.08`
  - `finetune_use_gt = 0.12`
- 关闭高干扰项：
  - `lambda_gt_image = 0.0`
  - `lambda_lpips = 0.0`
  - `lambda_semantic = 0.0`
  - `lambda_semantic_distribution = 0.0`
  - `use_adv = False`

旧 stage1 相比之下属于多目标同时优化（GT/LPIPS/semantic 等都开），更容易在前期把优化方向拉散。

### 3.2 时间步训练策略（本次代码新增）

在 `trainer.py` 中新增了两个配置项：

- `train.fixed_student_timestep`（默认 `True`，保持旧行为）
- `train.student_timestep_ratio`（默认 `1.0`）

新 stage1 effective 设置：

- `fixed_student_timestep: False`

含义：不再强制固定 `t = T-1`，而是允许随机时间步训练，有助于 stage1 打底阶段的稳定收敛和泛化。

### 3.3 训练节奏

新 stage1 effective：

- `iterations: 300000`（旧 stage1 为 120000）
- `lr: 5e-5`（旧 stage1 为 3e-5）
- `use_fp16: False`、`microbatch: 64`（优先稳定性）

---

## 4. 与官方原始 SinSR 的主要差异

即使采用了 `stage1 effective`，它仍然不是“官方原始 SinSR 训练”。

### 4.1 模型层面

- 官方：单 `UNetModelSwin` 主体（无 ProMax 包装模块）
- 当前：仍使用 `SinSRProMaxModel` 包装器（只是 stage1 暂时关闭多数外挂分支）

### 4.2 训练范式

- 官方更接近原始 SinSR 的蒸馏/采样与训练流程
- 当前 `main1` 依然在 `TrainerDistillDifIR` 框架下训练，具备扩展 loss 与后续 stage2 的统一接口

### 4.3 数据与验证口径

- 当前配置使用：
  - train: `LSDIR/train`
  - val: `DIV2K_V2_val`
- 与你历史“官方跑法”若数据/验证集不同，指标不能直接一一等价比较

---

## 5. 为什么这样改（核心思路）

一句话：**先把 stage1 从“复杂联合优化”改成“高确定性蒸馏打底”**。

先保证：

1. 主干可稳定逼近 teacher；
2. 指标曲线在中前期持续上升而不是早平台；
3. 再把复杂模块放到 stage2 发力。

---

## 6. 推荐训练命令（多卡，自动使用全部 GPU）

```bash
cd /home/zhangjunzheng/SinSR-main/SinSR-main1 && \
GPU_NUM=$(nvidia-smi -L | wc -l) && \
torchrun --nproc_per_node=${GPU_NUM} main_distill.py \
  --save_dir /home/zhangjunzheng/SinSR-main/saved_logs/promax_stage1_effective \
  --cfg_path ./configs/SinSR_ProMax_stage1_effective.yaml
```

如显存不足，可临时覆盖：

```bash
--override train.batch=[32,8] train.microbatch=8
```

---

## 7. 与后续 Stage2 的衔接建议

当 `stage1 effective` 的验证曲线（尤其 MUSIQ/clipiqa）明显优于旧 stage1 后，再进行 stage2 微调。  
建议把 `stage1` 最优 checkpoint 作为 `stage2` 的 `model.ckpt_path`，再开启对抗、语义和细节增强分支。

