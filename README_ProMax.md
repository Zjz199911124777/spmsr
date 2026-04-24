# SinSR ProMax：模型结构与训练说明

本文档面向需要**从输入到输出**完整理解当前工程的同学，说明 **SinSR ProMax**（`SinSRProMaxModel`）的模块组成、数据流、训练方式，以及与**原始 SinSR**（单主干 `UNetModelSwin`）的差异。

---

## 目录

1. [一句话概括](#1-一句话概括)
2. [整体系统：谁在训练、谁在冻结](#2-整体系统谁在训练谁在冻结)
3. [从像素到 latent：Autoencoder 与 Diffusion 的角色](#3-从像素到-latentautoencoder-与-diffusion-的角色)
4. [SinSRProMaxModel：从输入到输出的严格顺序](#4-sinsrpromaxmodel从输入到输出的严格顺序)
5. [各子模块详解](#5-各子模块详解)
6. [训练流程（代码级）](#6-训练流程代码级)
7. [损失项与配置中的权重](#7-损失项与配置中的权重)
8. [与原始 SinSR 的差异对照](#8-与原始-sinsr-的差异对照)
9. [相关文件与配置](#9-相关文件与配置)
10. [训练命令示例](#10-训练命令示例)

---

## 1. 一句话概括

- **原始 SinSR**：在 **VQ latent 空间**里，用 **Teacher（预训练 UNet-Swin）** 生成目标，训练 **Student（同样是 UNet-Swin）** 做一步式超分蒸馏。
- **SinSR ProMax**：在**同一套扩散蒸馏框架**上，把 Student 换成 **包装器 `SinSRProMaxModel`**：内部仍是 **UNet-Swin 主干**，但在主干前后增加了 **语义调制、空间语义图、细节分支、动态 LoRA 专家、不确定性引导** 等模块；训练侧增加了 **LPIPS、语义、语义分布、对抗、不确定性加权** 等可选损失与两阶段策略。

---

## 2. 整体系统：谁在训练、谁在冻结

| 组件 | 典型配置 | 是否训练 | 作用 |
|------|-----------|----------|------|
| **Teacher** | `UNetModelSwin`，`teacher_ckpt_path` 加载 | **冻结** | 通过 DDIM/DDPM 在 latent 上生成 `z_start_teacher`，作为蒸馏目标 |
| **Student** | `SinSRProMaxModel`（内含 backbone） | **训练** | 预测与 teacher / GT 对齐的输出 |
| **Autoencoder (VQ)** | `VQModelTorch` | **冻结** | 图像 ↔ latent 编解码 |
| **CLIP（若启用）** | `CLIPVisionModelWithProjection` | **冻结** | 仅从 LQ 提语义向量，不参与反传 |
| **LPIPS** | VGG 骨干 | **冻结** | 感知损失 |
| **判别器（Stage2）** | `PatchDiscriminator` | **训练** | 对抗损失 |

入口脚本：`main_distill.py` → `TrainerDistillDifIR`（`trainer.py`）。

---

## 3. 从像素到 latent：Autoencoder 与 Diffusion 的角色

### 3.1 训练时拿到的数据

- **`gt`**：高清图，数值范围按数据管线一般为 **[0, 1]**（以 `training_losses_distill` 注释为准）。
- **`lq`**：低清退化图，与 backbone 条件输入一致（常配合归一化到 **[-1, 1]** 送入网络，见模型内 `_clip_norm` 等）。

### 3.2 VQ Autoencoder

- **编码**：`encode_first_stage(...)` 把图像压到 **低分辨率 latent**（配置里 `z_channels: 3`，与超分因子配合，典型为原图 1/4 边长等）。
- **解码**：`decode_first_stage(...)` 把 latent 还原为 RGB，用于 **LPIPS / 语义 / 对抗** 等在**图像域**的损失。

### 3.3 Gaussian Diffusion（`gaussian_diffusion.py`）

- 维护 **时间步 schedule**（如 exponential `sqrt_etas`）、`q_sample`、`prior_sample`、teacher 采样循环等。
- **蒸馏训练**核心函数：`training_losses_distill(...)`：先算 teacher 的 `z_start_teacher`，再让学生网络在 `z_t` 上预测，与目标对齐。

---

## 4. SinSRProMaxModel：从输入到输出的严格顺序

实现文件：`models/sinsr_promax.py`，类名 **`SinSRProMaxModel`**。

Student 的 **Python 前向接口**为：

```text
forward(x, timesteps, lq=None, **kwargs)
```

其中：

- **`x`**：扩散侧送入的 **已缩放 latent**（即 `GaussianDiffusion._scale_input(z_t, t)` 之后的结果），形状与 latent 一致（如通道数常为三通道 latent，与 VQ 配置一致）。
- **`timesteps`**：时间步张量；在 `use_reflow=False` 时，训练里会被固定为 **最后一步**（见下文 `trainer.py`）。
- **`lq`**：低清图，用于 **条件 + 语义分支**；若 `cond_lq=True`，还会传入 **backbone**。

下面按 **`forward` 内代码执行顺序** 描述（这是「从输入到输出」的权威顺序）。

### 4.0 前置：从 kwargs 取出不确定性图

- **`uncertainty_map = kwargs.pop("uncertainty_map", None)`**  
  若训练器根据 teacher 多次前向估计了不确定性，会经 `student_extra_kwargs` 传入；用于后面在 **输入 latent `x`** 和 **主干输出 `out`** 上放大难区域响应。

---

### 4.1 若有 `lq`：全局语义向量 + 空间语义图（两条并行支路）

**条件：`lq is not None`**

1. **`semantic_embed = _semantic_embed(lq)`**  
   - 将 `lq` 双线性插值到 **224×224**。  
   - `_clip_norm`：把 **[-1,1]** 转到 **[0,1]** 再减均值除方差（CLIP 标准预处理）。  
   - **若有 CLIP**：`CLIPVisionModelWithProjection`，`no_grad`，得到 **`image_embeds`**，形状 **[B, 512]**。  
   - **若无 CLIP**：`fallback_semantic`（小 CNN）+ `fallback_proj` → 也是 **512 维** 语义向量。  
   - 用途：后面的 **FiLM 式调制**、**Detail 门控**、**动态 LoRA 路由**。

2. **`semantic_spatial = _semantic_spatial_map(lq, x)`**  
   - 输入：`lq * 0.5 + 0.5`（大致到 [0,1]）送入 **`semantic_spatial_encoder`**（3 → `semantic_prompt_channels` 的卷积塔，中间有 stride=2）。  
   - 若空间尺寸与 **`x`（当前 noisy latent）** 不一致，双线性插值到与 **`x` 相同 H×W**。  
   - 通道数 = **`semantic_prompt_channels`（默认 64）**。  
   - 用途：**加在输入 latent 上**（经 `semantic_spatial_scale`）；以及 **DetailBranch 的第四路拼接**。

---

### 4.2 不确定性引导（可选）：改 `x`，不改语义向量本身

**条件：`use_uncertainty_guidance and uncertainty_map is not None`**

1. 将 `uncertainty_map` 插值到与 **`x`** 同尺寸、同 device/dtype。  
2. **`x = x * (1 + uncertainty_attention_scale * uncertainty_map)`**  
   - 含义：在 teacher 认为「方差大、难」的区域，**放大输入特征幅度**，给 backbone 更多「容量」去处理。  
3. **若 `training` 且 `uncertainty_perturb_std > 0`**：  
   **`x = x + randn * uncertainty_perturb_std * uncertainty_map`**  
   - 含义：难区域加小噪声，**正则化**边缘/纹理恢复。

---

### 4.3 输入侧语义调制（FiLM）

**`x = _semantic_modulation(x, semantic_embed, semantic_input_proj, semantic_condition_scale)`**

- `semantic_input_proj`：Linear(512→512) → SiLU → Linear(512 → **`out_channels*2`**)。  
- 对 batch 维语义向量拆成 **scale / shift**，`tanh` 限幅，reshape 为 **[B, C, 1, 1]**。  
- 输出：  
  **`x = x * (1 + strength * scale) + strength * shift`**  
- **`strength`** = `semantic_condition_scale`（配置可调）。  
- **作用**：在 **进 backbone 之前**，按语义内容整体调节每个 latent 通道的尺度与偏置，使网络「先知道这张图大概是什么场景」。

---

### 4.4 空间语义图与 `x` 相加（带通道对齐）

**条件：`semantic_spatial is not None and semantic_spatial_scale > 0`**

- 若 **`semantic_spatial` 通道数 ≠ `x` 通道数**（例如 64 vs 3）：对空间图在通道维做 **mean 再 expand 到 `x` 的通道数**，保证可广播相加。  
- **`x = x + semantic_spatial_scale * semantic_spatial_input`**  
- **作用**：在 **空间上** 非均匀地注入与 LQ 相关的语义/纹理提示（与 4.3 的全局向量互补）。

---

### 4.5 主干网络（原始 SinSR 的「芯」）

**`out = self.backbone(x, timesteps, lq=lq, **kwargs)`**

- **`base_target`** 在配置里为 **`models.unet.UNetModelSwin`**。  
- **`base_params`**：`in_channels: 6`、`out_channels: 3`、`cond_lq: True` 等——即 **Swin-UNet 扩散超分主干**，在 latent 上预测（具体是 x0 还是 epsilon 由 diffusion 的 `model_mean_type` 决定，配置里多为 **predict xstart**）。  
- **这一层之前的一切（不确定性、语义调制、空间图）都是对 `x` 的预处理；这一层是整个 ProMax 的算力主体。**

---

### 4.6 不确定性引导（输出侧）

**条件：`use_uncertainty_guidance and uncertainty_map is not None`**

- 将 `uncertainty_map` 对齐到 **`out`** 的尺寸。  
- **`out = out * (1 + 0.5 * uncertainty_attention_scale * uncertainty_map)`**  
- **作用**：在输出 latent 上再次强调难区域（系数 0.5 弱于输入侧，避免过强）。

---

### 4.7 输出侧语义调制（FiLM，强度减半）

**`out = _semantic_modulation(out, semantic_embed, semantic_output_proj, 0.5 * semantic_condition_scale)`**

- 与 4.3 相同结构，但 **`strength` 为 `0.5 * semantic_condition_scale`**。  
- **作用**：主干算完后，再用同一语义向量做一次 **较温和** 的通道调制，对齐「内容语义」与「重建 latent」。

---

### 4.8 细节分支 DetailBranch2d（在图像域对齐的 LQ 信息）

**条件：`enable_detail_branch and lq is not None and detail_branch_strength > 0`**

1. **`detail_residual = self.detail_branch(out, lq, semantic_spatial)`**  
   - 内部：  
     - 将 **`lq` 插值到与 `out` 相同 H×W**。  
     - **`lq_highpass = lq - avg_pool_smooth(lq)`**，突出边缘/纹理。  
     - 在通道维 **concat**：**`[out, lq, lq_highpass, semantic_spatial]`**（通道数 = `out_ch + 3 + 3 + semantic_prompt_channels`）。  
     - 小 CNN：`in_proj` → depthwise+pointwise 残差 → `out_proj`（**`out_proj` 权重与 bias 初始化为 0**，训练初期近似恒等，稳定）。  
2. **可选：`semantic_detail_gate`**  
   - `detail_gate = sigmoid(semantic_detail_proj(semantic_embed))`，reshape **[B, C, 1, 1]**。  
   - **`detail_residual *= (1 + semantic_detail_gate * detail_gate)`**  
   - **作用**：按语义决定「细节支路」在哪些通道更强。  
3. **`out = out + detail_branch_scale * detail_branch_strength * detail_residual`**  
   - **`detail_branch_strength`** 可由训练器 **warmup**（`detail_branch_warmup_iters`）从 0 线性拉到 1。

**直观理解**：主干负责整体结构与 latent 语义一致；Detail 分支专门吃 **LQ 高频 + 空间语义图**，补 **纹理与边缘**。

---

### 4.9 动态 LoRA 专家（可选，且依赖 `lq`）

**条件：`enable_dynamic_lora and lq is not None`**

- 再次 **`sem = _semantic_embed(lq)`**（与前面同路径）。  
- **`gates = softmax(router(sem), dim=1)`**，**`router`: Linear(512 → num_experts)**。  
- 每个 **`LoRAExpert2d`**：`1×1 conv down (C→r)` → `1×1 conv up (r→C)`，`up` 初始为 0，初期近似不加残差。  
- **`residual = Σ_e gates_e * expert_e(out)`**  
- **返回：`out + lora_scale * residual`**  

**作用**：不同样本根据语义 **软路由** 到不同低秩专家，提升 **跨场景泛化**，参数相对全宽卷积更省。

**若 `enable_dynamic_lora=False` 或 `lq=None`**：直接 **`return out`**（无 LoRA 残差）。

---

### 4.10 小结：张量级别的「插入点」一览

| 顺序 | 位置 | 模块 | 对张量的操作 |
|:----:|------|------|----------------|
| 0 | kwargs | `uncertainty_map` | 取出，供后续使用 |
| 1 | 有 `lq` | `_semantic_embed` / `_semantic_spatial_map` | 得到 512 维向量 + H×W 空间图 |
| 2 | 输入 `x` 上 | 不确定性 | 乘性放大 + 可选加性噪声 |
| 3 | 输入 `x` 上 | `semantic_input_proj` | FiLM 调制 |
| 4 | 输入 `x` 上 | `semantic_spatial` | 加性偏置（通道对齐后） |
| 5 | 核心 | `backbone` (UNet-Swin) | latent → latent |
| 6 | 输出 `out` 上 | 不确定性 | 乘性放大 |
| 7 | 输出 `out` 上 | `semantic_output_proj` | FiLM 调制（弱） |
| 8 | 输出 `out` 上 | `DetailBranch2d` | 加性细节残差 |
| 9 | 输出 `out` 上 | Dynamic LoRA | 加性专家混合残差 |

---

## 5. 各子模块详解

### 5.1 LoRAExpert2d

- **结构**：两个 1×1 卷积，低秩 `rank`。  
- **初始化**：down 用 kaiming；**up 置 0** → 训练初期 **不改变 `out`**，优化稳定。  
- **作用**：在固定通道宽 `C` 上增加 **低成本可学习残差**。

### 5.2 DetailBranch2d

- **输入通道设计**：`out_channels + 6 + semantic_channels`  
  - `6` = LQ 的 RGB(3) + 高频 RGB(3)。  
  - `semantic_channels` = 空间语义图通道数（与 `semantic_prompt_channels` 一致）。  
- **作用**：显式利用 **LQ 高频** 与 **语义空间图**，修补主干在纹理上的不足。

### 5.3 语义：CLIP vs Fallback

- **CLIP**：更强语义对齐，权重冻结，仅前向。  
- **Fallback**：轻量 CNN + Linear → 512 维，保证无 transformers 时仍可训。

### 5.4 `extract_semantic_features`（训练器里用于语义损失）

- 对给定 **RGB 图像**（如解码后的 SR 或 GT）：  
  - 全局：`normalize(_semantic_embed(image), dim=1)`  
  - 空间：`normalize(_semantic_spatial_map(image, image), dim=1)`  
- 用于 **学生 vs 目标（teacher 解码 / GT 混合）** 的 cosine / L1 / 分布损失。

---

## 6. 训练流程（代码级）

### 6.1 主循环

`TrainerBase.train()`（`trainer.py`）：

1. `init_logger` → `build_model` → `setup_optimizaton` → `resume_from_ckpt`（可选）  
2. `build_dataloader`、`build_iqa`  
3. `for ii in range(iters_start, iterations)`：  
   - `data = prepare_data(next(train_loader))`  
   - `training_step(data)`  
   - 每 **`val_freq`** 步：`validation()`  
   - `adjust_lr()`  
   - 每 **`save_freq`** 步：`save_ckpt()`  
   - 分布式时 `sampler.set_epoch`

### 6.2 `TrainerDistillDifIR.build_model` 要点

- 构建 **teacher** 并 `load_state_dict`，**eval + requires_grad(False)**。  
- 构建 **student**：若 teacher/student 同构，可从 teacher **deepcopy** 初始化；异构则用配置 `target` 新建。  
- **DDP**：`find_unused_parameters` 可配置为 True，避免动态分支部分参数未参与 loss 时 DDP 报错。  
- 加载 **VQ autoencoder**、**LPIPS**、**GaussianDiffusion** 实例 `base_diffusion`。  
- **EMA**：rank0 维护 `ema_state`。

### 6.3 `training_step` 单次迭代（micro-batch）

1. **`microbatch`**：把大 batch 切成多段，梯度累积 `num_grad_accumulate`。  
2. **`_set_dynamic_lora_state()`**：根据配置开关 `set_dynamic_lora`、以及 **detail_branch_strength 的 warmup**。  
3. **时间步 `tt`**：  
   - 若 **`use_reflow=False`**（默认）：**`tt` 全部固定为 `num_timesteps - 1`** → 学生始终在 **最大噪声步** 上学习，对应 **一步式 / 固定步蒸馏** 设定。  
4. **`noise`**：与 latent 空间分辨率一致的高斯噪声。  
5. **`model_kwargs`**：`{'lq': micro_data['lq']}`（当 `cond_lq=True`）。  
6. **不确定性**（若 `use_uncertainty_guidance` 或 `lambda_uncert>0`）：  
   - 用 teacher 在 **加噪的 `z_t` 上多次前向**，算方差图并归一化 → **`uncertainty_map`**，可选传入 **`student_extra_kwargs`**。  
7. 调用 **`base_diffusion.training_losses_distill(...)`** 得到：  
   - `losses` 字典（含 `mse_distill`、`mse_xT`、`mse_gt` 等）  
   - `z_t, z0_pred, teacher_zstart`  
8. **`_aggregate_stage_loss`**：把蒸馏项与 LPIPS、语义、对抗、uncert 等加权合成 **`losses["loss"]`**。  
9. **反向**：`loss.mean() / num_grad_accumulate` → `backward`；micro 循环结束后 **`optimizer.step()`**。  
10. **对抗**（若开启）：另用 **全 batch** 算 fake、更新判别器（见 `trainer.py` 后续段落）。

### 6.4 混合精度与分布式

- **`use_fp16`**：`autocast` + `GradScaler`。  
- **多卡**：非最后一个 micro-batch 时用 **`model.no_sync()`** 减少通信。

---

## 7. 损失项与配置中的权重

主蒸馏在 **`training_losses_distill`** 中完成；**总损失**在 **`_aggregate_stage_loss`** 中汇总。

常见项（是否启用取决于 yaml）：

| 键名 | 含义 | 典型权重配置键 |
|------|------|----------------|
| `mse_distill` | 学生预测与 teacher 目标在 latent（或图像，若 `loss_in_image_space`）上的对齐 | `lambda_distill` |
| `mse_xT` | 学习噪声端 / 轨迹相关（`learn_xT`） | `lambda_xT` + `xT_warmup_iters` |
| `mse_gt` | 与真值 latent 对齐（`finetune_use_gt`） | `gt_warmup_iters` 等 |
| `mse_gt_image` / `loss_gt_image` | 解码后与 GT 图像 L2 | `lambda_gt_image` |
| `lpips` | 感知损失 | `lambda_lpips` + `lpips_warmup_iters` |
| `loss_semantic` | 全局 cosine + 空间 L1（学生 SR vs 混合目标） | `lambda_semantic` + `semantic_warmup_iters` |
| `loss_semantic_distribution` | 空间特征均值/协方差对齐 | `lambda_semantic_distribution` |
| `loss_uncert` | 不确定区域加权 MSE | `lambda_uncert_weight_mse` |
| `loss_adv` | 生成器对抗项 | `lambda_adv` |

**Warmup**：多项损失通过 `_get_linear_ramp_weight` 在指定 iteration 内从 0 线性升到配置值，避免训练初期各目标冲突导致发散。

**说明**：若 teacher 解码用于语义目标时 **显存不足（OOM）**，实现中可对 teacher 解码做跳过/降级，仅用 GT 语义目标继续训练（见 `trainer.py` 中语义分支的 try/except 逻辑）。

---

## 8. 与原始 SinSR 的差异对照

| 维度 | 原始 SinSR（`configs/SinSR.yaml`） | SinSR ProMax（`configs/SinSR_ProMax_stage*.yaml`） |
|------|-----------------------------------|-----------------------------------------------------|
| **Student 网络** | 单个 **`UNetModelSwin`** | **`SinSRProMaxModel`**：`UNetModelSwin` + 外挂模块 |
| **语义** | 无 CLIP 全局/空间显式分支 | **CLIP（或 fallback）** + **输入/输出 FiLM** + **空间语义图** |
| **细节** | 无独立 Detail 分支 | **DetailBranch2d**（LQ + 高频 + 语义图） |
| **动态容量** | 无 | **Router + 多 LoRAExpert2d**（可关） |
| **不确定性** | 无显式 uncertainty map 注入前向 | 可选 **teacher 方差图** 调制输入/输出 + **uncert 加权 loss** |
| **损失** | 以蒸馏 + xT/GT 等为主 | 额外 **LPIPS、语义、语义分布、对抗、uncert** 等 |
| **训练策略** | 单配置长训为主 | **Stage1 / Stage2**：先稳后强（对抗、LoRA、uncertainty 等分阶段） |
| **Teacher** | 同结构 Swin-UNet 蒸馏 | 可为 **异构**（`teacher_target` / `params_teacher` 与 student 不同宽） |

**不变的核心**：仍是 **VQ latent + Gaussian Diffusion 公式 + Teacher 生成蒸馏目标**；ProMax 是在 **同一套公式** 上增强 **Student 的表达与训练目标**。

---

## 9. 相关文件与配置

| 路径 | 说明 |
|------|------|
| `main_distill.py` | 入口：读 yaml、`TrainerDistillDifIR.train()` |
| `trainer.py` | `TrainerDistillDifIR`：蒸馏、多损失、DDP、AMP、验证、保存 |
| `models/sinsr_promax.py` | **SinSRProMaxModel** 完整前向与所有外挂子模块 |
| `models/gaussian_diffusion.py` | 扩散与 **`training_losses_distill`** |
| `models/unet/...` | **UNetModelSwin** 主干 |
| `configs/SinSR.yaml` | **原始 SinSR** 参考配置 |
| `configs/SinSR_ProMax_stage1.yaml` | ProMax 第一阶段（偏稳） |
| `configs/SinSR_ProMax_stage2.yaml` | ProMax 第二阶段（对抗 + 动态 LoRA + 不确定性等） |

当前常用训练频率（以 yaml 为准，可自行再改）：

- **`val_freq`**：验证间隔（步数）  
- **`save_freq`**：checkpoint 保存间隔（步数）

---

## 10. 训练命令示例

在 `SinSR-main1` 目录下，三卡示例：

```bash
cd /home/zhangjunzheng/SinSR-main/SinSR-main1
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_distill.py \
  --cfg_path ./configs/SinSR_ProMax_stage1.yaml \
  --save_dir ./saved_logs/promax_stage1
```

第二阶段将 `cfg_path` 换为 `SinSR_ProMax_stage2.yaml` 即可。

断点续训：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_distill.py \
  --cfg_path ./configs/SinSR_ProMax_stage1.yaml \
  --save_dir ./saved_logs/promax_stage1 \
  --resume /path/to/ckpts/model_1000.pth
```

---

## 附录：数据流简图（ASCII）

```text
  [GT, LQ] (像素)
       |
       v
  VQ Encoder ----> z_gt, z_lq (latent)
       |
       v
  Diffusion: prior_sample / teacher DDIM --> z_t, z_start_teacher
       |
       v
  Student: SinSRProMaxModel(x=scaled(z_t), t, lq)
       |     [语义 FiLM / 空间图 / 不确定 / backbone / 细节 / LoRA]
       v
  model_output (latent 域预测)
       |
       +--> Loss: distill (+ xT + gt + ... 在 trainer 聚合)
       |
       v
  VQ Decoder (可选) --> RGB --> LPIPS / 语义 / 对抗 ...
```

---

*文档版本：与仓库内 `SinSRProMaxModel` 及 `TrainerDistillDifIR` 逻辑一致；若你后续改动了 `forward` 或损失聚合，请同步更新本节。*
