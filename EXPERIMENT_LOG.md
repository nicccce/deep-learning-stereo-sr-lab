# 实验过程记录

> 用于实时记录实验命令、配置变化、结果和观察。每次训练或评估后追加一条，后续写实验报告时可以直接整理这里的内容。

## 2026-05-28 框架搭建

- 任务：双目图像超分辨率重建。
- 数据：Flickr1024 作为默认训练/验证集；Middlebury 作为可选补充；KITTI depth selection 用作推理、计时或合成 LR 测试，不作为严格双目 SR 标准测试标注。
- 基线：`StereoSRNet = 共享残差特征提取 + 双向视差注意力 + 融合重建 + PixelShuffle 上采样 + Bicubic 残差连接`。
- 消融：`MonoSRNet` 用相同训练/评估脚本验证双目融合是否有效。
- 损失：L1 重建损失 + 可选 Focal Frequency Loss + 可选注意力平滑正则。
- 指标：PSNR、SSIM、每对图像推理时间、模型参数量。
- 运行状态：`overfit` 固定单 batch 过拟合验证，`train` 完整数据训练。

## 2026-05-28 自检结果

- `conda run -n dl-lab python scripts/smoke_test.py`：通过，模型前向、FFL、注意力正则、PSNR/SSIM 计算可运行。
- `/tmp` 迷你 Flickr1024 格式数据集 overfit 入口：通过，CPU 跑 2 epoch，L1 从 0.4379 降到 0.4357，验证数据扫描、Dataset、训练循环和 checkpoint 路径可执行。

## 2026-05-29 CUDA overfit 与正式阶段测试

### 环境确认

- 运行环境：`conda run -n dl-lab python`。
- PyTorch/CUDA：`torch 1.11.0+cu113`，`torch.cuda.is_available() == True`。
- GPU：4 张 `Tesla K80`，本次训练显式使用 `cuda:0`；测试开始前显存空闲，正式短测时 GPU0 约 913 MiB 显存占用、GPU 利用率约 92%。
- 自检：`conda run -n dl-lab python scripts/smoke_test.py` 通过，前向、损失、PSNR/SSIM 计算正常。

### E1: overfit x2 单 batch 测试

- 命令：`conda run -n dl-lab python scripts/train.py --config configs/overfit_x2.json --output-dir runs/overfit_x2_cuda_30ep_20260529 --epochs 30 --device cuda:0`
- 配置摘要：Flickr1024 Train，`scale=2`，固定 1 个训练 batch，`hr_patch_size=96`，`batch_size=1`，轻量 `StereoSRNet`，约 0.175M 参数。
- 结果：通过。训练 L1 从 `0.096590` 降到 `0.051922`，说明模型、数据读取、损失与反传链路可以在固定 batch 上学习。
- 固定 batch 评估：第 10 epoch `PSNR=27.391 / SSIM=0.8508`，第 30 epoch `PSNR=27.523 / SSIM=0.8605`。
- 产物：`runs/overfit_x2_cuda_30ep_20260529/history.json`、`best.pt`、`latest.pt`、`config.json`。

### E3: stereo x2 正式阶段短测

- 命令：`conda run -n dl-lab python scripts/train.py --config configs/stereo_sr_x2.json --output-dir runs/formal_x2_cuda_stage_20260529 --epochs 5 --batch-size 4 --limit-train 64 --limit-val 8 --eval-train-every 1 --eval-train-limit 8 --device cuda:0`
- 数据：Flickr1024 `Train` / `Validation`，`scale=2`；训练阶段限制前 64 对，验证限制前 8 对；每个 epoch 额外在训练 split 前 8 对上做 deterministic eval，用于画训练集/测试集曲线。
- 模型：`StereoSRNet`，`channels=48`，`num_feature_blocks=6`，`num_reconstruct_blocks=4`，约 `515,524` 参数。
- 训练设置：`epochs=5`，`batch_size=4`，AdamW，`lr=2e-4`，CosineAnnealingLR，`amp=true`，`clip_grad_norm=1.0`，训练 patch `128`，评估中心裁剪 `512`。
- 通过判断：运行无报错；train/validation PSNR 与 SSIM 在 5 个 epoch 内整体上升，验证集最佳出现在 epoch 5。

| Epoch | Train L1 | Train PSNR | Train SSIM | Validation PSNR | Validation SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.074100 | 23.7175 | 0.7691 | 23.7019 | 0.7697 |
| 2 | 0.066499 | 23.7360 | 0.7708 | 23.7193 | 0.7725 |
| 3 | 0.061529 | 23.7450 | 0.7720 | 23.7286 | 0.7741 |
| 4 | 0.072433 | 23.7498 | 0.7725 | 23.7339 | 0.7744 |
| 5 | 0.067391 | 23.7508 | 0.7726 | 23.7349 | 0.7747 |

- 独立评估命令：`conda run -n dl-lab python scripts/evaluate.py --config runs/formal_x2_cuda_stage_20260529/config.json --checkpoint runs/formal_x2_cuda_stage_20260529/best.pt --limit 8 --output runs/formal_x2_cuda_stage_20260529/eval_results.json --device cuda:0`
- 独立评估结果：`PSNR=23.7349`，`SSIM=0.7747`，`seconds_per_pair=0.2097`，`num_pairs=8`，`parameters=515,524`。
- 曲线图：`runs/formal_x2_cuda_stage_20260529/history_curve.png`。

![Epoch 对训练集和验证集 PSNR/SSIM 跑分曲线](runs/formal_x2_cuda_stage_20260529/history_curve.png)

### 观察与下一步

- overfit 通过，说明单 batch 学习链路成立；正式短测通过，说明完整 train/validation split、checkpoint、history、evaluate 和绘图链路可用。
- 本次正式测试是阶段性短测，不是最终 50 epoch 全量结果；表中数值适合写入阶段 LOG，不应作为最终论文指标。
- 后续正式结果建议在相同记录格式下扩大到完整 `configs/stereo_sr_x2.json`，再补 `mono_sr_x2_ablation` 和 `stereo_sr_x4` 对照。

## 2026-05-29 论文式 x2 全量训练

### 配置调整

- 按参考代码的思路改为“小训练 patch + 大 batch + 验证/测试较大裁剪”。
- 已直接修改 `configs/stereo_sr_x2.json`：`hr_patch_size=96`，`eval_crop_size=256`，`batch_size=16`，`num_workers=8`，`limit_train=0`，`limit_val=0`，`output_dir=runs/stereo_sr_x2_paperlike`。
- 当前 x2 尺寸关系：训练 LR `48x48` -> SR/HR `96x96`；评估 LR `128x128` -> SR/HR `256x256`。
- 对照参考：SwiniPASSR/NTIRE22 x4 配置使用 `H_size=96`、LR `24x24`、stereo batch `16`；iPASSR x2 使用 LR `30x90` / HR `60x180` patch、batch `36`。因此本配置更接近论文代码的训练形态。

### 训练命令

```bash
conda run -n dl-lab python scripts/train.py   --config configs/stereo_sr_x2.json   --device cuda:0   --eval-train-every 1   --eval-train-limit 0
```

- 数据：Flickr1024 Train 全 800 对，Validation 全 112 对；另用 Test 全 112 对做独立评估。
- 训练：50 epoch，`StereoSRNet` 约 515,524 参数，AdamW，`lr=2e-4`，CosineAnnealingLR，AMP 开启。
- 速度：评估裁剪从 `512` 降到 `256` 后，每对评估约 `0.050~0.074s`；训练期间记录的 `seconds_per_pair` 约 `0.074s`。

### 训练曲线摘要

| Epoch | Train L1 | Train PSNR | Train SSIM | Validation PSNR | Validation SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.071034 | 25.0279 | 0.7932 | 24.8107 | 0.7838 |
| 10 | 0.062961 | 25.8018 | 0.8290 | 25.5696 | 0.8215 |
| 25 | 0.059049 | 26.4466 | 0.8468 | 26.1394 | 0.8384 |
| 50 | 0.057979 | 26.5861 | 0.8504 | 26.2677 | 0.8420 |

- 最佳 Validation：epoch 50，`PSNR=26.2677`，`SSIM=0.8420`。
- 曲线图：`runs/stereo_sr_x2_paperlike/history_curve.png`。

![论文式 x2 全量训练曲线](runs/stereo_sr_x2_paperlike/history_curve.png)

### 独立评估

Validation 全量评估命令：

```bash
conda run -n dl-lab python scripts/evaluate.py   --config runs/stereo_sr_x2_paperlike/config.json   --checkpoint runs/stereo_sr_x2_paperlike/best.pt   --output runs/stereo_sr_x2_paperlike/eval_validation_results.json   --device cuda:0
```

Validation 结果：`PSNR=26.2678`，`SSIM=0.8420`，`seconds_per_pair=0.0506`，`num_pairs=112`。

Test 全量评估命令：

```bash
conda run -n dl-lab python scripts/evaluate.py   --config runs/stereo_sr_x2_paperlike/config.json   --checkpoint runs/stereo_sr_x2_paperlike/best.pt   --split Test   --output runs/stereo_sr_x2_paperlike/eval_test_results.json   --device cuda:0
```

Test 结果：`PSNR=25.6567`，`SSIM=0.8431`，`seconds_per_pair=0.0505`，`num_pairs=112`。

### 说明

- 这次训练完整使用 Train/Validation/Test split 的全部图像；训练 patch 是随机裁剪，不是整图训练。
- 由于评估尺寸改为 `256x256` HR crop，指标不能和之前 `512x512` eval crop 的 run 逐项等价比较；但速度、batch 规模和训练方式更接近论文代码。

## 待跑实验

| 编号 | 配置 | 数据 | 命令 | 结果 |
| --- | --- | --- | --- | --- |
| E0 | smoke test | 随机张量 | `python scripts/smoke_test.py` | 已通过 |
| E1 | overfit x2 | Flickr1024 单 batch | `python scripts/train.py --config configs/overfit_x2.json` | 已通过：CUDA 30 epoch，L1 0.0966→0.0519，PSNR 27.391→27.523 |
| E2 | mono x2 | Flickr1024 | `python scripts/train.py --config configs/mono_sr_x2_ablation.json` | 待记录 |
| E3 | stereo x2 | Flickr1024 | `python scripts/train.py --config configs/stereo_sr_x2.json` | 已完成：论文式 x2 全量 50 epoch，Val PSNR 26.268/SSIM 0.8420，Test PSNR 25.657/SSIM 0.8431 |
| E4 | stereo x4 | Flickr1024 | `python scripts/train.py --config configs/stereo_sr_x4.json` | 待记录 |


## 2026-05-29 x2/x4 阶段训练结果汇总

> 说明：以下为当前轻量 `StereoSRNet` 与现有数据协议下的阶段结果，不是最终代码结果。远端结果来自 `runs/stereo_sr_x2_paperlike` 和 `runs/stereo_sr_x4_paperlike`。

### 当前统一数据协议

- 训练池：Flickr1024 全部 split + Middlebury2014，合并后固定切分为 train `931` 对、val `103` 对。
- 测试集：KITTI2012 `training + testing`，共 `778` 对。
- 评价：左右视图分别计算 PSNR/SSIM 后汇总，边界按 `scale` 裁剪。

### 训练与评估结果

| Scale | Train input -> output | Batch | Params | Val PSNR | Val SSIM | KITTI eval PSNR | KITTI eval SSIM | Time / pair |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x2 | `48x48 -> 96x96` | 16 | 515,524 | 26.2745 | 0.8457 | 28.2818 | 0.8699 | 0.0079s |
| x4 | `24x24 -> 96x96` | 16 | 598,660 | 22.8478 | 0.6586 | 25.8408 | 0.7610 | 0.0144s |

### 训练曲线摘要

| Scale | Epoch | Train L1 | Val PSNR | Val SSIM |
| --- | ---: | ---: | ---: | ---: |
| x2 | 1 | 0.070802 | 24.7145 | 0.7857 |
| x2 | 10 | 0.062002 | 25.5979 | 0.8263 |
| x2 | 25 | 0.057665 | 26.1509 | 0.8422 |
| x2 | 50 | 0.058275 | 26.2745 | 0.8457 |
| x4 | 1 | 0.101412 | 22.1470 | 0.6139 |
| x4 | 10 | 0.095394 | 22.6397 | 0.6469 |
| x4 | 25 | 0.092387 | 22.7958 | 0.6562 |
| x4 | 50 | 0.091620 | 22.8478 | 0.6586 |


- 曲线图：`runs/summary_x2_x4/x2_x4_validation_curves.svg`。

![x2/x4 阶段验证曲线](runs/summary_x2_x4/x2_x4_validation_curves.svg)

### 与论文/参考基线的阶段性对比

| Method / source | Scale | Params | KITTI2012 PSNR/SSIM | 备注 |
| --- | --- | ---: | --- | --- |
| Ours current StereoSRNet | x2 | 0.516M | 28.2818/0.8699 | 当前统一 KITTI `training+testing` 协议，非最终代码 |
| iPASSR paper baseline | x2 | 1.37M | 31.11/0.9240 | 论文公开 benchmark 协议，不能与当前协议严格逐项等价 |
| Ours current StereoSRNet | x4 | 0.599M | 25.8408/0.7610 | 当前统一 KITTI `training+testing` 协议，非最终代码 |
| StereoSR paper baseline | x4 | 1.42M | 24.53/0.7555 | SwinFSR/CVPRW 2023 汇总表中的 KITTI2012 x4 参考值 |
| PASSRnet paper baseline | x4 | 1.42M | 26.34/0.7981 | 同上 |
| iPASSR paper baseline | x4 | 1.42M | 26.56/0.8053 | 同上 |

### 解读

- x4 的训练尺寸与 SwiniPASSR 参考配置接近：HR patch `96`，LR 输入 `24x24`，batch `16`。
- 当前 x4 指标 `25.8408/0.7610` 已高于 StereoSR x4 参考值，但低于 PASSRnet/iPASSR；考虑到当前模型只有 `0.599M` 参数，且代码还不是最终版，这个阶段结果是可接受的。
- 上表论文基线来自公开 benchmark 汇总，测试集 split、退化生成、评估脚本与本项目当前 `KITTI training+testing` 协议不完全一致；最终报告应优先使用同一代码和同一测试协议下的消融比较。

## 2026-05-30 缺口一 + 缺口二：Swin Transformer 骨干 + 域对齐转换层

### 动机

当前 `StereoSRNet` 使用纯 CNN 残差块 (`ResidualStack`)，约 0.5M 参数，Val PSNR 26.27 (x2)。真正的 SwiniPASSR 使用 SwinIR 的 RSTB 模块（滑动窗口自注意力）+ 域对齐转换层。本次升级旨在填补最核心的两个差距：

1. **缺口一（骨干网络）**：CNN 感受野有限 → Swin Transformer 滑动窗口全局建模（预估 ~2 dB 提升）
2. **缺口二（转换层）**：Swin 特征分布与 CNN 域不对齐 → 引入两个 3×3 卷积转换层

### 参考代码分析

通过阅读三份参考实现提取关键架构信息：

- **SwinIR** (`refs/code/SwinIR/models/network_swinir.py`)：提取 RSTB、BasicLayer、SwinTransformerBlock、WindowAttention、PatchEmbed/UnEmbed 的核心逻辑
- **NTIRE22 SwiniPASSR** (`refs/code/NTIRE22_SSR_SwiniPASSR/models/network_swinipassr.py`)：提取整体 SwiniPASSR 拼装逻辑
- **iPASSR** (`refs/code/iPASSR/model.py`)：biPAM 参考（当前代码已有 `ParallaxAttention` 实现）

#### 参考代码的 SwiniPASSR 前向流程（`network_swinipassr.py` L913–L952）

```python
# 1. 浅层特征提取
first_x_left = conv_first(x_left)           # 3 → embed_dim

# 2. Stage 1: Swin 深层特征提取（前半 RSTB）
catfea_left = forward_features(first_x_left) # RSTB × (N/2)
x_left = conv_after_body(catfea_left)        # 3×3 conv 转换层

# 3. 视差注意力
x_leftT, ... = pam(catfea_left, catfea_right, x_left, x_right)

# 4. 融合 + Stage 2（后半 RSTB）
fused = fusion(cat([catfea_left, x_leftT]))
second_catfea_left = forward_features(fused, second=True)

# 5. 全局残差 + 上采样
x_left = conv_before_upsample(second_conv_after_body(second_catfea_left) + first_x_left)
x_left = conv_last(upsample(x_left)) + bicubic_upscale
```

#### 关键发现（相比用户初始方案的改进）

1. **双段式 RSTB**：不是简单替换 ResidualStack → Swin，而是 **两段 RSTB 夹 PAM**（Stage 1 → PAM → Stage 2）。Stage 2 让 PAM 的跨视图信息得到进一步精炼。
2. **两个转换层**：参考代码有两个 `conv_after_body`，分别在 Stage 1 和 Stage 2 之后，做 Swin→CNN 域对齐。
3. **全局残差**：从浅层特征到最终重建前有一条全局残差捷径 `second_conv_after_body(stage2) + shallow`。

### 架构设计

最终实现的 `SwinStereoSRNet` 架构：

```
LR ──┬── bicubic_upsample ──────────────────────────────────────────────── (+) ── SR
     │                                                                       ↑
     └── pad ── conv_first ──┬── [RSTB×3] ── norm ── conv_after_body₁ ── PAM ── fusion ──
                             │                Stage 1              转换层₁
                             │
                             │  ── [RSTB×3] ── norm ── conv_after_body₂ ── (+shallow) ── crop ── Upsampler
                             │       Stage 2              转换层₂       全局残差
                             └──────────────────────────────────────────────┘
```

### 超参配置

| 参数 | 值 | 来源 |
|---|---|---|
| `embed_dim` | 60 | SwiniPASSR 论文默认值 |
| `depths` | `[6, 6, 6, 6, 6, 6]` | 6 个 RSTB，每个 6 个 STL |
| `num_heads` | `[6, 6, 6, 6, 6, 6]` | 每个 STL 6 个注意力头 |
| `window_size` | 8 | SwiniPASSR `__main__` 示例 |
| `mlp_ratio` | 2.0 | SwiniPASSR `__main__` 示例 |
| `resi_connection` | `"1conv"` | RSTB 内残差用单个 3×3 conv |
| `drop_path_rate` | 0.1 | SwinIR 默认值 |
| RSTB 分配 | Stage 1: 3 个, Stage 2: 3 个 | 参考 `num_layers // 2` 分割 |

### Swin Transformer 核心组件详解

#### WindowAttention（窗口自注意力）

```
输入: (nW*B, window_size², C)
  │
  ├── QKV 线性投影 → Q, K, V     各 (nW*B, nH, N, C/nH)
  ├── Q·K^T / √d + 相对位置偏置   → 注意力矩阵 (nW*B, nH, N, N)
  ├── [可选] SW-MSA 掩码 (对不同窗口的 token 填 -100)
  ├── Softmax → Dropout
  ├── Attention · V
  └── 线性投影 → Dropout → 输出
```

- 相对位置偏置表大小: `(2*ws-1)² × nH = 225 × 6 = 1350` 个可学习参数
- 索引预计算为 buffer，不需要梯度

#### SwinTransformerLayer（单层 STL）

```
输入 x (B, H*W, C)
  │
  ├── LN → reshape (B, H, W, C) → [cyclic shift if SW-MSA]
  ├── window_partition → WindowAttention → window_reverse
  ├── [reverse cyclic shift] → reshape (B, H*W, C)
  ├── (+) 残差连接
  ├── LN → MLP (fc1 → GELU → drop → fc2 → drop)
  └── (+) 残差连接 → 输出
```

- W-MSA 和 SW-MSA 交替出现：偶数层 shift_size=0，奇数层 shift_size=window_size//2

#### RSTB（残差 Swin Transformer 块）

```
输入 x (B, H*W, C) tokens
  │
  ├── BasicLayer: [STL × depth 个，交替 W-MSA/SW-MSA]
  ├── PatchUnEmbed → (B, C, H, W) 空间
  ├── Conv2d 3×3           ← 跨窗口信息流通
  ├── PatchEmbed → (B, H*W, C) tokens
  └── (+) 残差连接 → 输出
```

- 每个 RSTB 内部的 3×3 Conv 是关键：让不同窗口之间的特征可以交流

#### 转换层（Conversion Layer）

在 RSTB 堆和 biPAM 之间的 3×3 卷积：

- **conv_after_body₁**：Stage 1 RSTB 输出 → CNN 域对齐 → 送入 ParallaxAttention
- **conv_after_body₂**：Stage 2 RSTB 输出 → CNN 域对齐 → 全局残差 → 上采样

原理：Swin Transformer 的输出特征分布偏向注意力代数分布（LayerNorm 约束、softmax 归一化），
与传统 CNN 模块（LeakyReLU 激活、不做 token 归一化）的空间几何分布存在差异。
3×3 Conv 充当"缓冲层"，让梯度在两个域之间平滑传播。

### 代码改动清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/stereo_sr_lab/models/swin_blocks.py` | **新增** | Swin 基础模块：DropPath, Mlp, WindowAttention, SwinTransformerLayer, BasicLayer, RSTB, PatchEmbed/UnEmbed。手搓实现，无 timm/skimage 依赖。~370 行。 |
| `src/stereo_sr_lab/models/swin_stereo_sr.py` | **新增** | `SwinStereoSRNet` 类：双段 RSTB + PAM + 两个转换层 + 全局残差 + 窗口填充。与 `StereoSRNet` 同签名。~210 行。 |
| `src/stereo_sr_lab/models/factory.py` | **修改** | 新增 `name == "swin_stereo_sr"` 分支，从 config 读取 Swin 超参并构建模型。 |
| `src/stereo_sr_lab/models/__init__.py` | **修改** | 导出 `SwinStereoSRNet`。 |
| `configs/swin_stereo_sr_x2.json` | **新增** | x2 配置：embed_dim=60, depths=[6,6,6,6,6,6], window_size=8, batch_size=8, img_size=48。 |
| `configs/swin_stereo_sr_x4.json` | **新增** | x4 配置：同上但 scale=4, img_size=24。 |
| `scripts/smoke_test.py` | **修改** | 新增 Swin 模型的前向/反向/损失自检（轻量配置: embed_dim=12, depths=[2,2], window_size=4）。 |

### 参数量对比

| 模型 | Scale | 参数量 | 说明 |
|---|---|---|---|
| StereoSRNet (CNN) | x2 | ~0.516M | 当前基线 |
| StereoSRNet (CNN) | x4 | ~0.599M | 当前基线 |
| **SwinStereoSRNet** | **x2** | **~1.4–1.5M** | 本次新增，接近论文基线规模 |
| **SwinStereoSRNet** | **x4** | **~1.5–1.6M** | 本次新增 |
| SwiniPASSR 论文 | x2/x4 | ~1.4M | 参考值 |

### 关键设计选择与理由

1. **复用现有 `ParallaxAttention`**：当前的 biPAM 实现使用 Q/K/V 投影 + 行级注意力，逻辑清晰且已验证可训练。参考 `ParallaxAttentionModule` 功能类似但代码更复杂（含 M_Relax 遮挡处理）。保持现有实现，减少引入 bug 的风险。
2. **无 timm 依赖**：`DropPath`、`trunc_normal_`、`to_2tuple` 均内联实现，~50 行代码，避免引入外部库。
3. **窗口填充用 reflect 而非 zero**：避免在图像边缘引入人工零值特征。
4. **`PatchEmbed` 带 LayerNorm**：整体阶段的 PatchEmbed 做一次 LayerNorm（参考 `patch_norm=True`），RSTB 内部的 PatchEmbed 不做 norm（纯 reshape）。
5. **`img_size` 仅用于预计算注意力掩码**：不限制实际输入尺寸。评估时若尺寸不同，SwinTransformerLayer 会动态重算 SW-MSA 掩码。

### 运行命令

#### Step 1: Smoke Test（验证前向/反向链路）

```bash
conda run -n dl-lab python scripts/smoke_test.py
```

预期输出：`[CNN] OK ...` 和 `[Swin] OK ...` 两段，最后 `All smoke tests passed!`。

#### Step 2: Overfit 测试（单 batch 过拟合验证学习能力）

```bash
conda run -n dl-lab python scripts/train.py \
  --config configs/swin_stereo_sr_x2.json \
  --output-dir runs/swin_overfit_x2 \
  --mode overfit \
  --epochs 30 \
  --batch-size 1 \
  --device cuda:0
```

预期：loss 持续下降，PSNR 持续上升（应比 CNN 基线 overfit 更快收敛）。

#### Step 3: x2 全量训练

```bash
conda run -n dl-lab python scripts/train.py \
  --config configs/swin_stereo_sr_x2.json \
  --device cuda:0 \
  --eval-train-every 5 \
  --eval-train-limit 0
```

> 如果 OOM，降 batch_size：加 `--batch-size 4`。  
> 如果仍然 OOM，在 config 中设 `"use_checkpoint": true`（梯度检查点，省显存但慢 ~20%）。

#### Step 4: x2 独立评估

```bash
conda run -n dl-lab python scripts/evaluate.py \
  --config runs/swin_stereo_sr_x2/config.json \
  --checkpoint runs/swin_stereo_sr_x2/best.pt \
  --output runs/swin_stereo_sr_x2/eval_results.json \
  --device cuda:0
```

#### Step 5: x4 全量训练

```bash
conda run -n dl-lab python scripts/train.py \
  --config configs/swin_stereo_sr_x4.json \
  --device cuda:0 \
  --eval-train-every 5 \
  --eval-train-limit 0
```

### 预期结果（待跑后填写）

| Scale | Model | Params | Val PSNR | Val SSIM | KITTI PSNR | KITTI SSIM | Time/pair |
|---|---|---:|---:|---:|---:|---:|---:|
| x2 | StereoSRNet (CNN 基线) | 0.516M | 26.2745 | 0.8457 | 28.2818 | 0.8699 | 0.0079s |
| x2 | **SwinStereoSRNet** | ~1.5M | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| x4 | StereoSRNet (CNN 基线) | 0.599M | 22.8478 | 0.6586 | 25.8408 | 0.7610 | 0.0144s |
| x4 | **SwinStereoSRNet** | ~1.5M | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |

预期 Swin 模型在 x2 上 Val PSNR 应达到 27.5–28.5（相比 CNN 基线提升 1.2–2.2 dB）。

