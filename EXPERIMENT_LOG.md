# 实验过程记录

> 用于实时记录实验命令、配置变化、结果和观察。每次训练或评估后追加一条，后续写实验报告时可以直接整理这里的内容。

## 2026-05-28 框架搭建

- 任务：双目图像超分辨率重建。
- 数据：Flickr1024 作为默认训练/验证集；Middlebury 作为可选补充；KITTI depth selection 用作推理、计时或合成 LR 测试，不作为严格双目 SR 标准测试标注。
- 基线：`StereoSRNet = 共享残差特征提取 + 双向视差注意力 + 融合重建 + PixelShuffle 上采样 + Bicubic 残差连接`。
- 消融：`SwinMonoSRNet` 使用与 `SwinStereoSRNet` 对齐的 Swin 两阶段骨干，只移除跨视图 PAM，用于验证双目融合是否有效。
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
- 后续正式结果建议在相同记录格式下扩大到完整 `configs/stereo_sr_x2.json`，再补 `swin_mono_sr_x2_ablation` 和 `stereo_sr_x4` 对照。

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
| E2 | swin mono x2 | Flickr1024 + Middlebury2014 | `python scripts/train.py --config configs/swin_mono_sr_x2_ablation.json` | 远端待跑 |
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

1. **缺口一（骨干网络）**：CNN 感受野有限 → Swin Transformer 滑动窗口全局建模，验证全局建模是否能带来收益
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
| `configs/swin_stereo_sr_x2.json` | **新增** | x2 配置：embed_dim=60, depths=[6,6,6,6,6,6], window_size=8, batch_size=16, img_size=48。 |
| `configs/swin_stereo_sr_x4.json` | **新增** | x4 配置：同上但 scale=4, batch_size=16, img_size=24。 |
| `scripts/smoke_test.py` | **修改** | 新增 Swin 模型的前向/反向/损失自检（轻量配置: embed_dim=12, depths=[2,2], window_size=4）。 |

### 参数量对比

| 模型 | Scale | 参数量 | 说明 |
|---|---|---|---|
| StereoSRNet (CNN) | x2 | ~0.516M | 当前基线 |
| StereoSRNet (CNN) | x4 | ~0.599M | 当前基线 |
| **SwinStereoSRNet** | **x2** | **1.582M** | 本次新增，接近论文基线规模 |
| **SwinStereoSRNet** | **x4** | **1.712M** | 本次新增，x4 上采样头使参数量略高 |
| SwiniPASSR 论文 | x2/x4 | ~1.4M | 参考值 |

### 关键设计选择与理由

1. **复用现有 `ParallaxAttention`**：当前的 biPAM 实现使用 Q/K/V 投影 + 行级注意力，逻辑清晰且已验证可训练。参考 `ParallaxAttentionModule` 功能类似但代码更复杂（含 M_Relax 遮挡处理）。保持现有实现，减少引入 bug 的风险。
2. **无 timm 依赖**：`DropPath`、`trunc_normal_`、`to_2tuple` 均内联实现，~50 行代码，避免引入外部库。
3. **窗口填充用 reflect 而非 zero**：避免在图像边缘引入人工零值特征。
4. **`PatchEmbed` 带 LayerNorm**：整体阶段的 PatchEmbed 做一次 LayerNorm（参考 `patch_norm=True`），RSTB 内部的 PatchEmbed 不做 norm（纯 reshape）。
5. **`img_size` 仅用于预计算注意力掩码**：不限制实际输入尺寸。评估时若尺寸不同，SwinTransformerLayer 会动态重算 SW-MSA 掩码。

### 实际验证与远端结果

结果来源：

- 远端目录：`~/dl-lab/deep-learning-stereo-sr-lab/runs/swin_stereo_sr_x2`
- 远端目录：`~/dl-lab/deep-learning-stereo-sr-lab/runs/swin_stereo_sr_x4`
- 已同步到本机：`runs/swin_stereo_sr_x2`、`runs/swin_stereo_sr_x4`
- 结果文件：`config.json`、`history.json`、`eval_results.json`、`best.pt`、`latest.pt`

本机 smoke test 已通过，`scripts/smoke_test.py` 同时覆盖 CNN 与 Swin 两条路径，前向、损失、反传、PSNR/SSIM 计算均可运行。

### 训练配置

| Scale | Model | Epochs | Batch | Train patch | Eval crop | Device | AMP |
|---|---|---:|---:|---|---:|---|---|
| x2 | SwinStereoSRNet | 50 | 16 | `48x48 -> 96x96` | 256 | `cuda:0` | true |
| x4 | SwinStereoSRNet | 50 | 16 | `24x24 -> 96x96` | 0 | `cuda:0` | true |

训练池仍为 Flickr1024 全部 split + Middlebury2014，固定切分为 train/val；独立评估仍使用 KITTI2012 `training + testing`，共 `778` 对，协议与 2026-05-29 的 CNN 基线一致。

### 真实评估结果

| Scale | Model | Params | Val PSNR | Val SSIM | KITTI PSNR | KITTI SSIM | Time/pair |
|---|---|---:|---:|---:|---:|---:|---:|
| x2 | StereoSRNet (CNN 基线) | 0.516M | 26.2745 | 0.8457 | 28.2818 | 0.8699 | 0.0079s |
| x2 | **SwinStereoSRNet** | **1.582M** | **26.2801** | **0.8454** | **28.2809** | **0.8700** | **0.3677s** |
| x4 | StereoSRNet (CNN 基线) | 0.599M | 22.8478 | 0.6586 | 25.8408 | 0.7610 | 0.0144s |
| x4 | **SwinStereoSRNet** | **1.712M** | **22.8441** | **0.6586** | **25.8552** | **0.7616** | **0.5699s** |

### 训练曲线摘要

| Scale | Epoch | Train L1 | Train total | Val PSNR | Val SSIM |
|---|---:|---:|---:|---:|---:|
| x2 | 1 | 0.092979 | 0.093446 | 24.5686 | 0.7680 |
| x2 | 50 | 0.057335 | 0.058036 | 26.2801 | 0.8454 |
| x4 | 1 | 0.104462 | 0.105416 | 22.1432 | 0.6129 |
| x4 | 50 | 0.092571 | 0.093549 | 22.8441 | 0.6586 |

- x2：50 epoch 内 Val PSNR 从 `24.5686` 提升到 `26.2801`，提升 `+1.7114 dB`；训练 total loss 下降约 `37.9%`。最佳 PSNR 出现在 epoch 50，最佳 SSIM 出现在 epoch 48。
- x4：Val PSNR 从 `22.1432` 提升到 `22.8441`，提升 `+0.7009 dB`；训练 total loss 下降约 `11.3%`。最佳 PSNR 出现在 epoch 50，最佳 SSIM 出现在 epoch 49。
- 两个 scale 的验证曲线仍在后期缓慢上升或趋于平台，未出现明显验证集崩塌，说明训练链路正常；x4 收敛更慢，主要受更大上采样倍数和更少 LR 细节约束影响。

### 与 CNN 基线的真实差异

| Scale | Val PSNR Δ | Val SSIM Δ | KITTI PSNR Δ | KITTI SSIM Δ | Params ratio | Time ratio |
|---|---:|---:|---:|---:|---:|---:|
| x2 | +0.0056 | -0.0003 | -0.0009 | +0.0001 | 3.07x | 46.54x |
| x4 | -0.0037 | +0.0000 | +0.0144 | +0.0006 | 2.86x | 39.57x |

### 结论分析

- 真实结果没有达到原先对 Swin x2 的乐观预期。x2 在 Val 和 KITTI 上与轻量 CNN 基线基本打平，PSNR/SSIM 差异都处在极小范围内，不能据此宣称 Swin 骨干带来显著质量提升。
- x4 在 KITTI 上有 `+0.0144 dB / +0.0006 SSIM` 的小幅提升，但验证集 PSNR 略低于 CNN。这个幅度更像是协议和随机波动内的边际收益，结论应写成“x4 有轻微正向趋势”，不宜夸大。
- Swin 模型参数量约为 CNN 的 `2.86x-3.07x`，推理时间约为 `39.57x-46.54x`。在当前实现和训练设置下，性能收益远不足以抵消推理开销。
- 这次结果更适合作为“结构复现和训练链路跑通”的证据：Swin/RSTB、转换层、PAM 融合和上采样全流程可以稳定训练，但当前融合设计、训练策略或数据协议还没有释放出 Transformer 骨干的质量优势。
- 远端结果目录没有保存单图预测图片，因此当前“预测/推理”分析以 `eval_results.json` 中的 KITTI 全量指标和 `seconds_per_pair` 为准。若报告需要视觉对比图，应另跑 `scripts/infer.py` 从 `best.pt` 生成固定样例。

## 2026-05-30 Swin 同框架单目消融 baseline

### 目标

为判断当前双目 PAM / fusion 模块是否真的提供跨视图收益，新增与 `SwinStereoSRNet` 尽可能对齐的单目消融模型 `SwinMonoSRNet`。旧的 CNN `MonoSRNet` 已删除，避免拿不同骨干的单目 baseline 与 Swin 双目模型硬比。

### 结构对齐方式

`SwinMonoSRNet` 继承 `SwinStereoSRNet` 的主干结构：

```text
LR -> bicubic residual
   -> conv_first
   -> Stage 1 RSTB
   -> conv_after_body1
   -> fusion([conv_feat, zero_context])
   -> Stage 2 RSTB
   -> conv_after_body2 + shallow residual
   -> PixelShuffle upsampler
   -> SR
```

与 `SwinStereoSRNet` 的唯一区别是移除 `ParallaxAttention`，默认 `fusion_context="zero"`，即 fusion 第二路输入为零特征，保证模型不能读取右图信息。左右图仍分别输出 `sr_left` / `sr_right`，接口保持与训练、评估脚本一致。

### 代码与配置变化

| 文件 | 变化 |
|---|---|
| `src/stereo_sr_lab/models/swin_mono_sr.py` | 新增 `SwinMonoSRNet`，复用 Swin 双目骨干并移除 PAM。 |
| `src/stereo_sr_lab/models/factory.py` | 新增 `name == "swin_mono_sr"` 分支，删除旧 `mono_sr` 分支。 |
| `src/stereo_sr_lab/models/__init__.py` | 导出 `SwinMonoSRNet`，移除 `MonoSRNet`。 |
| `configs/swin_mono_sr_x2_ablation.json` | 新增 x2 单目消融配置；`data`、`train` 和 Swin 超参与 `swin_stereo_sr_x2.json` 对齐。 |
| `configs/swin_mono_sr_x4_ablation.json` | 新增 x4 单目消融配置；与 `swin_stereo_sr_x4.json` 对齐。 |
| `configs/mono_sr_x2_ablation.json` / `src/stereo_sr_lab/models/mono_sr.py` | 删除旧 CNN 单目 baseline。 |
| `scripts/smoke_test.py` | 新增 `SwinMonoSRNet` 前向、loss、反传自检。 |

### 本机验证

- `conda run -n dl-lab python scripts/smoke_test.py`：通过，覆盖 CNN `StereoSRNet`、`SwinStereoSRNet` 和 `SwinMonoSRNet`。
- 参数量：x2 `SwinMonoSRNet = 1,567,323`，x2 `SwinStereoSRNet = 1,581,964`，差异主要来自移除 PAM 的 Q/K/V/proj 和 logit scale。
- 本机 CUDA 不可用：`torch.cuda.is_available() == False`，`nvidia-smi` 报 `Failed to initialize NVML: Unknown Error`。CPU 探针可以跑通训练入口，但不适合完成 50 epoch 全量实验；正式 x2 单目训练改为推到远端 GPU 运行后再拉回分析。

### 远端待跑命令

```bash
cd ~/dl-lab/deep-learning-stereo-sr-lab
git pull origin master
conda run -n dl-lab python scripts/train.py \
  --config configs/swin_mono_sr_x2_ablation.json \
  --device cuda:0
conda run -n dl-lab python scripts/evaluate.py \
  --config runs/swin_mono_sr_x2_ablation/config.json \
  --checkpoint runs/swin_mono_sr_x2_ablation/best.pt \
  --device cuda:0 \
  --output runs/swin_mono_sr_x2_ablation/eval_results.json
```

### 远端训练与回传

远端环境：`ssh -p 20058 u2605173@211.87.224.135`，GPU `Tesla V100-SXM2-32GB`，PyTorch `2.7.1+cu118`。代码通过 GitHub 同步到远端，训练后已将结果目录拉回本机：`runs/swin_mono_sr_x2_ablation`。

远端实际命令：

```bash
cd ~/dl-lab/deep-learning-stereo-sr-lab
git pull origin master
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl-lab
python scripts/train.py --config configs/swin_mono_sr_x2_ablation.json --device cuda:0
python scripts/evaluate.py \
  --config runs/swin_mono_sr_x2_ablation/config.json \
  --checkpoint runs/swin_mono_sr_x2_ablation/best.pt \
  --device cuda:0 \
  --output runs/swin_mono_sr_x2_ablation/eval_results.json
```

### 训练结果

| Model | Scale | Epochs | Params | Best Val Epoch | Val PSNR | Val SSIM | Val Time / pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwinMonoSRNet | x2 | 50 | 1,567,323 | 50 | 26.2955 | 0.8458 | 0.3932s |
| SwinStereoSRNet | x2 | 50 | 1,581,964 | 50 | 26.2801 | 0.8454 | 0.3677s* |

`*` SwinStereoSRNet 的 time/pair 来自 KITTI eval 记录；原日志未单独列出 Val time。

训练曲线摘要：`SwinMonoSRNet` Val PSNR 从 epoch 1 的 `24.604` 稳步上升到 epoch 50 的 `26.2955`；后 10 个 epoch 基本进入平台期，说明 50 epoch 设置足够得到稳定阶段结果。

### KITTI 同协议测试对比

| Model | Params | KITTI PSNR | KITTI SSIM | Time / pair | Num pairs |
|---|---:|---:|---:|---:|---:|
| SwinStereoSRNet | 1,581,964 | 28.2809 | 0.8700 | 0.3677s | 778 |
| SwinMonoSRNet | 1,567,323 | 28.3074 | 0.8704 | 0.3580s | 778 |
| Mono - Stereo | -14,641 | +0.0265 | +0.0004 | -0.0097s | 0 |

### 结论

这次同框架单目消融没有证明当前双目 PAM / fusion 模块有显著正作用。相反，在相同训练数据、相同 epoch、相同 Swin 骨干、相同 KITTI `training + testing` 评估协议下，`SwinMonoSRNet` 的 KITTI PSNR/SSIM 略高于 `SwinStereoSRNet`，推理也略快。

差异幅度很小，仍可能落在单次训练随机波动范围内，因此不宜写成“单目显著优于双目”。但更重要的是：当前结果也不能支持“这个双目模块带来了显著收益”。报告里更稳妥的表述是：**当前 PAM/fusion 设计在本协议下没有体现出可观增益，需要进一步改进注意力约束、遮挡/有效性建模或训练策略后再验证。**

## 2026-05-30 iPASSR 同协议重训与评估 baseline

### 目标

为避免直接拿 iPASSR 论文表格与当前实验硬比，本次在 `refs/code/iPASSR` 下新增适配脚本，使 iPASSR 使用本项目的统一数据协议和评估口径进行重训与复评。该结果可作为经典双目 SR baseline，和当前 `StereoSRNet`、`SwinStereoSRNet` 放在同一张表中公平比较。

### 新增适配脚本

| 文件 | 作用 |
|---|---|
| `refs/code/iPASSR/train_stereo_sr_lab.py` | 复用 iPASSR `model.Net`，但使用 `deep-learning-stereo-sr-lab` 的 config、数据扫描、随机 crop、bicubic LR 生成、train/val split 和 checkpoint 记录方式进行训练。 |
| `refs/code/iPASSR/evaluate_stereo_sr_lab.py` | 复用 iPASSR `model.Net`，但使用当前最终 `evaluate.py` 的同协议指标：KITTI `training + testing`、左右视图共同统计、`crop_border=scale`、PSNR/SSIM、`seconds_per_pair` 和参数量。 |

没有修改 iPASSR 原始 `model.py`、`train.py`、`test.py`。适配脚本中只额外处理了 iPASSR 原代码顶部的可选依赖导入问题，使评估/训练不依赖未使用的 `matplotlib`、`skimage`。

### 数据协议

- 训练池：读取 `deep-learning-stereo-sr-lab/configs/stereo_sr_x2.json` / `stereo_sr_x4.json` 中的 `data.train_sources`。
- 训练数据：Flickr1024 全部 split + Middlebury2014，合并后按 `split_seed=42` 和 `val_ratio=0.1` 固定切分为 train `931` 对、val `103` 对。
- LR 生成：不使用 iPASSR 原始 MATLAB 预生成 patch，而是复用 `StereoSRDataset`，从 HR 左右图现场 bicubic downsample 得到 LR。
- 测试集：读取 config 中的 `data.test_source`，即 KITTI2012 `training + testing`，共 `778` 对。
- 评价：左右视图分别计算 PSNR/SSIM 后汇总，边界按 `scale` 裁剪；推理时间统计为 `seconds_per_pair`。

### 远端训练与同步

远端训练入口：

```bash
cd ~/dl-lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl-lab

python refs/code/iPASSR/train_stereo_sr_lab.py \
  --scale 2 \
  --config deep-learning-stereo-sr-lab/configs/stereo_sr_x2.json \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 16 \
  --amp

python refs/code/iPASSR/train_stereo_sr_lab.py \
  --scale 4 \
  --config deep-learning-stereo-sr-lab/configs/stereo_sr_x4.json \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 16 \
  --amp
```

远端评估入口：

```bash
python refs/code/iPASSR/evaluate_stereo_sr_lab.py \
  --scale 2 \
  --config deep-learning-stereo-sr-lab/configs/stereo_sr_x2.json \
  --checkpoint refs/code/iPASSR/runs/ipassr_x2_retrain/best.pt \
  --device cuda:0 \
  --output refs/code/iPASSR/runs/ipassr_x2_retrain/eval_results.json

python refs/code/iPASSR/evaluate_stereo_sr_lab.py \
  --scale 4 \
  --config deep-learning-stereo-sr-lab/configs/stereo_sr_x4.json \
  --checkpoint refs/code/iPASSR/runs/ipassr_x4_retrain/best.pt \
  --device cuda:0 \
  --output refs/code/iPASSR/runs/ipassr_x4_retrain/eval_results.json
```

结果已从远端同步回本机：

- `refs/code/iPASSR/runs/ipassr_x2_retrain`
- `refs/code/iPASSR/runs/ipassr_x4_retrain`

每个目录包含 `config.json`、`history.json`、`best.pt`、`latest.pt`、`eval_results.json`。

### 训练配置

| Scale | Model | Epochs | Batch | Train patch | Eval crop | Optimizer | AMP |
|---|---|---:|---:|---|---:|---|---|
| x2 | iPASSR | 50 | 16 | `48x48 -> 96x96` | 256 | Adam, lr `2e-4`, StepLR | true |
| x4 | iPASSR | 50 | 16 | `24x24 -> 96x96` | 0 | Adam, lr `2e-4`, StepLR | true |

训练 loss 沿用 iPASSR 原论文代码中的组合：`loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)`。

### 验证与 KITTI 评估结果

| Scale | Params | Best Val Epoch | Val PSNR | Val SSIM | KITTI PSNR | KITTI SSIM | Time / pair | Num pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x2 | 1,377,931 | 49 | 26.6243 | 0.8550 | 28.4171 | 0.8732 | 0.0196s | 778 |
| x4 | 1,427,851 | 46 | 22.8742 | 0.6619 | 25.8990 | 0.7645 | 0.0337s | 778 |

### 与当前模型的同协议对比

| Method | Scale | Params | KITTI PSNR | KITTI SSIM | Time / pair |
|---|---|---:|---:|---:|---:|
| StereoSRNet (CNN 基线) | x2 | 515,524 | 28.2818 | 0.8699 | 0.0079s |
| SwinStereoSRNet | x2 | 1,582,000 左右 | 28.2809 | 0.8700 | 0.3677s |
| **iPASSR 重训** | x2 | **1,377,931** | **28.4171** | **0.8732** | **0.0196s** |
| StereoSRNet (CNN 基线) | x4 | 598,660 | 25.8408 | 0.7610 | 0.0144s |
| SwinStereoSRNet | x4 | 1,712,000 左右 | 25.8552 | 0.7616 | 0.5699s |
| **iPASSR 重训** | x4 | **1,427,851** | **25.8990** | **0.7645** | **0.0337s** |

### 结论

- 在当前统一协议下，重训 iPASSR 在 x2 和 x4 的 KITTI PSNR/SSIM 均略高于当前 `StereoSRNet` 和 `SwinStereoSRNet`。
- iPASSR 参数量约 `1.38M/1.43M`，推理时间明显慢于轻量 CNN，但远快于当前 Swin 版本。
- 这组结果比直接引用 iPASSR 论文指标更适合作为报告中的公平 baseline：同训练数据、同 LR 退化、同 KITTI 测试源、同 PSNR/SSIM/时间统计方式。

