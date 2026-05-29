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

## 待跑实验

| 编号 | 配置 | 数据 | 命令 | 结果 |
| --- | --- | --- | --- | --- |
| E0 | smoke test | 随机张量 | `python scripts/smoke_test.py` | 已通过 |
| E1 | overfit x2 | Flickr1024 单 batch | `python scripts/train.py --config configs/overfit_x2.json` | 已通过：CUDA 30 epoch，L1 0.0966→0.0519，PSNR 27.391→27.523 |
| E2 | mono x2 | Flickr1024 | `python scripts/train.py --config configs/mono_sr_x2_ablation.json` | 待记录 |
| E3 | stereo x2 | Flickr1024 | `python scripts/train.py --config configs/stereo_sr_x2.json` | 阶段性通过：5 epoch/64 train/8 val，best val PSNR 23.735，SSIM 0.7747 |
| E4 | stereo x4 | Flickr1024 | `python scripts/train.py --config configs/stereo_sr_x4.json` | 待记录 |
