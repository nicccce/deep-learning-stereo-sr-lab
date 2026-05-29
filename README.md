# 双目图像超分辨率实验代码

本目录是一个可跑通、可扩展的实验框架：先实现轻量级双目 SR 基线，再提供单目消融、频域损失、PSNR/SSIM/推理时间评估和可视化入口。代码没有直接搬运参考仓库的大文件，核心网络用 PyTorch 重新组织为小模块，后续可以把 SwinIR、Mamba 或更强的融合块逐步替换进去。

## 代码结构

```text
deep-learning-stereo-sr-lab/
  configs/                  # x2/x4/单目消融/单 batch 过拟合配置
  scripts/
    prepare_data.py          # 解压本地数据包
    train.py                 # 训练入口
    evaluate.py              # PSNR、SSIM、推理时间评估
    infer.py                 # 单张或数据集推理
    smoke_test.py            # 随机张量快速自检
  src/stereo_sr_lab/
    data/                    # Flickr1024、Middlebury、KITTI 扫描和 Dataset
    models/                  # 残差块、双向视差注意力、StereoSR/MonoSR
    training/                # 损失、指标、训练/验证循环
  tools/
    visualize_attention.py   # 导出视差注意力矩阵
    visualize_spectrum.py    # 导出傅里叶频谱图
  EXPERIMENT_LOG.md          # 实验过程记录
```

## 数据准备

在本目录执行：

```bash
conda activate dl-lab
python scripts/prepare_data.py --data-root ../data --out-root datasets
```

默认会解压：

- `../data/flickr1024/Flickr1024.zip` 到 `datasets/flickr1024/Flickr1024`
- `../data/middlebury_2014/perfect_train/*.zip` 到 `datasets/middlebury_2014/perfect_train`
- `../data/kitti/data_depth_selection.zip` 到 `datasets/kitti`

当前训练配置默认使用 Flickr1024。Middlebury 可以作为额外验证或少量微调数据；KITTI depth selection 包含 depth-completion 结构，部分 split 不是标准左右图 SR 标注，因此本框架主要用它做推理、计时或合成 LR 的展示测试。

## 快速自检

```bash
python scripts/smoke_test.py
```

这个脚本会创建一个小模型，跑一次前向、损失、PSNR 和 SSIM，适合确认环境与代码路径没有问题。

## 两种运行状态

框架支持两个状态，便于在本地和 A800 上分工排查：

- `overfit`：固定一组训练 batch 反复训练，用来验证数据读取、模型前向、损失和反传是否真的能学。若这个状态下 loss 不下降或 PSNR 不上升，优先查代码、尺度、损失和数据配对。
- `train`：使用完整训练/验证 split 正常训练，用于正式实验和报告结果。

单 batch 过拟合验证：

```bash
python scripts/train.py --config configs/overfit_x2.json
```

也可以在任意配置上临时切换：

```bash
python scripts/train.py --config configs/stereo_sr_x2.json --mode overfit --epochs 200 --batch-size 1
```

正式训练则使用配置里的默认 `run_mode: train`，或显式传入 `--mode train`。

## 训练

x2 双目基线：

```bash
python scripts/train.py --config configs/stereo_sr_x2.json
```

x4 双目基线：

```bash
python scripts/train.py --config configs/stereo_sr_x4.json
```

单目消融：

```bash
python scripts/train.py --config configs/mono_sr_x2_ablation.json
```

常用覆盖参数：

```bash
python scripts/train.py \
  --config configs/stereo_sr_x2.json \
  --data-root datasets/flickr1024/Flickr1024 \
  --output-dir runs/debug_x2 \
  --epochs 2 \
  --batch-size 2 \
  --limit-train 20 \
  --limit-val 5
```

训练输出保存在 `runs/...`，包括 `config.json`、`history.json`、`latest.pt` 和 `best.pt`。

## 评估

```bash
python scripts/evaluate.py \
  --config configs/stereo_sr_x2.json \
  --checkpoint runs/stereo_sr_x2_light_ffl/best.pt \
  --output runs/stereo_sr_x2_light_ffl/eval_results.json
```

输出指标包括平均 PSNR、SSIM、每对图像推理时间和参数量。默认对 Validation 前 20 对做评估；修改配置里的 `limit_val` 或命令行 `--limit` 可以扩大范围。

## 推理与可视化

对一对左右图推理：

```bash
python scripts/infer.py \
  --config configs/stereo_sr_x2.json \
  --checkpoint runs/stereo_sr_x2_light_ffl/best.pt \
  --left path/to/left.png \
  --right path/to/right.png \
  --out-dir outputs/demo
```

导出注意力矩阵：

```bash
python tools/visualize_attention.py \
  --config configs/stereo_sr_x2.json \
  --checkpoint runs/stereo_sr_x2_light_ffl/best.pt \
  --left path/to/lr_left.png \
  --right path/to/lr_right.png \
  --output outputs/attention_row.png
```

导出频谱图：

```bash
python tools/visualize_spectrum.py outputs/demo/sr_left.png path/to/hr_left.png --out-dir outputs/spectra
```

## 实验建议

建议先跑 `configs/overfit_x2.json` 看单 batch 是否能明显过拟合，再用 `--limit-train 20 --limit-val 5 --epochs 1` 跑通全流程，最后取消 limit 正式训练。报告里建议至少对比三组：单目 SR、双目 SR、双目 SR + FFL。若显存紧张，优先减小 `batch_size`、`hr_patch_size`、`channels` 或 `eval_crop_size`。
