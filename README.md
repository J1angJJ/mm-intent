# 多模态 AR 交互意图理解

本项目面向 AR 眼镜交互场景，研究如何利用多源传感器数据识别用户交互意图。任务输入包含 IMU、手势视频、音频、ASR 文本和场景视觉五类模态，输出为 6 类意图与 2 类场景组合而成的 12 类联合标签。

项目在 Given Baseline / Given Improved Baseline 的基础上，完成了全量数据处理、特征提取、训练评估、模态缺失与噪声鲁棒性分析，并进一步引入基于 MediaPipe hand landmarks 的手部几何时序特征。该特征在主测试集和文本缺失条件下均带来了明显提升，是当前最稳定的改进方向。

## 任务定义

意图类别：

```text
menu, select, magnify, narrow, brush, cancel
```

场景类别：

```text
office, museum
```

主任务为 12 类联合分类：

```text
{office, museum} x {menu, select, magnify, narrow, brush, cancel}
```

评价指标：

| 指标 | 含义 |
|---|---|
| `joint_acc` | 场景和意图同时预测正确 |
| `intent_acc` | 只评价 6 类意图预测 |
| `scene_acc` | 只评价 2 类场景预测 |

## 数据与模态

| 模态 | 数据来源 | 特征表示 |
|---|---|---|
| IMU | HoloLens/同步传感器数据 | 10 帧时序 IMU 特征 |
| Gesture | fisheye 视频 | MediaPipe 手部检测/裁剪 + CLIP 视觉特征，或 hand landmarks 几何特征 |
| Audio | HoloLens 视频音频 | MFCC |
| Text | ASR 转写文本 | Whisper + SentenceTransformer |
| Scene | fisheye 视频 | ViT 场景视觉特征 |

特征统一对齐到检测出的交互片段，每个片段使用固定长度时序窗口。数据集和模型缓存体积较大，默认不进入 git；仓库主要保留代码、轻量图表和复现实验入口。

## 方法概览

### Given Baseline

课程给定程序包含基础多模态分类流程：对各模态分别提取特征，按交互片段对齐后送入多模态融合网络，完成 12 类联合分类。其中手势模态本身已经使用 MediaPipe hands/HandLandmarker 定位手部区域，再对手部裁剪图提取 CLIP 视觉特征；当本地 MediaPipe 版本不兼容或模型文件缺失时，代码才会退化为整帧 CLIP fallback。

### Given Improved Baseline

Given Improved Baseline 在 Given Baseline 上加入了更强的融合结构：

- 以 Gesture / Text / Scene 作为主锚点进行融合。
- 通过 Perceiver-style latent tokens 汇聚多模态信息。
- 使用 IMU / Audio 残差辅助注入。
- 引入 modality gate、intent/scene 辅助头、gesture intent 辅助监督。
- 使用 label smoothing、dropout、weight decay、gradient clipping 等训练稳定策略。

该模型在完整测试集上已经达到较高准确率，但诊断显示它对文本语义依赖较强：当 Text 模态缺失时，性能下降明显。

### Ours: Hand Geometry

为降低模型对 ASR 文本的依赖，项目新增了基于 MediaPipe hand landmarks 的显式手部几何时序特征：

```text
code/feature_extraction/extract_hand_geometry_features.py
```

每个交互片段抽取 `(10, 96)` 维手部几何序列，包含：

- 手部是否存在、中心点、bbox 尺寸。
- 21 个 hand landmarks 的相对坐标。
- 指尖到手腕距离。
- 拇指与其他指尖的 pinch 距离。
- 掌宽、手高等尺度归一化几何量。

与给定流程中的“MediaPipe 手部裁剪 + CLIP 视觉编码”相比，Hand Geometry 不再把 landmarks 只作为裁剪依据，而是直接把关键点相对位置、尺度和指尖关系作为时序输入。因此它更关注动作轨迹和手形结构，尤其适合区分 `brush`、`magnify`、`select` 等细粒度交互动作。

### Ours: Factorized Heads

为恢复 Hand Geometry 在显式 Scene 模态缺失时的场景判别能力，进一步采用分头晚期融合：

```text
Hand Geometry checkpoint  -> intent logits
Given Improved Baseline   -> scene logits
joint logits = intent logits + scene logits
```

该方法不在输入层拼接两种手势特征，而是让几何分支负责动作意图，让原 MediaPipe-cropped CLIP 分支提供场景上下文。它是针对显式 Scene 缺失的扩展方法，主方法仍为单模型 Hand Geometry。

## 实验结果

### 主结果

![主结果对比](docs/figures/main_results_methods.png)

| 模型 / 特征 | joint_acc | intent_acc | scene_acc |
|---|---:|---:|---:|
| Given Baseline | 0.9516 | 0.9637 | 0.9879 |
| Given Improved Baseline | 0.9839 | 0.9839 | 1.0000 |
| Ours: Hand Geometry | **0.9946** | **0.9946** | **1.0000** |
| Ours: Factorized Heads | **0.9946** | **0.9946** | **1.0000** |

Ours: Hand Geometry 在主测试集上只错 4 个样本，12 类联合分类准确率达到 `0.9946`。Factorized Heads 保持相同完整模态性能，主要收益体现在显式 Scene 模态缺失场景。

![方法定位](docs/figures/method_positioning.png)

![Hand Geometry 混淆矩阵](docs/figures/hand_geometry_confusion_matrix.png)

![Hand Geometry 训练曲线](docs/figures/hand_geometry_loss_curve.png)

### 模态缺失鲁棒性

![模态缺失鲁棒性](docs/figures/hand_geometry_robustness_missing.png)

| 缺失设置 | Given Improved Baseline | Ours: Hand Geometry | 变化 |
|---|---:|---:|---:|
| no_text | 0.4933 | **0.7675** | +0.2742 |
| no_audio_text | 0.3723 | **0.7473** | +0.3750 |
| no_imu_text | 0.4462 | **0.8091** | +0.3629 |
| no_gesture_text | 0.5672 | **0.6465** | +0.0793 |
| no_scene | **0.9785** | 0.8293 | -0.1492 |

结果说明：Hand Geometry 显著增强了文本缺失条件下的意图识别能力。即使没有 ASR 文本，模型仍能从手部轨迹中恢复大量动作信息。

需要注意的是，Hand Geometry 对 `no_scene` 条件不占优。该设置下 intent 准确率仍有 `0.9933`，但 scene 准确率下降到 `0.8333`，导致 joint 准确率降低。这说明手部几何主要增强动作意图，不替代场景识别。

### 噪声鲁棒性

![噪声鲁棒性](docs/figures/hand_geometry_robustness_noise.png)

| 噪声设置 | Given Improved Baseline | Ours: Hand Geometry |
|---|---:|---:|
| gesture_noise_60 | **0.9879** | 0.9772 |
| text_noise_60 | 0.9718 | **0.9946** |
| audio_noise_60 | 0.9839 | **0.9946** |
| imu_noise_60 | 0.9839 | **0.9946** |
| scene_noise_60 | 0.9839 | **0.9933** |

Ours: Hand Geometry 在大多数噪声设置下保持在 `0.99` 左右。文本高噪声下仍能保持主结果水平，进一步说明动作几何特征缓解了模型对文本模态的依赖。

### 泛化分析

![泛化分析](docs/figures/factorized_generalization.png)

| 泛化设置 | Given Improved Baseline | Ours: Hand Geometry | Ours: Factorized Heads | Factorized Heads (No Explicit Scene) |
|---|---:|---:|---:|---:|
| 3-seed mean | 0.9615 | 0.9901 | **0.9906** | 0.9875 |
| 4-date holdout mean | 0.9550 | **0.9746** | 0.9740 | 0.9620 |

随机种子和按采集日期整批留出的结果表明，Hand Geometry 的提升不只存在于默认 seed 42。日期留出中最困难的 `2026-01-31` 组 joint accuracy 为 `0.9387`，说明模型仍受采集批次、动作表现和文本分布变化影响。

Ours: Factorized Heads 使用 Hand Geometry checkpoint 的 intent head 和 Given Improved Baseline 的 scene head。它在不降低完整模态主结果的情况下，将默认 split 的 `no_scene` joint accuracy 恢复到 `0.9933`。这里的 `no_scene` 指移除显式 ViT scene feature；MediaPipe 裁剪后的 CLIP gesture 仍可能保留部分背景信息，因此该结果应解释为“显式 Scene 模态缺失鲁棒性”，而不是完全无场景视觉输入。

此外，前述模态缺失和噪声表格中的模型是在对应扰动条件下重新训练的。使用干净 checkpoint 直接面对测试时突发文本噪声时性能仍会明显下降，说明任意 test-time OOD 鲁棒性仍是后续工作。

## 尝试过程

项目中还尝试了多种模型侧和特征侧改进：

| 尝试 | 结论 |
|---|---|
| Hierarchical Margin Loss | 可作为辅助探索，但提升不稳定 |
| Focal Loss / Missing Modality Distillation | 未稳定超过 Given Improved Baseline |
| Supervised Contrastive Loss / Prototype 分类 / Ensemble | 整体收益有限 |
| ASR 文本模板增强 | 降低主任务准确率，可能稀释语义空间 |
| MediaPipe-cropped CLIP gesture + Hand Geometry 早期拼接 | 主任务降至 0.9704，不如纯 Hand Geometry |
| Ours: Hand Geometry 替换 gesture 特征 | 当前最有效，主任务 0.9946，文本缺失鲁棒性明显提升 |
| Ours: Factorized Heads | 保持完整模态性能，并恢复显式 Scene 缺失下的场景判别 |

这些结果表明，在当前数据集上继续堆叠融合头或损失函数的收益有限；更有效的方向是改进动作本身的表示，使模型获得更直接的手部轨迹证据。

## 代码结构

```text
code/
  train.py                               # 统一训练入口
  test.py                                # 测试与报告读取入口
  baseline_real_scene.py                 # baseline 训练与评估
  train_and_test.py                      # improved 训练与评估
  run_missing_experiments.py             # 模态缺失实验
  run_noise_experiments.py               # 噪声实验
  run_gesture_geometry_suite.py          # hand geometry 一键实验
  run_gesture_fusion_suite.py            # CLIP + geometry 拼接实验
  summarize_robustness_results.py        # 鲁棒性结果汇总与可视化
  visualize_method_comparison.py         # 方法主结果、定位与泛化可视化
  feature_extraction/
    get_timestamp.py
    strong_gesture2.0.py
    extract_hand_geometry_features.py
    ASR.py
    mfcc.py
    imu.py

docs/figures/
  hand_geometry_robustness_main.png
  hand_geometry_robustness_missing.png
  hand_geometry_robustness_noise.png
  hand_geometry_confusion_matrix.png
  hand_geometry_loss_curve.png
  main_results_methods.png
  method_positioning.png
  factorized_generalization.png
```

## 复现实验

### Hand Geometry 主实验

```bash
python code/run_gesture_geometry_suite.py \
  --execute \
  --skip-feature-check \
  --epochs 100 \
  --patience 4
```

训练输出的 `metrics.json` 会记录课程要求的运行时间字段：

```text
runtime.train_avg_seconds_per_sample
runtime.test_avg_seconds_per_sample
runtime.train_total_seconds
runtime.test_total_seconds
```

也可以通过测试入口直接查看：

```bash
python code/test.py \
  --model improved \
  --output-dir outputs/feature_suite/hand_geometry/main
```

### 工作流级端到端

`train.py` 用于检查缓存特征、必要时执行全量特征提取，并启动完整训练评估。该入口适合正式全量重跑：

```bash
python code/train.py \
  --model improved \
  --skip-feature-check \
  --epochs 100 \
  --patience 4 \
  --batch-size 64 \
  --gesture-feature-dir dataset/AR_Data_Process3.0/data/hand_geometry_features \
  --gesture-feature-dim 96 \
  --output-dir outputs/workflow_e2e_hand_geometry
```

如需在特征缺失时自动串起五类特征提取，去掉 `--skip-feature-check` 并添加 `--extract-features`。

### Batch 级端到端

`batch_end_to_end.py` 是独立的 batch 级闭环检查，不会触发全量训练。默认复用已有缓存特征，统计缓存加载、单 batch 训练 step 和单 batch 测试 forward 的平均样本时间：

```bash
python code/batch_end_to_end.py \
  --model improved \
  --hand-geometry \
  --batch-size 32 \
  --output-dir outputs/batch_e2e_hand_geometry
```

如果要单独评估当前 batch 从 fisheye 原视频现算 MediaPipe hand geometry 的耗时，可加：

```bash
python code/batch_end_to_end.py \
  --model improved \
  --hand-geometry \
  --raw-hand-geometry \
  --allow-raw-fallback \
  --batch-size 32 \
  --output-dir outputs/batch_e2e_hand_geometry_raw
```

### 端到端计时结果

服务器环境为 NVIDIA RTX 5880 Ada，batch size 为 64。正式工作流级端到端使用已缓存的 Hand Geometry 特征，主要统计模型训练与测试阶段耗时；batch 级 raw 结果额外统计当前 batch 从 fisheye 原视频现算 MediaPipe hand geometry 的耗时。

| 设置 | 样本数 | 平均样本训练时间 | 平均样本测试时间 | 特征相关耗时 |
|---|---:|---:|---:|---:|
| workflow cached Hand Geometry | train seen 16949 / test 744 | 0.000434 s | 0.000127 s | 使用缓存特征 |
| batch cached Hand Geometry | 32 | 0.015178 s | 0.000806 s | cache load 0.000480 s/sample |
| batch raw Hand Geometry | 32 | 0.013226 s | 0.000729 s | raw geometry 1.758790 s/sample |
| full raw workflow | train seen 10967 / test 744 | 0.000436 s | 0.000126 s | total wall time 2:01:30 |

对应完整模态测试结果：

| 模型 | joint_acc | intent_acc | scene_acc | best_epoch |
|---|---:|---:|---:|---:|
| Ours: Hand Geometry workflow E2E | 0.9946 | 0.9946 | 1.0000 | 13 |
| Ours: Hand Geometry full raw E2E | 0.9852 | 0.9866 | 0.9987 | 7 |

需要注意：workflow 计时反映“缓存特征后的训练/测试吞吐”；raw batch 计时反映 MediaPipe hand geometry 从视频现算的代价。二者协议不同，应分别汇报。

Full raw workflow 从原始数据重新执行 timestamp、MediaPipe-cropped CLIP gesture、MFCC、ASR、IMU、Hand Geometry 与训练测试。总墙钟时间为 2:01:30，其中主要耗时来自 `strong_gesture2.0.py` 约 60 分钟和 `extract_hand_geometry_features.py` 约 55 分钟。

公平比较需要区分固定缓存特征协议和重新提取原始特征协议：

| 协议 | Given Improved Baseline | Ours: Hand Geometry | 提升 |
|---|---:|---:|---:|
| cached features | 0.9839 | 0.9946 | +0.0107 |
| full raw E2E | 0.9530 | 0.9855 mean / 0.9879 best | +0.0325 / +0.0349 |

### Hand Geometry 鲁棒性实验

```bash
python code/run_missing_experiments.py \
  --model improved \
  --output-model-name hand_geometry \
  --max-missing 2 \
  --epochs 100 \
  --patience 4 \
  --gesture-feature-dir dataset/AR_Data_Process3.0/data/hand_geometry_features \
  --gesture-feature-dim 96 \
  --skip-feature-check \
  --execute

python code/run_noise_experiments.py \
  --model improved \
  --output-model-name hand_geometry \
  --epochs 100 \
  --patience 4 \
  --gesture-feature-dir dataset/AR_Data_Process3.0/data/hand_geometry_features \
  --gesture-feature-dim 96 \
  --skip-feature-check \
  --execute
```

### 汇总与可视化

```bash
python code/summarize_robustness_results.py \
  --models improved hand_geometry
```

输出：

```text
outputs/summary/robustness_summary.csv
outputs/summary/robustness_summary.md
outputs/summary/robustness_main.png
outputs/summary/robustness_missing.png
outputs/summary/robustness_noise.png
```

## 环境说明

实验主要在 GPU 服务器上运行。模型、数据集视频、特征缓存、训练权重和完整日志体积较大，默认不纳入 git。仓库中保留的 `docs/figures/` 是轻量可视化结果，用于快速查看主要实验结论。
