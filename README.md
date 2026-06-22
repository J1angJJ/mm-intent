# 多模态 AR 交互意图理解（WJY 本地复现）

本文件沿用原 `README.md` 的结构，但所有数值、图表和结论均来自当前 Windows 本地环境的全量复现实验，不沿用原 README 中未能在本机精确复现的结果。任务输入包含 IMU、手势视频、音频、ASR 文本和场景视觉五类模态，输出为 6 类意图与 2 类场景组合而成的 12 类联合标签。

当前全量数据包含 39 个视频、1991 个对齐事件样本；训练用户 A+B，测试用户 C，主测试集 `N=744`。主结果、模态缺失、模态噪声、三随机种子、四日期留出以及 Factorized Heads 评估均已完成。

## 任务定义

意图类别：

| ID | Intent |
|---:|---|
| 0 | menu |
| 1 | select |
| 2 | magnify |
| 3 | narrow |
| 4 | brush |
| 5 | cancel |

场景类别：

| ID | Scene |
|---:|---|
| 0 | office |
| 1 | museum |

联合标签形式为：

```text
{scene}_{intent}
```

评价指标：

| 指标 | 含义 |
|---|---|
| `joint_acc` | 场景和意图同时预测正确 |
| `intent_acc` | 只评价 6 类意图预测 |
| `scene_acc` | 只评价 2 类场景预测 |

## 数据与模态

| 模态 | 数据来源 | 当前特征表示 |
|---|---|---|
| IMU | HoloLens/同步传感器数据 | 10 帧时序 IMU 特征 |
| Gesture | fisheye 视频 | MediaPipe 手部裁剪 + CLIP，或 hand landmarks 几何特征 |
| Audio | HoloLens 视频音频 | MFCC |
| Text | Whisper ASR 转写 | SentenceTransformer，10 帧复制对齐 |
| Scene | fisheye 视频 | ViT 场景视觉特征 |

特征目录为 `dataset/AR_Data_Process3.0/data_full/`。39 个视频的 Gesture、Audio、Text、IMU 和 Hand Geometry 特征均包含 1991 个对齐样本。Text 特征全部非零，1977/1991 个事件获得非空转写。Hand Geometry 的逐帧手部检测率为约 `30.1%`，约 `78.9%` 的事件至少有一帧检测到手；该覆盖率是解释几何分支性能时必须保留的限制。

## 方法概览

### Given Baseline

Given Baseline 对五类模态分别编码，再通过多模态 Perceiver-style 融合网络直接预测 12 类联合标签。当前实现对随机初始化高度敏感：seed 42、7、123 的测试 joint accuracy 分别为 `0.8616`、`0.9731`、`0.9758`。标准输出采用验证准确率优先、验证损失次优的 seed 123 checkpoint，而不是按测试集准确率挑选。

### Given Improved Baseline

Given Improved Baseline 使用 Gesture / Text / Scene 作为主锚点，并通过残差路径注入 IMU / Audio，同时加入 intent、scene 和 gesture-intent 辅助监督。当前环境下该模型在默认 seed 42 上达到 `0.9825` joint accuracy。

### Ours: Hand Geometry

Hand Geometry 将 MediaPipe 21 个 hand landmarks 直接编码为 `(10, 96)` 几何时序，包括：

- 手部存在标记、中心点和 bbox 尺寸。
- 21 个 landmarks 的相对坐标。
- 指尖到手腕距离。
- 拇指与其他指尖的 pinch 距离。
- 掌宽、手高和尺度归一化几何量。

该表示主要提供动作轨迹和手形结构。当前主测试集 joint accuracy 为 `0.9839`，仅比 Improved 高 1 个正确样本，因此主任务优势不能表述为稳定或显著提升；其主要证据来自 Text 缺失条件。

### Ours: Factorized Heads

Factorized Heads 使用 Hand Geometry checkpoint 的 intent logits 与 Given Improved Baseline 的 scene logits：

```text
Hand Geometry checkpoint  -> intent logits
Given Improved Baseline   -> scene logits
joint logits = intent_weight * intent logits
             + scene_weight  * scene logits
```

当前 evaluator 在候选网格中选择 `intent_weight=0.75`、`scene_weight=0.5`，完整模态 joint accuracy 为 `0.9906`。由于当前脚本直接在测试集上选择最优权重，该结果属于探索性上界，不应作为完全独立、无偏的最终测试估计；正式论文应在验证集选权重后固定测试。

## 实验结果

### 主结果

![主结果对比](docs_wjy/figures/main_results_methods.png)

| 模型 / 特征 | joint_acc | intent_acc | scene_acc |
|---|---:|---:|---:|
| Given Baseline (seed 123) | 0.9758 | 0.9758 | **1.0000** |
| Given Improved Baseline (seed 42) | 0.9825 | 0.9839 | 0.9987 |
| Ours: Hand Geometry (seed 42) | 0.9839 | **0.9919** | 0.9919 |
| Ours: Factorized Heads | **0.9906** | **0.9919** | 0.9987 |

在 `N=744` 的主测试集上，Improved、Hand Geometry 和 Factorized Heads 分别预测正确 731、732 和 737 个样本。Hand Geometry 相对 Improved 只增加 1 个正确样本，同时 scene accuracy 下降，因此主结果本身不足以建立明确的几何特征优势。Factorized Heads 恢复了 scene head，并将 joint accuracy 提高到 `0.9906`，但存在前述测试集权重选择偏差。

![方法定位](docs_wjy/figures/method_positioning.png)

![Hand Geometry 混淆矩阵](docs_wjy/figures/hand_geometry_confusion_matrix.png)

![Hand Geometry 训练曲线](docs_wjy/figures/hand_geometry_loss_curve.png)

### 模态缺失鲁棒性

![模态缺失鲁棒性](docs_wjy/figures/hand_geometry_robustness_missing.png)

以下模型均在对应缺失条件下重新训练，不是对干净 checkpoint 的直接遮蔽测试：

| 缺失设置 | Given Improved Baseline | Ours: Hand Geometry | 变化 |
|---|---:|---:|---:|
| no_text | 0.4435 | **0.7742** | +0.3306 |
| no_audio_text | 0.4005 | **0.7285** | +0.3280 |
| no_imu_text | 0.4261 | **0.7688** | +0.3427 |
| no_gesture_text | **0.6815** | 0.6250 | -0.0565 |
| no_scene | **0.9516** | 0.8024 | -0.1492 |

结果支持一个有限而明确的结论：当 Text 缺失且几何手势仍可用时，Hand Geometry 相对 Improved 提升约 `0.33–0.34`；但当 Gesture 与 Text 同时缺失时，Hand Geometry 反而更差。它也不能替代显式 Scene，`no_scene` 下 intent accuracy 为 `0.9745`、scene accuracy 为 `0.8266`，最终 joint accuracy 降至 `0.8024`。

### 噪声鲁棒性

![噪声鲁棒性](docs_wjy/figures/hand_geometry_robustness_noise.png)

> 协议说明：下表是此前在已提取特征上加噪后训练得到的历史结果，不能替代课程 PDF 要求的“原始模态数据加噪”实验。仓库现已提供 raw-data 噪声管线；重新运行后的合规结果应从 `outputs/raw_noise_experiments/` 汇总，并在完成前与本表分开报告。

以下模型同样在对应噪声条件下重新训练：

| 噪声设置 | Given Improved Baseline | Ours: Hand Geometry | 变化 |
|---|---:|---:|---:|
| gesture_noise_60 | **0.9825** | 0.9798 | -0.0027 |
| text_noise_60 | **0.9852** | 0.9731 | -0.0121 |
| audio_noise_60 | 0.9812 | **0.9852** | +0.0040 |
| imu_noise_60 | **0.9839** | **0.9839** | 0.0000 |
| scene_noise_60 | 0.9798 | **0.9825** | +0.0027 |

当前结果不支持“Hand Geometry 在大多数噪声设置下更优”的宽泛表述。它在 Audio 和 Scene 60% 噪声下略优，在 IMU 噪声下持平，但在 Gesture 和 Text 60% 噪声下略差。所有差异均较小，且当前未进行多种子置信区间或配对显著性检验。

### 泛化分析

![泛化分析](docs_wjy/figures/factorized_generalization.png)

| 泛化设置 | Given Improved Baseline | Ours: Hand Geometry | Ours: Factorized Heads | Factorized Heads (No Explicit Scene) |
|---|---:|---:|---:|---:|
| 3-seed mean | 0.9691 | 0.9673 | **0.9758** | 0.9642 |
| 4-date holdout mean | 0.9549 | 0.9588 | **0.9735** | 0.9674 |

三随机种子结果表明，Hand Geometry 的均值没有超过 Improved；这说明默认 seed 42 的 1 个样本优势不能外推为稳定主效应。按日期留出时 Hand Geometry 均值略高于 Improved，但 Factorized Heads 的完整模态均值最高。最困难的 `2026-01-31` 留出中，Improved、Hand Geometry、Factorized Heads 的 joint accuracy 分别为 `0.9013`、`0.9173`、`0.9213`。

默认 split 的 `no_scene` 测试时，Factorized Heads joint accuracy 为 `0.9530`。这里仅移除显式 ViT Scene；MediaPipe 裁剪后的 CLIP Gesture 仍可能包含背景信息，因此不能解释为完全无场景视觉输入。此外，Factorized 鲁棒性使用干净 checkpoint 做测试时扰动，与前述“按扰动重新训练”的鲁棒性表不是同一协议，不能直接横向比较。

## 尝试过程

当前本地环境完整重跑了 Baseline、Improved、Hand Geometry、Factorized Heads、鲁棒性和泛化实验。仓库中保留了其他探索方法的代码，但本次没有重新执行其负结果，因此不沿用原 README 的具体负结果数值。

| 尝试 | 当前环境结论 |
|---|---|
| Hierarchical Margin / Focal / Missing-Modality Distillation | 代码存在，本次未重跑 |
| Supervised Contrastive / Prototype / Ensemble | 代码存在，本次未重跑 |
| ASR 文本增强 | 代码存在，本次未重跑 |
| CLIP Gesture + Hand Geometry 早期拼接 | 代码存在，本次未重跑 |
| Ours: Hand Geometry | 主任务 0.9839；Text 缺失条件优势明确，但三种子均值不优于 Improved |
| Ours: Factorized Heads | 主任务 0.9906；当前评估受测试集权重选择影响 |

## 代码结构

```text
code/
  train.py
  baseline_real_scene.py
  train_and_test.py
  run_missing_experiments.py
  run_noise_experiments.py
  run_generalization_experiments.py
  run_factorized_full_suite.py
  evaluate_factorized_head_fusion.py
  visualize_wjy_results.py
  feature_extraction/
    get_timestamp.py
    strong_gesture2.0.py
    extract_hand_geometry_features.py
    ASR.py
    mfcc.py
    imu.py

docs_wjy/figures/
  factorized_generalization.png
  generalization_checks.png
  hand_geometry_confusion_matrix.png
  hand_geometry_loss_curve.png
  hand_geometry_robustness_main.png
  hand_geometry_robustness_missing.png
  hand_geometry_robustness_noise.png
  main_results_hand_geometry.png
  main_results_methods.png
  main_results_mptasks.png
  method_positioning.png
  noise_robustness.png
  single_modality_missing.png
```

## 复现实验

### 当前环境配置

在 PowerShell 中先设置：

```powershell
$full = "$PWD\dataset\AR_Data_Process3.0\data_full"
$env:MM_INTENT_PROCESSED_DATA_DIR = $full
$env:MM_INTENT_OUTPUT_DIR = "$PWD\outputs"
$env:HF_HOME = "$HOME\.cache\huggingface"
```

### 主实验

Baseline 使用经验证集规则选定的 seed 123：

```powershell
.\.venv\Scripts\python.exe code\train.py `
  --model baseline `
  --seed 123 `
  --epochs 100 `
  --patience 10 `
  --output-dir outputs\baseline_real_scene_perceiver_io_mptasks
```

Improved 与 Hand Geometry：

```powershell
.\.venv\Scripts\python.exe code\train.py `
  --model improved `
  --epochs 100 `
  --patience 4 `
  --output-dir outputs\improved_real_scene_anchor2_perceiver_io_mptasks

.\.venv\Scripts\python.exe code\run_gesture_geometry_suite.py `
  --execute `
  --skip-generate `
  --gesture-output-dir "$full\hand_geometry_features" `
  --epochs 100 `
  --patience 4
```

### 鲁棒性实验

Improved、Hand Geometry 和 Baseline 均使用 `run_missing_experiments.py` 与 `run_noise_experiments.py`。缺失实验包含 5 个单模态和 10 个双模态组合；噪声实验包含五种模态的 20%/40%/60% 噪声。

课程要求的 raw missing 全量训练与独立测试可直接运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_raw_missing_reruns_utf8.ps1
```

该流程为每个组合建立可审计的轻量条件缓存，只硬链接未缺失模态，缺失模态不执行提取；训练阶段跳过重复的内置测试，随后由 `code/test.py` 对 checkpoint 独立推理。`--skip-existing` 支持中断后续跑。结果写入 `outputs/raw_missing_experiments/`。

### Factorized Heads 与泛化实验

```powershell
.\.venv\Scripts\python.exe code\run_factorized_full_suite.py `
  --epochs 100 `
  --patience 4 `
  --seeds 7 42 123 `
  --geometry-feature-dir "$full\hand_geometry_features" `
  --scene-gesture-feature-dir "$full\strong_gesture_features" `
  --execute
```

### 当前图表生成

```powershell
.\.venv\Scripts\python.exe code\visualize_wjy_results.py
```

输出：

```text
docs_wjy/figures/*.png
outputs/main_results_summary.csv
outputs/factorized_full_summary.csv
```

## 环境说明

本次复现在 Windows、NVIDIA GeForce RTX 4060 Laptop GPU、CUDA 12.8、PyTorch 2.11.0、MediaPipe 0.10.35 环境完成。所有结果均来自 `dataset/AR_Data_Process3.0/data_full/` 与当前 `outputs/`，而不是原 README 的历史图表或未纳入 Git 的服务器 checkpoint。

本地原始视频、全量特征、Scene 缓存、训练权重和日志体积较大，默认不纳入 Git。`docs_wjy/figures/` 是当前环境可复核的轻量结果图；复现结论应以该目录和本文件为准。
