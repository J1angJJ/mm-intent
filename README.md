# 多模态用户交互意图理解项目说明

这个仓库是机器学习课程项目“多模态用户交互意图理解”的工作区。本文档不是正式报告，而是给队友看的主页说明：用于快速了解项目要求、当前进度、代码结构、实验结果和后续写报告时该怎么组织材料。

## 1. 项目目标

课程 PDF 的核心要求是：在给定多模态用户交互意图识别模型基础上，针对 AR 眼镜场景中的模态噪声和模态缺失问题，重构训练/测试代码，并改进模型，使其在正常、缺失、噪声条件下都能更稳定地识别用户意图。

本项目使用五类模态：

| 模态 | 来源 | 当前特征 |
|---|---|---|
| IMU | `dataset/imu.csv` | 时间窗口内 IMU 时序特征 |
| Gesture | fisheye 视频 | MediaPipe HandLandmarker 裁剪手部区域，再用 CLIP 提取视觉特征 |
| Audio | HoloLens 视频音频 | MFCC 音频特征 |
| Text | ASR 转写文本 | Whisper + SentenceTransformer 文本嵌入 |
| Scene | fisheye 视频 | ViT 场景视觉特征 |

任务标签由场景和意图组成。意图共 6 类：

```text
menu, select, magnify, narrow, brush, cancel
```

场景共 2 类：

```text
office, museum
```

主分类头是 12 类联合分类：

```text
office_menu, office_select, ..., museum_cancel
```

因此 `joint_acc` 表示场景和意图同时预测正确的比例；`intent_acc` 只看 6 类意图是否正确；`scene_acc` 只看 2 类场景是否正确。

## 2. 当前完成度

对照课程要求，目前已经完成：

| PDF 要求 | 当前状态 |
|---|---|
| A/B 训练，C 测试 | 已按给定代码的默认 train/test 视频划分完成，并额外补了按日期留出泛化检查 |
| 训练/测试代码重构成端到端入口 | 已新增 `code/train.py` 和 `code/test.py`，并完成 smoke test |
| 模态噪声 baseline | baseline 与 improved 的 15 组噪声实验均已完成 |
| 模态缺失 baseline | baseline 与 improved 的 15 组缺失实验均已完成 |
| 引入新模块或损失项改进模型 | 已完成 anchor/residual/gate/multi-task auxiliary losses 等改进 |
| 报告截图证明端到端跑通 | 已有 `logs_e2e_train_smoke.txt`、`logs_e2e_test_smoke.txt` 和 `outputs/e2e_smoke_improved_mptasks/` |
| 丰富实验结果展示 | 主结果、泛化检查、缺失实验、噪声实验、混淆矩阵、loss 曲线、模态贡献图均已生成 |

还需要队内后续完成：

- 整理正式课程报告。
- 整理不少于 15 篇参考文献。
- 明确队员分工和贡献比例。
- 做 5 分钟答辩 PPT。

## 3. 我的主要工作

本仓库当前工作主要围绕以下几件事展开。

### 3.1 环境与数据迁移

- 在本机整理项目仓库，补充 `.gitignore`、`.gitattributes`、`.editorconfig`。
- 使用 git 同步代码到服务器。
- 大体量数据集通过 `scp` 单独迁移，不进入 git。
- 本机只作为代码、文档、结果包中转和可视化查看环境；训练与特征提取主要在服务器上完成。
- 服务器环境使用 RTX 5880 Ada，适合完成完整特征提取与模型训练。

### 3.2 特征提取链路打通

已完成并验证 39 个交互视频的五类特征：

```text
timestamp
strong gesture
audio / MFCC
text / ASR
IMU
```

其中手势特征做了关键适配：

- 原 legacy `mp.solutions.hands` 在当前 MediaPipe 版本不可用。
- 已将 `strong_gesture2.0.py` 适配为优先使用 MediaPipe Tasks `HandLandmarker`。
- 模型文件为：

```text
models/hand_landmarker.task
```

新链路为：

```text
fisheye frame -> HandLandmarker -> hand crop -> CLIP vision encoder -> gesture feature
```

这比整帧 CLIP fallback 更符合“手势模态”的定义。

### 3.3 训练/测试入口重构

新增端到端入口：

```text
code/train.py
code/test.py
```

`train.py` 的作用：

- 检查五类缓存特征是否齐全。
- 在需要时可调用特征提取脚本。
- 分发到 baseline 或 improved 训练脚本。
- 支持缺失模态实验、噪声实验、随机种子、测试集视频列表等参数。

`test.py` 的作用：

- 读取指定输出目录中的 `metrics.json`。
- 打印 best checkpoint 和测试指标。
- 可附带打印分类报告。

端到端 smoke test 已完成：

```text
outputs/e2e_smoke_improved_mptasks/
logs_e2e_train_smoke.txt
logs_e2e_test_smoke.txt
```

这组只训练 3 epoch，不作为最终性能结果，只证明入口链路完整打通：

```text
feature check -> train.py -> train_and_test.py -> save outputs -> test.py -> report metrics
```

### 3.4 模型改进

Baseline 是给定思路上的统一多模态 PerceiverIO 融合模型。

Improved 模型做了这些改动：

- 将 Gesture / Text / Scene 作为强锚点模态。
- 将 IMU / Audio 作为残差辅助模态。
- 引入模态 gate，控制弱模态注入强度。
- 引入多任务辅助监督：
  - 12 类 joint classification
  - 6 类 intent auxiliary classification
  - 2 类 scene auxiliary classification
  - base intent auxiliary head
  - gesture intent auxiliary head
- 使用 label smoothing、dropout、weight decay、grad clipping 等训练稳定手段。

Improved 的总损失大致为：

```text
joint_loss
+ 0.35 * intent_loss
+ 0.10 * base_intent_loss
+ 0.15 * scene_loss
+ 0.25 * gesture_intent_loss
```

这个结构的设计目标是：让强语义/视觉模态作为稳定主干，同时让 IMU 和 Audio 以较弱、可控的方式提供补充证据，避免弱模态噪声破坏主判断。

## 4. 结果总览

### 4.1 主实验结果

![主实验结果](docs/figures/main_results_mptasks.png)

| 模型 | joint_acc | intent_acc | scene_acc |
|---|---:|---:|---:|
| Baseline + MediaPipe Tasks | 0.9516 | 0.9637 | 0.9879 |
| Improved + MediaPipe Tasks | 0.9839 | 0.9839 | 1.0000 |

结论：

- Improved 在正常测试条件下优于 baseline。
- joint_acc 从 95.16% 提升到 98.39%。
- scene_acc 达到 100%，但报告里需要保守解释为“当前数据集场景视觉线索高度可分”，不要过度声称模型具备完美场景语义理解。

### 4.2 泛化与过拟合检查

![泛化检查](docs/figures/generalization_checks.png)

默认测试集多随机种子：

| 实验 | joint_acc |
|---|---:|
| seed 7 | 0.9395 |
| seed 42 | 0.9839 |
| seed 123 | 0.9610 |

按采集日期整批留出：

| 留出日期 | joint_acc |
|---|---:|
| 20260131 | 0.9240 |
| 20260227 | 0.9669 |
| 20260301 | 0.9698 |
| 20260306 | 0.9592 |

结论：

- `98.39%` 是当前默认划分下的最佳 seed 结果，偏乐观。
- 默认测试集 3 seed 平均 joint_acc 约 96.15%。
- 按日期留出平均 joint_acc 约 95.50%。
- 模型没有出现严重“训练集记忆、测试集崩溃”的过拟合，但默认划分确实比日期留出更乐观。

### 4.3 模态缺失实验

![单模态缺失](docs/figures/single_modality_missing.png)

单模态缺失对比：

| 缺失模态 | Baseline joint_acc | Improved joint_acc |
|---|---:|---:|
| no_imu | 0.8562 | 0.9839 |
| no_audio | 0.9422 | 0.9691 |
| no_gesture | 0.9664 | 0.9906 |
| no_text | 0.4516 | 0.4933 |
| no_scene | 0.8360 | 0.9785 |

结论：

- Improved 在 5 组单模态缺失中全部优于 baseline。
- Text 是最关键的意图模态，去掉 Text 后两种模型都大幅下降。
- Gesture 和 Scene 对场景判断很关键；二者单独缺一个时还能互相补偿，同时缺失时 scene_acc 接近随机。
- IMU 和 Audio 在当前数据集上更像辅助模态，单独缺失影响较小。

几个代表性的双模态缺失结果：

| 缺失组合 | Baseline joint_acc | Improved joint_acc | 说明 |
|---|---:|---:|---|
| no_audio_text | 0.3078 | 0.3723 | 文本语义缺失后性能最低 |
| no_gesture_text | 0.5215 | 0.5672 | 文本和手势同时缺失影响明显 |
| no_gesture_scene | 0.5148 | 0.5161 | 场景视觉线索被打掉 |
| no_imu_scene | 0.7675 | 0.9785 | Improved 对该组合非常稳 |

### 4.4 噪声鲁棒性实验

![噪声鲁棒性](docs/figures/noise_robustness.png)

噪声实验结论非常清楚：Improved 在 15 组单模态噪声实验中全部优于 baseline。

代表性结果：

| 噪声设置 | Baseline joint_acc | Improved joint_acc |
|---|---:|---:|
| text_noise_20 | 0.8118 | 0.9825 |
| text_noise_40 | 0.7419 | 0.9812 |
| gesture_noise_60 | 0.8938 | 0.9879 |
| audio_noise_60 | 0.8938 | 0.9839 |
| imu_noise_40 | 0.8871 | 0.9839 |

结论：

- Improved 对单模态噪声很稳定。
- Baseline 对 Text、Gesture、Audio 噪声更敏感。
- Improved 的锚点融合、弱模态残差辅助、gate 和多任务损失能明显提升噪声鲁棒性。
- 噪声实验和模态缺失实验要分开解释：Text 完全缺失会大幅下降，但 Text 特征加噪仍较稳，说明当前噪声设置更接近特征扰动，而不是彻底删除语义。

## 5. 仓库结构

```text
code/
  baseline_real_scene.py              # baseline 训练与测试主脚本
  train_and_test.py                   # improved 训练与测试主脚本
  train.py                            # 重构后的训练入口
  test.py                             # 重构后的测试/报告入口
  run_generalization_experiments.py   # 泛化/过拟合检查实验
  run_missing_experiments.py          # 模态缺失实验
  run_noise_experiments.py            # 噪声鲁棒性实验
  collect_experiment_results.py       # 汇总 metrics.json 为 CSV
  visualize_report_summary.py         # 生成 README/报告用汇总图
  feature_extraction/                 # 五类特征提取脚本

docs/figures/
  main_results_mptasks.png
  generalization_checks.png
  single_modality_missing.png
  noise_robustness.png

outputs/
  baseline_real_scene_perceiver_io_mptasks/
  improved_real_scene_anchor2_perceiver_io_mptasks/
  generalization/
  missing_experiments/
  noise_experiments/

private/
  README_LOG.md       # 从建仓库到当前的详细操作复盘
  RESULT_SUMMARY.md   # 当前所有实验结果和报告口径总结
```

注意：

- `outputs/`、`models/`、`logs/`、`private/`、大视频和特征缓存都不进 git。
- 根 README 中的图来自 `docs/figures/`，是根据本地 CSV 重新生成的小图，可以进 git 给队友看。

## 6. 重要输出文件

主结果：

```text
outputs/experiment_summary_mptasks.csv
outputs/baseline_real_scene_perceiver_io_mptasks/
outputs/improved_real_scene_anchor2_perceiver_io_mptasks/
```

泛化检查：

```text
outputs/experiment_summary_generalization.csv
outputs/generalization/
```

端到端 smoke：

```text
outputs/experiment_summary_with_e2e.csv
outputs/e2e_smoke_improved_mptasks/
logs_e2e_train_smoke.txt
logs_e2e_test_smoke.txt
```

模态缺失：

```text
outputs/experiment_summary_missing_baseline.csv
outputs/experiment_summary_missing_improved.csv
outputs/missing_experiments/baseline/
outputs/missing_experiments/improved/
```

噪声鲁棒性：

```text
outputs/experiment_summary_noise_baseline.csv
outputs/experiment_summary_noise_improved.csv
outputs/noise_experiments/baseline/
outputs/noise_experiments/improved/
```

更详细的复盘和报告素材：

```text
private/README_LOG.md
private/RESULT_SUMMARY.md
```

## 7. 常用命令

### 7.1 端到端训练入口

```bash
python code/train.py \
  --model improved \
  --epochs 100 \
  --patience 4 \
  --output-dir outputs/improved_real_scene_anchor2_perceiver_io_mptasks
```

### 7.2 测试/读取结果入口

```bash
python code/test.py \
  --model improved \
  --output-dir outputs/improved_real_scene_anchor2_perceiver_io_mptasks \
  --show-reports
```

### 7.3 模态缺失实验

```bash
python code/run_missing_experiments.py \
  --model improved \
  --epochs 100 \
  --patience 4 \
  --execute
```

### 7.4 噪声实验

```bash
python code/run_noise_experiments.py \
  --model improved \
  --epochs 100 \
  --patience 4 \
  --execute
```

### 7.5 汇总结果

```bash
python code/collect_experiment_results.py \
  --root outputs \
  --out outputs/experiment_summary_all.csv
```

### 7.6 重新生成 README 图

```bash
python code/visualize_report_summary.py
```

## 8. 报告建议结构

正式报告可以按下面结构写：

1. 项目背景与任务定义
2. 数据集与五类模态说明
3. 特征提取与对齐方法
4. Baseline 模型结构
5. Improved 模型结构与创新点
6. 端到端重构说明与运行截图
7. 正常条件 baseline vs improved
8. 泛化与过拟合检查
9. 模态缺失实验
10. 噪声鲁棒性实验
11. 模态贡献与可解释性分析
12. 局限性与后续工作
13. 队员分工与贡献比例
14. 参考文献

报告里最值得突出的创新点：

- MediaPipe Tasks + CLIP 的手势特征适配。
- Gesture/Text/Scene anchor fusion。
- IMU/Audio residual support。
- Modality gate。
- 多任务辅助损失。
- 完整的缺失/噪声/泛化实验矩阵。

## 9. 结论草稿

当前实验支持这样的总体结论：

> 本项目完成了从多模态数据预处理、特征提取、端到端训练测试入口重构，到模型改进和系统实验评估的完整流程。相比 baseline，改进模型在正常测试集上将 joint accuracy 从 95.16% 提升至 98.39%；在 5 组单模态缺失实验中全部优于 baseline；在 15 组单模态噪声实验中全部优于 baseline。实验表明，文本语义是意图识别的核心模态，Gesture 与 Scene 是场景判断的重要视觉线索，IMU 和 Audio 在当前数据集中主要提供辅助信息。改进模型通过锚点主导融合、弱模态残差辅助、模态 gate 和多任务辅助损失，提高了多模态意图识别在模态缺失和噪声扰动条件下的稳定性。

这段不能直接原封不动当最终报告，但可以作为报告摘要和答辩讲稿的基础。
