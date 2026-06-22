# 智能眼镜多模态交互意图识别代码说明

## 目录结构

### 1. 模型训练和测试代码重构
- `train_and_test.py`
  - 基于锚点主干与残差辅助的多模态用户交互意图模型代码
  - 将这个脚本拆分成train.py和test.py，将特征提取代码整合进train.py和test.py代码，调试，实现端到端输入输出；重构代码后，运行train.py脚本进行模型训练，运行test.py脚本进行测试。
- `train.py`
  - `--input-mode raw` 从 HoloLens 视频、fisheye 视频和 `imu.csv` 开始，运行预处理后训练模型；中间特征只作为可追踪缓存。
  - `--input-mode features` 保留原有缓存特征训练方式。
  - 可通过 `--missing-modalities` 和 `--noise-modality --noise-level` 启动模态缺失/噪声实验。
- `test.py`
  - 独立测试入口：加载 checkpoint、`scalers.pkl` 和 `label_encoder.pkl`，对测试数据执行真实前向推理。
  - 保存 `independent_test_metrics.json`、逐样本预测和独立分类报告，不再读取旧 `metrics.json` 作为测试结果。
- `run_missing_experiments.py`
  - 生成或执行单模态缺失、双模态缺失实验命令。
- `run_noise_experiments.py`
  - 默认生成或执行原始模态 20%、40%、60% 噪声实验命令，结果写入 `outputs/raw_noise_experiments/`。
- `collect_experiment_results.py`
  - 汇总多个实验输出目录中的 `metrics.json`，生成 CSV 指标表。

原始数据端到端训练与独立测试：

```powershell
.\.venv\Scripts\python.exe code\train.py `
  --model baseline `
  --input-mode raw `
  --epochs 100 `
  --patience 10 `
  --output-dir outputs\baseline_raw_end_to_end

.\.venv\Scripts\python.exe code\test.py `
  --model baseline `
  --input-mode raw `
  --raw-cache-dir outputs\raw_feature_cache\clean_seed42 `
  --output-dir outputs\baseline_raw_end_to_end
```

原始模态噪声的 15 组实验：

```powershell
# 先检查命令，不训练
.\.venv\Scripts\python.exe code\run_noise_experiments.py `
  --model baseline `
  --epochs 100 `
  --patience 10

# 正式执行
.\.venv\Scripts\python.exe code\run_noise_experiments.py `
  --model baseline `
  --epochs 100 `
  --patience 10 `
  --execute
```

原始噪声定义固定记录在每个缓存的 `raw_preprocessing_manifest.json`：Audio 在 MFCC 前对波形加噪；Gesture/Scene 在视觉编码前对像素加噪；IMU 在派生运动学量前对原始通道加噪；Text 在语义编码前按比例删除转写字符。未受扰动的模态复用 `data_full` 中由原始数据确定性生成的缓存，目标模态必须重新提取。

Hand Geometry 使用 `--gesture-representation hand_geometry`。Gesture 噪声会在 MediaPipe landmark 检测前加入原始像素噪声，输出缓存与 CLIP Gesture 缓存分离。

为减少重复视频解码，`precompute_gesture_noise_levels.py` 在同一次原始帧读取中生成 20%/40%/60% 三档 Gesture 噪声，并按视频跳过已完成缓存。`clone_hand_geometry_noise_caches.py` 让 Hand Geometry 与其他模型共享同一份非 Gesture 原始扰动，只替换手势表示。`run_noise_experiments.py --skip-existing` 可跳过已有 `metrics.json`，因此中断后无需从头训练。

模型训练完成会保存成以下文件：
- `*.pt`：训练完成后的模型权重文件
- `scalers.pkl`：输入特征标准化器
- `label_encoder.pkl`：类别标签编码器

### 2. 特征提取代码，位于feature_extraction文件夹
- `get_timestamp.py`：提取时间戳
- `strong_gesture2.0.py`：提取手势特征
  - 优先使用 MediaPipe Tasks `HandLandmarker` 新接口，模型路径默认为 `models/hand_landmarker.task`，也可通过 `MM_INTENT_HAND_LANDMARKER_TASK` 指定。
  - 若未提供 `.task` 模型，会尝试 legacy `mp.solutions.hands`；两者都不可用时退化为整帧 CLIP 特征。
- `mfcc.py`：提取音频特征
- `ASR.py`：语音转文本特征
- `imu.py`：提取IMU特征

### 3. 数据集
- 位于dataset文件夹，包含3个用户在office场景和museum场景的6种意图交互数据。
- fisheye文件夹为鱼眼摄像头获取的视频，用作场景模态和手势模态数据。
- HoloLens文件夹为头显设备获取的视频，用作音频模态数据。
- imu.csv文件为IMU数据，用作IMU模态数据。根据交互视频的时间获取相应的IMU数据。
- AR_Data_Process3.0文件夹为各模态特征提取脚本。在重构代码时，在你新建的train.py和test.py中调用特征提取脚本。
- 训练集：
   # ================ office ===============
     "interaction_20260131_120024.mp4",    # Luo
     "interaction_20260227_132951.mp4",    
     "interaction_20260227_133408.mp4",     
     "interaction_20260131_114156.mp4",    
     "interaction_20260131_115150.mp4",    
     "interaction_20260131_114852.mp4"
    
     "interaction_20260301_073041.mp4",    # Gu
     "interaction_20260301_064753.mp4",    
     "interaction_20260306_072721.mp4",    
     "interaction_20260301_071948.mp4",    
     "interaction_20260131_121548.mp4",
     "interaction_20260301_073435.mp4",    
     "interaction_20260301_072503.mp4"

    # ============== museum ===============
    "interaction_20260131_071552.mp4",      # Luo
    "interaction_20260131_072412.mp4",
    "interaction_20260131_084300.mp4",
    "interaction_20260131_084732.mp4",
    "interaction_20260131_085207.mp4",
    "interaction_20260131_085611.mp4",
    "interaction_20260131_090139.mp4",

    "interaction_20260131_065459.mp4",       # Gu
    "interaction_20260131_070722.mp4",
    "interaction_20260131_090541.mp4",
    "interaction_20260131_090917.mp4",
    "interaction_20260131_091249.mp4",
    "interaction_20260131_091657.mp4",

- 测试集：
    # ============== office ================
    "interaction_20260306_072344.mp4",    # Bian
    "interaction_20260227_122606.mp4",    
    "interaction_20260227_122952.mp4",    
    "interaction_20260227_123354.mp4",    
    "interaction_20260227_124559.mp4",    
    "interaction_20260227_123745.mp4"

    # ============= museum ==============
    "interaction_20260306_082346.mp4"   # Bian
    "interaction_20260306_083107.mp4"
    "interaction_20260306_083434.mp4"
    "interaction_20260306_084406.mp4"
    "interaction_20260306_084853.mp4"
    "interaction_20260306_085830.mp4"
    "interaction_20260306_090441.mp4"

### 4. 课程项目代码提交
- 提交小组完整代码压缩文件，包括已训练模型文件。在报告中汇报模型训练过程和测试过程、模型效果、代码结构、代码运行方式等。汇报总训练时间、平均样本训练时间、平均样本测试等。

### 5. 泛化与过拟合检查

在模态缺失和噪声实验前，建议先运行泛化检查，确认主结果不是由单一随机种子或固定测试划分偶然得到。

新增脚本：

```bash
python code/run_generalization_experiments.py
```

默认 dry-run，不会真正训练。它会生成两类实验：

- 默认测试集多随机种子重复：`seed=7,42,123`。
- 按采集日期整批留出测试：`20260131`、`20260227`、`20260301`、`20260306`。

服务器正式执行：

```bash
python code/run_generalization_experiments.py \
  --model improved \
  --epochs 100 \
  --patience 4 \
  --execute 2>&1 | tee logs_generalization_improved.txt
```

断点续跑：

```bash
python code/run_generalization_experiments.py \
  --model improved \
  --epochs 100 \
  --patience 4 \
  --skip-existing \
  --execute 2>&1 | tee -a logs_generalization_improved.txt
```

只跑多随机种子：

```bash
python code/run_generalization_experiments.py --seed-only --execute
```

只跑按日期留出：

```bash
python code/run_generalization_experiments.py --date-only --execute
```

汇总：

```bash
python code/collect_experiment_results.py \
  --root outputs \
  --out outputs/experiment_summary_generalization.csv
```

相关环境变量：

- `SMART_AR_RANDOM_SEED`：控制随机种子。
- `SMART_AR_VAL_SPLIT`：控制训练集内部验证比例，默认 `0.2`。
- `SMART_AR_TEST_VIDEO_NAMES`：用逗号分隔指定测试视频列表。
