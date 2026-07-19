![Logo](logo.png)
# CMK-AGN：三模态临床康复评估

本项目由珠海复旦创新研究院的医学人工智能科技创新中心研发团队开发，CMK-AGN 是一个三模态深度学习模型，将一次康复试次中同步采集的
**EEG · EMG · IMU** 信号映射为四项相互独立的临床评分。各模态先经
加权贝叶斯 DTW（EMG↔IMU）加线性重采样的 EEG 进行对齐，再由 CMK-AGN
主干网络融合；每个任务训练一个独立的输出头（不共享输出头，也不使用联合损失）。

> **命名说明。** 模型对外名称为 **CMK-AGN**。出于向后兼容，Python 模块内部
> 仍沿用旧标识符 `adk_mdfan`（例如 `src/models/adk_mdfan_tri.py`、类
> `ADKMDFANTriBackbone`）。二者指的是同一套 CMK-AGN 架构。

## 临床任务

每个任务作为独立模型训练。

| 任务键          | 临床量表                        | 类型        | 取值范围 / 类别            |
|-----------------|---------------------------------|-------------|----------------------------|
| `FMA_UE`        | FMA-UE 手部子分                 | 回归        | 0 – 20（整数）             |
| `BI`            | Barthel 指数                    | 回归        | 0 – 100（步长 5）          |
| `hand_tone`     | 手部 MAS（改良 Ashworth）       | 6 类有序    | `0, 1, 1+, 2, 3, 4`        |
| `hand_function` | Brunnstrom 手部分期             | **5 类**有序 | `2, 3, 4, 5, 6`           |

代码层的键名 `hand_tone` / `hand_function` 是为兼容 manifest 而保留的；
临床上分别对应手部 MAS 与 Brunnstrom 手部分期。

### 为什么 `hand_function` 是 5 分类任务（Brunnstrom 2–6）

数据集以合成数据为主（见下文）。使用固定随机种子 **2024** 复现采样流程后，
95 份合成样本中**恰好只有 1 份**——**S83**——被采样到 Brunnstrom **1 期**。
S83 的其他指标为 FMA-UE = 5（非零，即存在可测量的自主手部运动）、手部 MAS = 0，
这与 1 期所定义的“完全弛缓性瘫痪”不一致；且 1 期与 2 期之间在临床上并无严格
划分，因此将 S83 重新归入 **2 期**。这样一来数据集中不再存在 1 期样本，
`hand_function` 即成为覆盖 2–6 期的干净 **5 分类**任务。

该重分配在 `simulate_data.py`（`make_new_subjects()`）中以确定性方式执行，
因此从零重新生成的数据与随附的
`samples_manifest_tri_4tasks_100subjects.csv` 一致；同时它也体现在
`src/task_config.py` 中（`hand_function` 类别 `= (2, 3, 4, 5, 6)`）。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 获取 / 生成数据

原始信号**未**包含在本仓库中（见下文 *数据* 一节）。将 5 位真实患者的记录
放到 `BJH/` 下之后，即可重新生成完整的 100 例合成数据集、manifest 与标签：

```bash
python simulate_data.py
python src/patient_splits.py \
    --manifest samples_manifest_tri_4tasks_100subjects.csv \
    --output splits/3fold_patient_split_tri_4tasks_100subjects.json --force
```

仓库中随附的 manifest 与 3 折划分正是由上述命令（种子 2024）生成的；
若你磁盘上已有对应的 `BJH/` 记录，也可跳过重新生成这一步。

### 3. 训练 CMK-AGN（单任务）

```bash
# 只训练某一折：
python src/train.py --task FMA_UE --fold 1

# 训练全部 3 折（fold 0 表示“所有折”）：
python src/train.py --task hand_function --fold 0

# 自定义输出目录：
python src/train.py --task BI --fold 1 --out-dir experiments/BI/run1
```

输出默认写入 `RESULT_newdata/<task>/baseline/`（可用 `--out-dir` 覆盖；
随附的 `RESULT_newdata_*` 目录即通过显式 `--out-dir` 生成），其中包含
检查点 `<task>_fold<n>.pth`、每折的 `<task>_fold<n>_logs/`
（`training_history.csv`、`val_predictions.csv`、`metrics.json`），
以及跨折汇总 `<task>_3fold_summary.{csv,json}` 与 `config.json`。

常用参数：`--manifest`、`--split-json`、`--modalities`（例如 `emg+imu`
用于模态消融）、`--epochs`（默认 120）、`--lr`（默认 1e-4）、`--device`。

### 4. 推理

```bash
python src/predict.py --task FMA_UE \
    --checkpoint "RESULT_newdata_CMK-AGN(Ours)/FMA_UE/baseline/FMA_UE_fold1.pth" \
    --manifest samples_manifest_tri_4tasks_100subjects.csv
# 或一次性预测全部四个任务：
python src/predict.py --all-tasks
```

### 5. 基线与消融实验

```bash
# 列出可用基线（cnn1d、eegnet、bilstm_attn、mlp、xgboost、svm_rbf 等）：
PYTHONPATH=src python -m baselines.train_baseline --list

# 训练某个基线：
PYTHONPATH=src python -m baselines.train_baseline --model cnn1d --task FMA_UE --fold 0

# 完整的模态 + 模块消融扫描（全部任务 × 3 折）：
bash scripts/ablation_sweep.sh
```

---

## 目录结构

```
CMK-AGN/
├── simulate_data.py                          # 合成数据生成器（种子 2024）
├── samples_manifest_tri_4tasks_100subjects.csv
├── samples_manifest_tri_4tasks_15subjects.csv
├── bjh_labels.json                           # 每位患者的临床标签（索引器输入）
├── splits/                                   # 患者互斥的 3 折划分
├── src/
│   ├── train.py                              # CMK-AGN 单任务训练器
│   ├── predict.py                            # 推理
│   ├── task_config.py                        # 任务规格（取值范围、类别、编码器）
│   ├── clinical_model.py                     # 统一模型封装（四个任务）
│   ├── data_indexer_tri_modified.py          # 三模态 manifest 索引器
│   ├── patient_splits.py                     # K 折患者划分
│   ├── subject_aggregation.py                # 试次袋 → 受试者级聚合
│   ├── aggregate_ablation.py                 # 跨折 / 跨消融汇总
│   ├── alignment/                            # 加权贝叶斯 DTW + 三模态对齐
│   ├── bjh_io/                               # EEG(.bdf) / EMG(.csv) 加载器 + 缓存
│   ├── models/                               # CMK-AGN 主干（adk_mdfan_tri.py）
│   └── baselines/                            # 深度学习 + 机器学习基线对比框架
├── scripts/ablation_sweep.sh                 # 消融实验驱动脚本
├── RESULT_newdata_CMK-AGN(Ours)/             # CMK-AGN 检查点 + 日志/指标
├── RESULT_newdata_baseline/                  # CMK-AGN 与基线对比（日志/指标）
├── RESULT_newdata_ablation/                  # 模态 + 模块消融运行结果
└── requirements.txt
```

---

## 数据（用于复现）

原始记录未纳入版本库。要端到端复现，协作者需要 **5 位真实锚点患者（S1–S5）**——
`simulate_data.py` 会从磁盘扫描这些文件，并据此合成其余 95 位受试者：

- `BJH/EEG_new/S1_*.bdf … S5_*.bdf` — 真实 EEG 记录（BDF 格式）
- `BJH/EMG_new/S{1..5}_*.csv` — 对应的 EMG CSV（每个文件同时内嵌 IMU 列）

准备好这些文件后，执行 **快速开始 → 2** 中的两条命令。其余内容
（95 位合成受试者、manifest、标签、划分）都会在种子 2024 下确定性地重新生成，
随后即可训练 CMK-AGN。
