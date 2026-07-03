# CMK-AGN: Tri-Modal Clinical Rehabilitation Assessment

CMK-AGN is a tri-modal deep-learning model that maps synchronized
**EEG · EMG · IMU** recordings of a rehabilitation trial to four independent
clinical scores. Modalities are aligned with a Weighted-Bayes DTW scheme
(EMG↔IMU) plus linearly-resampled EEG, then fused by the CMK-AGN backbone,
which trains one independent head per task (no shared head, no joint loss).

> **Note on naming.** The public model name is **CMK-AGN**. For backward
> compatibility, the Python modules still use the legacy identifier
> `adk_mdfan` (e.g. `src/models/adk_mdfan_tri.py`, class
> `ADKMDFANTriBackbone`). These refer to the same CMK-AGN architecture.

## Clinical tasks

Each task is trained as a separate model.

| Task key        | Clinical scale                | Type            | Range / Classes         |
|-----------------|-------------------------------|-----------------|-------------------------|
| `FMA_UE`        | FMA-UE hand subscore          | Regression      | 0 – 20 (integer)        |
| `BI`            | Barthel Index                 | Regression      | 0 – 100 (step 5)        |
| `hand_tone`     | Hand MAS (Modified Ashworth)  | 6-class ordinal | `0, 1, 1+, 2, 3, 4`     |
| `hand_function` | Brunnstrom hand stage         | **5-class** ordinal | `2, 3, 4, 5, 6`     |

The code-level keys `hand_tone` / `hand_function` are kept for manifest
compatibility; clinically they are Hand MAS and Brunnstrom hand-stage.

### Why `hand_function` is a 5-class task (Brunnstrom 2–6)

The dataset is largely synthetic (see below). When the sampling pipeline is
reproduced with the fixed seed **2024**, exactly **one** of the 95 synthetic
subjects — **S83** — is drawn to Brunnstrom **stage 1**. S83's other
indicators are FMA-UE = 5 (a non-zero, i.e. measurable, voluntary hand
movement) and hand MAS = 0. These are inconsistent with the complete flaccid
paralysis that defines stage 1, and the stage-1 / stage-2 boundary is not
sharply delimited clinically, so S83 is reassigned to **stage 2**. This leaves
no stage-1 subjects, making `hand_function` a clean **5-class** task over
stages 2–6.

This reassignment is applied deterministically in `simulate_data.py`
(`make_new_subjects()`), so a from-scratch regeneration matches the shipped
`samples_manifest_tri_4tasks_100subjects.csv`, and it is reflected in
`src/task_config.py` (`hand_function` classes `= (2, 3, 4, 5, 6)`).

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Get / generate the data

The raw signals are **not** included in this repository (see *Data* below).
With the 5 real-patient recordings placed under `BJH/`, regenerate the full
100-subject synthetic dataset, manifest and labels:

```bash
python simulate_data.py
python src/patient_splits.py \
    --manifest samples_manifest_tri_4tasks_100subjects.csv \
    --output splits/3fold_patient_split_tri_4tasks_100subjects.json --force
```

The manifest and 3-fold split shipped in this repo were produced by exactly
these commands (seed 2024), so you can also skip regeneration if you already
have the matching `BJH/` recordings on disk.

### 3. Train CMK-AGN on one task

```bash
# One fold:
python src/train.py --task FMA_UE --fold 1

# All 3 folds (fold 0 == "all folds"):
python src/train.py --task hand_function --fold 0

# Custom output location:
python src/train.py --task BI --fold 1 --out-dir experiments/BI/run1
```

Outputs default to `RESULT_newdata/<task>/baseline/` (override with `--out-dir`;
the shipped `RESULT_newdata_*` folders were produced with explicit `--out-dir`)
and contain the checkpoint `<task>_fold<n>.pth`, per-fold `<task>_fold<n>_logs/`
(`training_history.csv`, `val_predictions.csv`, `metrics.json`), and the
cross-fold summary `<task>_3fold_summary.{csv,json}` + `config.json`.

Key arguments: `--manifest`, `--split-json`, `--modalities` (e.g. `emg+imu`
for modality ablation), `--epochs` (default 120), `--lr` (default 1e-4),
`--device`.

### 4. Inference

```bash
python src/predict.py --task FMA_UE \
    --checkpoint "RESULT_newdata_CMK-AGN(Ours)/FMA_UE/baseline/FMA_UE_fold1.pth" \
    --manifest samples_manifest_tri_4tasks_100subjects.csv
# or predict all four tasks at once:
python src/predict.py --all-tasks
```

### 5. Baselines and ablations

```bash
# List available baselines (cnn1d, eegnet, bilstm_attn, mlp, xgboost, svm_rbf, ...):
PYTHONPATH=src python -m baselines.train_baseline --list

# Train one baseline:
PYTHONPATH=src python -m baselines.train_baseline --model cnn1d --task FMA_UE --fold 0

# Full modality + module ablation sweep (all tasks × 3 folds):
bash scripts/ablation_sweep.sh
```

---

## Repository layout

```
CMK-AGN/
├── simulate_data.py                          # Synthetic data generator (seed 2024)
├── samples_manifest_tri_4tasks_100subjects.csv
├── samples_manifest_tri_4tasks_15subjects.csv
├── bjh_labels.json                           # Per-patient clinical labels (indexer input)
├── splits/                                   # Patient-disjoint 3-fold split
├── src/
│   ├── train.py                              # CMK-AGN single-task trainer
│   ├── predict.py                            # Inference
│   ├── task_config.py                        # Task specs (ranges, classes, encoders)
│   ├── clinical_model.py                     # Unified model wrapper (all 4 tasks)
│   ├── data_indexer_tri_modified.py          # Tri-modal manifest indexer
│   ├── patient_splits.py                     # K-fold patient splitter
│   ├── subject_aggregation.py                # Bag-of-trials → subject-level aggregation
│   ├── aggregate_ablation.py                 # Cross-fold / cross-ablation summary
│   ├── alignment/                            # Weighted-Bayes DTW + tri-modal alignment
│   ├── bjh_io/                               # EEG (.bdf) / EMG (.csv) loaders + cache
│   ├── models/                               # CMK-AGN backbone (adk_mdfan_tri.py)
│   └── baselines/                            # DL + ML baseline comparison framework
├── scripts/ablation_sweep.sh                 # Ablation driver
├── RESULT_newdata_CMK-AGN(Ours)/             # CMK-AGN checkpoints + logs/metrics
├── RESULT_newdata_baseline/                  # CMK-AGN vs. baselines (logs/metrics)
├── RESULT_newdata_ablation/                  # Modality + module ablation runs
└── requirements.txt
```

---

## Data (for reproduction)

Raw recordings are not committed. To reproduce end-to-end, a collaborator needs
the **5 real anchor patients (S1–S5)** — `simulate_data.py` scans these off disk
and synthesizes the remaining 95 subjects from them:

- `BJH/EEG_new/S1_*.bdf … S5_*.bdf` — real EEG recordings (BDF)
- `BJH/EMG_new/S{1..5}_*.csv` — matching EMG CSVs (each also carries the
  embedded IMU columns)

With those in place, run the two commands in **Quick start → 2**. Everything
else (the 95 synthetic subjects, manifest, labels, splits) is regenerated
deterministically under seed 2024, and CMK-AGN can then be trained.
