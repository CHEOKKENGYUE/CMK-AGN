#!/usr/bin/env bash
# Section 4.6 Ablation Studies — full sweep driver.
#
# Modality ablation (7 variants) + Module ablation (3 variants) + 2 extra
# Ours seeds for best-of-3 reporting.  Each variant runs all 4 tasks x 3 folds
# in a single train.py invocation (--fold 0 means "all folds").
#
# Outputs are written under:
#   RESULT_newdata_ablation/modality/<tag>/<task>/
#   RESULT_newdata_ablation/module/<tag>/<task>/
#
# Usage:
#   bash scripts/ablation_sweep.sh                # default seed=2024
#   SEED=42 bash scripts/ablation_sweep.sh        # override base seed
#   TASKS="FMA_UE BI" bash scripts/ablation_sweep.sh   # restrict tasks
#   PYTHON=python3.10 bash scripts/ablation_sweep.sh   # pick interpreter
#   CACHE_DIR=/data/aligned_cache bash scripts/ablation_sweep.sh   # override alignment cache
#   DRY_RUN=1 bash scripts/ablation_sweep.sh      # print commands only
set -euo pipefail

# Force line-buffered stdout/stderr so `... | tee log` runs show train.py
# progress in real time instead of holding everything in the default 8 KB
# block buffer.
export PYTHONUNBUFFERED=1

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PY="${PYTHON:-python}"
SEED="${SEED:-2024}"                 # matches train.py default; baselines used 2024
TASKS_ENV="${TASKS:-FMA_UE BI hand_tone hand_function}"
read -r -a TASKS <<<"$TASKS_ENV"
DRY="${DRY_RUN:-0}"

# Shared on-disk alignment cache. Cache key in train.py is keyed on
# (eeg/emg fstamps, seq_len, alignment mode, eeg_fs, WBYDTWConfig) — NOT on
# --modalities — so every modality variant and --no-mdfan reuses the existing
# `tri` cache. Only --alignment-mode adk_no_dtw / resample regenerate; they
# get their own files in the same dir (different mode → different disk key).
CACHE_DIR="${CACHE_DIR:-.aligned_cache}"
mkdir -p "$CACHE_DIR"

run_one () {                          # $1=tag $2..=extra args
  local tag="$1"; shift
  for task in "${TASKS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[$(date '+%F %T')] tag=$tag  task=$task  seed=$SEED"
    echo "  extra: $*"
    echo "============================================================"
    local cmd=("$PY" src/train.py --task "$task" --fold 0 --seed "$SEED" \
               --ablation-tag "$tag" --cache-dir "$CACHE_DIR" "$@")
    if [ "$DRY" = "1" ]; then
      printf '%q ' "${cmd[@]}"; echo
    else
      "${cmd[@]}"
    fi
  done
}

# ---------------- Modality ablation (7 variants) ----------------
# run_one tri        --modalities eeg+emg+imu        # = Ours, baseline seed
run_one eeg_only   --modalities eeg
run_one emg_only   --modalities emg
run_one imu_only   --modalities imu
run_one eeg_emg    --modalities eeg+emg
run_one eeg_imu    --modalities eeg+imu
run_one emg_imu    --modalities emg+imu

# ---------------- Module ablation (3 variants; Full reuses 'tri') ----------------
run_one wo_mdfan    --no-mdfan
run_one wo_wbydtw   --alignment-mode adk_no_dtw
run_one wo_trialign --alignment-mode resample

# ---------------- Ours multi-seed (best-of-3) ----------------
# for extra_seed in 42 1337; do
#   SEED="$extra_seed" run_one "tri_seed${extra_seed}" --modalities eeg+emg+imu
# done

# ---------------- Aggregate Table 5 / 6 + long.csv ----------------
echo ""
echo "[$(date '+%F %T')] Aggregating ablation summary..."
if [ "$DRY" = "1" ]; then
  echo "$PY src/aggregate_ablation.py --root RESULT_newdata_ablation --out RESULT_newdata_ablation/_summary"
else
  "$PY" src/aggregate_ablation.py \
      --root RESULT_newdata_ablation \
      --out  RESULT_newdata_ablation/_summary
fi

echo ""
echo "[$(date '+%F %T')] Ablation sweep complete."
