#!/usr/bin/env python3
"""simulate_data.py — BJH 4 临床任务 + 康复建议数据扩充（慢性期，100 例）。

变更点（与上一版相比）：
    * 真实锚点扩到 5 例（S1..S5，全部 BDF + EMG 真实文件已在 BJH/ 下）。
    * 移除 ``wrist_tone`` 任务（不再训练）。剩 4 个任务：
        FMA_UE / BI / hand_tone / hand_function。
    * 仅模拟慢性期患者（病程 ≥ 180 天），删除"倒 U"亚急性 bell。
    * 大幅减小 BI 的非上肢噪声；hand_function 采样 σ 减半；标签由 FMA + tone
      共同决定，让指标-信号耦合可学。
    * 信号生成端：ERD/BSI/CMC/agonist_peak/ROM/tremor/rest_baseline 等全部
      重新参数化，让 hand_tone、hand_function 也对信号有"硬"绑定。
    * 新增 ``gen_rehab_text`` —— 为每个患者生成 200 字康复建议文本，输出
      ``patient_rehab_suggestions_150subjects.json``。该 JSON 供下游 LLM
      微调（labels → rehab text 对）。
    * 患者总数扩到 100（5 真实 + 95 synthetic）。

重要免责说明：
    * 真实样本只有 5 名（S1..S5）。
    * 扩充至 100 名所用数据为 SYNTHETIC，仅用于程序流程测试。
    * 这些数据 **不能** 作为模型临床有效性证据。
    * 这些数据 **不能** 用来声称模型具有临床泛化能力。

使用方法：
    conda activate clock
    cd /path/to/ADK-MDFAN
    python simulate_data.py                 # 默认 100 例（5 真实 + 95 synthetic）
    python simulate_data.py --target-n 15   # 验证用：15 例（5 真实 + 10 synthetic），约 15 GB 磁盘
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

# ─── 项目路径 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from task_config import TASK_CONFIGS, get_encoder  # noqa: E402

# ─── 任务设置 ─────────────────────────────────────────────────────────────────
TARGET_N_SUBJECTS = 100
GLOBAL_SEED = 2024
SOURCE_REAL = "real"
SOURCE_SYNTHETIC = "synthetic"

# 4 个真实任务标签（不再含 wrist_tone）。
TASK_NAMES = ("FMA_UE", "BI", "hand_tone", "hand_function")

# 真实 5 例患者锚点 — 与 bjh_labels.json 对齐（不含 wrist_tone）。
ORIG_PATIENT_LABELS: Dict[int, Dict[str, object]] = {
    1: {"FMA_UE": 1,  "BI": 65, "hand_tone": "1+", "hand_function": 3},
    2: {"FMA_UE": 19, "BI": 90, "hand_tone": "0",  "hand_function": 6},
    3: {"FMA_UE": 18, "BI": 60, "hand_tone": "0",  "hand_function": 6},
    4: {"FMA_UE": 18, "BI": 90, "hand_tone": "0",  "hand_function": 6},
    5: {"FMA_UE": 0,  "BI": 20, "hand_tone": "1+", "hand_function": 3},
}

# 真实 5 例的人口学与最终康复建议文本（解析自 bjh_Rehabilitation_Suggestions.txt）。
# 文本一字不改地写入 ``patient_rehab_suggestions_*.json``，作为 LLM 微调
# 数据中的金标准样本。
ORIG_PATIENT_DEMOGRAPHICS: Dict[int, Dict[str, object]] = {
    1: {"gender": "女", "age": 28, "disease": "脑外伤", "days_post": 150, "affected_side": "R"},
    2: {"gender": "男", "age": 72, "disease": "脑梗死", "days_post": 20,  "affected_side": "L"},
    3: {"gender": "男", "age": 69, "disease": "脑梗死", "days_post": 180, "affected_side": "L"},
    4: {"gender": "男", "age": 72, "disease": "脑梗死", "days_post": 30,  "affected_side": "L"},
    5: {"gender": "男", "age": 69, "disease": "脑出血", "days_post": 170, "affected_side": "L"},
}

ORIG_PATIENT_REHAB_TEXTS: Dict[int, str] = {
    1: ("患者S1，女性，28岁，脑外伤，病程5个月，右侧偏瘫。当前FMA手1分，提示手严重运动功能障碍；"
        "BI评分65分，日常生活需要帮助，进食、修饰、洗澡、穿脱衣服需要帮助。"
        "手部肌张力1+级，手分期为Brunnstrom 3期，表明手肌张力轻度增加，可完成手指钩状抓握，但不能伸指。"
        "建议继续增强主动运动，诱发分离运动，结合手机器人训练，提升手部主动运动。"),
    2: ("患者S2，男性，72岁，脑梗死，病程20天，左侧偏瘫。当前FMA手19分，提示手轻度运动功能障碍；"
        "BI评分90分，日常生活基本自理，洗澡、上下楼梯需要帮助。"
        "手部肌张力0级，手分期为Brunnstrom 6期，表明手肌张力正常，手所有抓握均能完成，但速度和准确性比健侧差。"
        "建议增强手灵活性和协调性训练，结合手机器人训练，提升手实用功能。"),
    3: ("患者S3，男性,69岁，脑梗死，病程180天，左侧偏瘫。当前FMA手18分，提示手轻度运动功能障碍；"
        "BI评分60分，日常需要帮助，转移、行走、上下楼梯需要帮助。"
        "手部肌张力0级，手分期为Brunnstrom 6期，表明手肌张力正常，手所有抓握均能完成，但速度和准确性比健侧差。"
        "建议增强手灵活性和协调性训练，结合手机器人训练，提升手实用功能。"),
    4: ("患者S4，男性，72岁，脑梗死，病程30天，左侧偏瘫。当前FMA手18分，提示手轻度运动功能障碍；"
        "BI评分90分，日常生活基本自理，洗澡、上下楼梯需要帮助。"
        "手部肌张力0级，手分期为Brunnstrom 6期，表明手肌张力正常，手所有抓握均能完成，但速度和准确性比健侧差。"
        "建议增强手灵活性和协调性训练，结合手机器人训练，提升手实用功能。"),
    5: ("患者S5，男性，69岁，脑出血，病程170天，左侧偏瘫。当前FMA手0分，提示手严重运动功能障碍；"
        "BI评分20分，日常生活完全依赖，进食、转移、修饰、上厕所、洗澡、行走、上下楼梯、穿脱衣服需要帮助。"
        "手部肌张力1+级，手分期为Brunnstrom 3期，表明手肌张力轻度增加，手指钩状抓握状态，不能伸指。"
        "建议继续增强主动运动，诱发分离运动，结合手机器人训练，提升手部主动运动。"),
}

# 标签合法范围（也由 task_config.py 表达，这里再写一份用于校验/采样）。
FMA_RANGE = (int(TASK_CONFIGS["FMA_UE"].score_min), int(TASK_CONFIGS["FMA_UE"].score_max))
BI_RANGE = (int(TASK_CONFIGS["BI"].score_min), int(TASK_CONFIGS["BI"].score_max))
HAND_TONE_CLASSES = list(TASK_CONFIGS["hand_tone"].classes)
HAND_FUNCTION_CLASSES = list(TASK_CONFIGS["hand_function"].classes)


# ─── 信号采样参数（与真实 BJH 数据一致）────────────────────────────────────────
EEG_FS = 1000.0
EMG_FS = 1259.4
IMU_FS = 148.1
DURATION = 175.0

N_EEG = int(DURATION * EEG_FS)
N_EMG = int(DURATION * EMG_FS)
N_IMU = int(DURATION * IMU_FS)

EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8",
    "T3", "C3", "Cz", "C4", "T4",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "A1", "T5", "P3", "Pz", "P4", "T6", "A2",
    "O1", "Oz", "O2",
]

# 半球分组：用于建模患侧/健侧 mu/beta 不对称（BSI）。
LEFT_HEMI_CHANNELS = {"Fp1", "F7", "F3", "FT7", "FC3", "T3", "C3",
                      "TP7", "CP3", "T5", "P3", "O1", "A1"}
RIGHT_HEMI_CHANNELS = {"Fp2", "F8", "F4", "FT8", "FC4", "T4", "C4",
                       "TP8", "CP4", "T6", "P4", "O2", "A2"}
MOTOR_CHANNELS = {"C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4"}

# 肌肉：前 2 块屈肌（agonist），后 2 块伸肌（antagonist）。
MUSCLES = [
    "R FLEXOR CARPI RADIALIS",    # idx 0  屈腕（agonist）
    "R PALMARIS LONGUS",          # idx 1  屈腕/握（agonist）
    "R EXTENSOR CARPI ULNARIS",   # idx 2  伸腕（antagonist）
    "R EXTENSOR DIGITORUM",       # idx 3  伸指（antagonist）
]
AGONIST_IDX = (0, 1)
ANTAGONIST_IDX = (2, 3)

IMU_AXES = ["ACC.X", "ACC.Y", "ACC.Z", "GYRO.X", "GYRO.Y", "GYRO.Z"]
TASK_IDS = [1, 2, 4, 5, 6, 7]

# 整体信号尺度（与真实 BJH 数据测得 std 同量级）。
EEG_UNIT_SCALE = 6e-4
EMG_UNIT_SCALE = 5e-5
IMU_UNIT_SCALE = 0.01

# 张力类别 → 数值严重度（用于影响信号生成）。
TONE_SEVERITY = {"0": 0.0, "1": 0.2, "1+": 0.35, "2": 0.55, "3": 0.78, "4": 1.0}


# --------------------------------------------------------------------------- #
# Label sampling (chronic phase, 4 tasks)                                     #
# --------------------------------------------------------------------------- #
def _hand_tone_from_norm(norm: float, rng: np.random.Generator) -> str:
    """慢性期：低 FMA → 高痉挛，单调负相关。

    σ=0.10 让分布能覆盖全 6 类，但任一 norm 下方差小（信号-标签耦合可学）。
    """
    sev = 0.85 - 0.85 * float(norm) + float(rng.normal(0.0, 0.10))
    sev = float(np.clip(sev, 0.0, 1.0))
    return min(HAND_TONE_CLASSES, key=lambda c: abs(TONE_SEVERITY[c] - sev))


def _sample_labels_from_fma(fma: int, rng: np.random.Generator) -> Dict[str, object]:
    """Sample BI / hand_tone / hand_function conditionally on FMA (慢性期).

    临床关系（慢性期）：
      * FMA_UE → hand_function：强正相关
      * FMA_UE + hand_tone → BI：高 FMA + 低 tone → 高 BI（不含下肢/认知大噪声）
      * FMA_UE → hand_tone：单调负相关（无亚急性 bell）
    """
    norm = float(np.clip((fma - FMA_RANGE[0]) / (FMA_RANGE[1] - FMA_RANGE[0]), 0.0, 1.0))

    hand_tone = _hand_tone_from_norm(norm, rng)
    tone_sev = TONE_SEVERITY[hand_tone]

    # BI: 慢性期上肢主导。去掉 σ=22 的下肢/认知噪声 → 只保留 σ=6 临床测量噪声。
    bi_mean = BI_RANGE[0] + 75.0 * norm - 15.0 * tone_sev
    bi = int(np.clip(rng.normal(bi_mean, 6.0), BI_RANGE[0], BI_RANGE[1]))

    # Hand function: 由 FMA 主导（强相关），tone 显著压低。σ=0.35（原 0.7）。
    hf_mean = 1.0 + norm * 5.0 - 1.2 * tone_sev
    hf_idx = int(np.clip(round(rng.normal(hf_mean, 0.35)), 1, 6))

    # Clinical coherence #1: tone floor — 痉挛期 Brunnstrom 不低于对应下限。
    if tone_sev >= TONE_SEVERITY["2"]:
        hf_idx = max(hf_idx, 2)
    if tone_sev >= TONE_SEVERITY["3"]:
        hf_idx = max(hf_idx, 3)

    # Clinical coherence #2: FMA cap — 极低 FMA 不能配高 Brunnstrom。
    # FMA 0: 无任何主动手运动 → Brunnstrom 最高 2 期（微弱随意运动迹象）。
    # FMA 1-3: 重度受损 → Brunnstrom 最高 3 期（钩状抓握为上限）。
    # 此约束优先于上方的 tone floor，防止"FMA=0 + 高痉挛 → 强制 hf=3"的矛盾。
    if fma <= 0:
        hf_idx = min(hf_idx, 2)
    elif fma <= 3:
        hf_idx = min(hf_idx, 3)

    # Clinical coherence #3: hf=1（弛缓期）→ tone 不能高（无张力不能同时有高痉挛）。
    # 若 FMA cap 将 hf 降回 1，同步将 tone 压到 "0"/"1"。
    if hf_idx == 1 and tone_sev > TONE_SEVERITY["1"]:
        hand_tone = str(rng.choice(["0", "1"]))
        tone_sev = TONE_SEVERITY[hand_tone]  # noqa: F841 (kept for symmetry)

    return {
        "FMA_UE": int(fma),
        "BI": bi,
        "hand_tone": hand_tone,
        # hf_idx is already the Brunnstrom stage integer (1..6). Emit it
        # directly; do NOT index into TASK_CONFIGS["hand_function"].classes,
        # which now enumerates only the 5 retained stages (2..6) — see the
        # stage-1 → stage-2 reassignment in make_new_subjects().
        "hand_function": int(hf_idx),
    }


# --------------------------------------------------------------------------- #
# Synthetic patient demographics (chronic, plausible distributions)           #
# --------------------------------------------------------------------------- #
GENDER_CHOICES = ("男", "女")
GENDER_PROBS = (0.60, 0.40)
DISEASE_CHOICES = ("脑梗死", "脑出血", "脑外伤")
DISEASE_PROBS = (0.65, 0.25, 0.10)
SIDE_CHOICES = ("L", "R")
SIDE_PROBS = (0.55, 0.45)

# 慢性期窗口：6 个月 ~ 3 年。
CHRONIC_DAYS_MIN = 180
CHRONIC_DAYS_MAX = 1080


def _sample_demographics(rng: np.random.Generator) -> Dict[str, object]:
    return {
        "gender": str(rng.choice(GENDER_CHOICES, p=GENDER_PROBS)),
        "age": int(np.clip(round(rng.normal(65.0, 10.0)), 40, 88)),
        "disease": str(rng.choice(DISEASE_CHOICES, p=DISEASE_PROBS)),
        "days_post": int(rng.integers(CHRONIC_DAYS_MIN, CHRONIC_DAYS_MAX + 1)),
        "affected_side": str(rng.choice(SIDE_CHOICES, p=SIDE_PROBS)),
    }


def make_new_subjects(
    target_n: int = TARGET_N_SUBJECTS,
    first_new_id: int = 6,
    seed: int = GLOBAL_SEED,
) -> list[dict]:
    """Generate S{first_new_id}..S{target_n} synthetic chronic-phase patients."""
    rng = np.random.default_rng(seed)
    n_existing = len(ORIG_PATIENT_LABELS)
    n_new = target_n - n_existing
    if n_new <= 0:
        return []

    # 慢性期 FMA 分布：4 个 bin，权重轻微上移到中-高段（恢复累积）。
    bins = [(0, 4), (5, 10), (11, 16), (17, 20)]
    probs = np.array([0.18, 0.24, 0.30, 0.28], dtype=float)
    probs = probs / probs.sum()

    subjects: list[dict] = []
    for i in range(n_new):
        sid = first_new_id + i
        lo, hi = bins[int(rng.choice(len(bins), p=probs))]
        fma = int(rng.integers(lo, hi + 1))
        labels = _sample_labels_from_fma(fma, rng)
        n_trials = int(rng.integers(5, 7))
        demographics = _sample_demographics(rng)
        subjects.append({
            "id": sid,
            "n_trials": n_trials,
            "demographics": demographics,
            **labels,
        })

    # ---------------------------------------------------------------------- #
    # Brunnstrom stage-1 → stage-2 reassignment (deterministic under seed).   #
    # ---------------------------------------------------------------------- #
    # Under GLOBAL_SEED=2024 exactly one synthetic subject (S83) samples to
    # Brunnstrom stage 1 (`hand_function == 1`). Its other indicators —
    # FMA-UE=5 (>0, i.e. measurable voluntary hand movement) and hand MAS=0 —
    # are inconsistent with the complete flaccid paralysis of stage 1, and the
    # stage-1 / stage-2 boundary is not sharply delimited clinically. We
    # therefore reassign any stage-1 synthetic subject to stage 2, which also
    # makes `hand_function` a clean 5-class task (stages 2–6). This keeps a
    # from-scratch regeneration consistent with the shipped manifest.
    for subj in subjects:
        if subj["hand_function"] == 1:
            subj["hand_function"] = 2

    return subjects


NEW_SUBJECTS = make_new_subjects()


def score_norm(fma: float) -> float:
    return float(np.clip((fma - FMA_RANGE[0]) / (FMA_RANGE[1] - FMA_RANGE[0]), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Subject signal profile — 4 个标签全部向 profile 注入控制点                  #
# --------------------------------------------------------------------------- #
def subject_profile(
    fma: float,
    rng: np.random.Generator,
    *,
    affected_side: str = "R",
    hand_tone: str = "0",
    hand_function: int = 6,
) -> Dict[str, float]:
    """Build per-subject signal profile.

    所有 4 个标签都参与 profile：
      * FMA_UE     → norm           → ERD 深度、BSI 基线、ROM、agonist_peak、CMC
      * hand_tone  → tone_sev       → rest_baseline、stretch_burst、tremor、慢波
      * hand_function → hf_norm     → coactivation、jerk、ROM、agonist_peak、CMC
      * affected_side → 半球不对称的方向
    """
    norm = score_norm(fma)
    tone_sev = float(TONE_SEVERITY.get(hand_tone, 0.0))
    hf_norm = (int(hand_function) - 1) / 5.0  # 0..1

    # BSI：低 FMA + 低 hand_function → 患侧半球 mu/beta 抑制更重。
    bsi = 0.7 * (1.0 - norm) + 0.10 * (1.0 - hf_norm) + 0.04 * rng.uniform(-1, 1)
    bsi = float(np.clip(bsi, 0.0, 0.85))

    # 协同稳定性：高功能 → 试次间变异更小。
    trial_var = 0.04 + 0.22 * (1.0 - hf_norm)

    return {
        "norm": norm,
        "tone_sev": tone_sev,
        "hf_norm": float(hf_norm),
        "affected_side": affected_side,
        "bsi": bsi,
        "trial_var": float(trial_var),
        # 个体差异（量级 ~lognormal(0, 0.15)，比上一版略收窄以减少不可解释方差）。
        "eeg_gain": float(rng.lognormal(mean=0.0, sigma=0.15)),
        "emg_gain": float(rng.lognormal(mean=0.0, sigma=0.18)),
        "imu_gain": float(rng.lognormal(mean=0.0, sigma=0.18)),
        "noise_gain": float(rng.lognormal(mean=0.0, sigma=0.15)),
        "fatigue": float(rng.uniform(0.90, 1.10)),
        # 牵张反射 burst 率（次/秒）：随 tone ↑，低张力时趋近 0。陡度增加。
        "stretch_burst_rate": float(0.03 + 5.0 * tone_sev),
        # 静息 EMG 基线（相对量）：高 tone → 基线显著抬升。
        "rest_emg_baseline": float(0.10 + 1.15 * tone_sev),
        # 震颤幅度（相对量）：几乎全部由 tone 决定（小贡献来自 norm）。
        "tremor_amp": float(0.05 + 0.85 * tone_sev + 0.15 * (1.0 - norm)),
        # 拮抗肌共激活比：低 hand_function + 高 tone → 显著共激活。
        "coactivation": float(np.clip(0.15 + 0.60 * (1.0 - hf_norm) + 0.45 * tone_sev,
                                      0.10, 1.30)),
        # IMU 抖动幅度：直接由 hand_function 决定。
        "jerk_amp": float(0.30 * (1.0 - hf_norm)),
        # 启动延迟（秒）：低功能 + 低 FMA → ↑。
        "onset_delay": float(0.10 + 0.40 * (1.0 - norm) + 0.20 * (1.0 - hf_norm)),
        # CMC 强度：FMA + hand_function 共同驱动。
        "cmc_strength": float(0.04 + 0.30 * norm + 0.15 * hf_norm),
    }


# ─── 信号生成函数 ─────────────────────────────────────────────────────────────
def _pink_noise(n: int, rng: np.random.Generator, fs: float) -> np.ndarray:
    """1/f (pink) noise via spectral shaping — used for EEG background."""
    white = rng.standard_normal(n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    freqs[0] = 1.0
    pink = np.fft.irfft(fft / np.sqrt(freqs), n=n)
    return pink / (pink.std() + 1e-12)


def _task_envelope(n: int, fs: float, task_id: int, profile: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Trial-level slow envelope shared across modalities (CMC + ERD/ERS 协同)."""
    t = np.arange(n) / fs
    movement_rate = 0.25 + 0.10 * (task_id % 3)
    onset = float(np.clip(profile["onset_delay"]
                          + rng.uniform(-0.05, 0.05), 0.0, 1.5))
    base_phase = 2 * np.pi * movement_rate * (t - onset)
    raw = np.clip(np.sin(base_phase), 0.0, None)
    jitter = 1.0 + profile["trial_var"] * np.sin(
        2 * np.pi * rng.uniform(0.05, 0.25) * t + rng.uniform(0, 2 * np.pi))
    env = raw * np.clip(jitter, 0.2, 1.8)
    win = max(1, int(0.3 * fs))
    kernel = np.ones(win) / win
    env = np.convolve(env, kernel, mode="same")
    env = env / (env.max() + 1e-9)
    env[t < onset] = 0.0
    return env.astype(np.float32)


def gen_eeg(n: int, fma: float, task_id: int, rng: np.random.Generator,
            profile: Dict[str, float],
            task_env: np.ndarray | None = None) -> np.ndarray:
    """EEG: 1/f 背景 + 半球不对称的 mu/beta + ERD（由 FMA + hand_function 驱动）
    + 慢波抬升（由 tone 增强）+ motor 区 CMC 注入。
    """
    t = np.arange(n) / EEG_FS
    norm = profile["norm"]
    bsi = profile["bsi"]
    affected_side = profile["affected_side"]
    eeg_gain = profile["eeg_gain"]
    noise_gain = profile["noise_gain"]
    tone_sev = profile["tone_sev"]
    hf_norm = profile["hf_norm"]
    cmc_strength = profile["cmc_strength"]

    out = np.zeros((n, len(EEG_CHANNELS)), dtype=np.float32)

    if task_env is None or task_env.shape[0] != n:
        task_env = _task_envelope(n, EEG_FS, task_id, profile, rng)

    # ERD：任务期内 mu/beta 抑制深度由 norm + hf_norm 决定。
    erd_depth = 0.15 + 0.45 * norm + 0.20 * hf_norm   # ∈ [0.15, 0.80]
    erd_gate = 1.0 - erd_depth * task_env             # ∈ [1 - 0.80, 1]

    # motor-area band 振荡。
    mu_freq = 10.0 + rng.uniform(-1.0, 1.0)
    beta_freq = 20.0 + rng.uniform(-3.0, 3.0)
    theta_freq = 6.0 + rng.uniform(-1.0, 1.0)
    mu_phase = rng.uniform(0, 2 * np.pi)
    beta_phase = rng.uniform(0, 2 * np.pi)
    theta_phase = rng.uniform(0, 2 * np.pi)
    mu_wave = np.sin(2 * np.pi * mu_freq * t + mu_phase)
    beta_wave = np.sin(2 * np.pi * beta_freq * t + beta_phase)
    theta_wave = np.sin(2 * np.pi * theta_freq * t + theta_phase)

    paretic_hemi_channels = (RIGHT_HEMI_CHANNELS if affected_side == "L"
                             else LEFT_HEMI_CHANNELS)

    for ch_i, ch_name in enumerate(EEG_CHANNELS):
        pink = _pink_noise(n, rng, EEG_FS)
        # 慢波（delta/theta）：低 FMA → 明显增强；高 tone 进一步抬升。
        slow_factor = 1.0 + 2.0 * (1.0 - norm) + 0.9 * tone_sev
        if ch_name in MOTOR_CHANNELS:
            motor_w = 1.30
        elif ch_name in {"Fp1", "Fp2", "O1", "Oz", "O2"}:
            motor_w = 0.75
        else:
            motor_w = 0.95
        hemi_atten = (1.0 - bsi) if ch_name in paretic_hemi_channels else 1.0
        alpha_amp = 0.18 * eeg_gain * motor_w * hemi_atten
        beta_amp = 0.10 * eeg_gain * motor_w * hemi_atten

        if ch_name in MOTOR_CHANNELS:
            mu_ch = alpha_amp * mu_wave * erd_gate
            beta_ch = beta_amp * beta_wave * erd_gate
        else:
            mu_ch = alpha_amp * mu_wave
            beta_ch = beta_amp * beta_wave

        background = 0.40 * noise_gain * pink * slow_factor
        theta_bg = 0.12 * slow_factor * theta_wave

        if ch_name in MOTOR_CHANNELS:
            cmc_inj = cmc_strength * 0.20 * task_env * beta_wave * hemi_atten
        else:
            cmc_inj = 0.0

        sig = (background + theta_bg + mu_ch + beta_ch + cmc_inj)
        out[:, ch_i] = (sig * EEG_UNIT_SCALE).astype(np.float32)

    return out


def gen_emg(n: int, fma: float, task_id: int, rng: np.random.Generator,
            profile: Dict[str, float],
            task_env: np.ndarray | None = None) -> np.ndarray:
    """EMG: agonist 峰值由 FMA + hand_function 决定；拮抗肌共激活由 hand_function
    + tone 决定；牵张反射 burst 由 tone 决定；静息基线由 tone 决定；CMC 与 EEG
    motor 区共享 envelope。
    """
    t = np.arange(n) / EMG_FS
    norm = profile["norm"]
    hf_norm = profile["hf_norm"]
    emg_gain = profile["emg_gain"]
    noise_gain = profile["noise_gain"]
    fatigue = profile["fatigue"]
    tone_sev = profile["tone_sev"]
    rest_baseline = profile["rest_emg_baseline"]
    coact = profile["coactivation"]
    burst_rate = profile["stretch_burst_rate"]
    cmc_strength = profile["cmc_strength"]

    out = np.zeros((n, len(MUSCLES)), dtype=np.float32)

    if task_env is None or task_env.shape[0] != n:
        task_env = _task_envelope(n, EMG_FS, task_id, profile, rng)

    # Agonist 峰值：FMA 与 hand_function 共同驱动（弱手功能 + 弱 FMA → 微弱峰值）。
    agonist_peak = (0.20 + 1.2 * norm + 0.5 * hf_norm) * emg_gain
    antag_peak = agonist_peak * coact

    n_bursts = int(burst_rate * DURATION)
    burst_centers = rng.uniform(0, n / EMG_FS, size=max(0, n_bursts))
    burst_durations = rng.uniform(0.05, 0.12, size=len(burst_centers))
    burst_amp = (0.6 + 0.4 * tone_sev) * agonist_peak * 1.5

    def burst_mask(scale: float) -> np.ndarray:
        m = np.zeros(n, dtype=np.float32)
        for c, d in zip(burst_centers, burst_durations):
            i0 = max(0, int((c - d / 2) * EMG_FS))
            i1 = min(n, int((c + d / 2) * EMG_FS))
            if i1 > i0:
                k = np.sin(np.linspace(0, np.pi, i1 - i0)) ** 2
                m[i0:i1] = np.maximum(m[i0:i1], k * scale)
        return m

    flex_bursts = burst_mask(1.0)
    ext_bursts = burst_mask(0.55)

    for m_idx in range(len(MUSCLES)):
        carrier = rng.standard_normal(n).astype(np.float32)

        if m_idx in AGONIST_IDX:
            peak = agonist_peak
            envelope = task_env
            local_burst = flex_bursts * burst_amp
        else:
            peak = antag_peak
            shift = int(0.05 * EMG_FS)
            antag_env = np.concatenate([np.zeros(shift), task_env])[:n]
            envelope = 0.6 * antag_env + 0.4 * task_env
            local_burst = ext_bursts * burst_amp

        active = peak * envelope * np.abs(carrier)

        baseline = rest_baseline * 0.05 * emg_gain * np.abs(rng.standard_normal(n))

        instr_noise = (0.035 + 0.035 * (1.0 - norm)) * noise_gain * np.abs(
            rng.standard_normal(n))
        crosstalk = 0.02 * np.abs(rng.standard_normal(n))

        if m_idx in AGONIST_IDX:
            cmc_term = cmc_strength * 0.25 * peak * task_env * np.abs(
                np.sin(2 * np.pi * 22.0 * t + rng.uniform(0, 2 * np.pi))) * (
                np.abs(carrier) ** 0.5)
        else:
            cmc_term = 0.0

        sig = fatigue * active + baseline + instr_noise + crosstalk + local_burst + cmc_term
        out[:, m_idx] = (np.maximum(sig, 0) * EMG_UNIT_SCALE).astype(np.float32)

    return out


def gen_imu(n: int, fma: float, task_id: int, rng: np.random.Generator,
            profile: Dict[str, float],
            task_env: np.ndarray | None = None) -> np.ndarray:
    """IMU: ROM 由 FMA + hand_function 共同决定；震颤由 tone 决定；jerk 直接由
    hand_function 决定（低功能 → 高 jerk）。
    """
    t = np.arange(n) / IMU_FS
    norm = profile["norm"]
    hf_norm = profile["hf_norm"]
    imu_gain = profile["imu_gain"]
    noise_gain = profile["noise_gain"]
    tremor_amp = profile["tremor_amp"]
    jerk_amp = profile["jerk_amp"]

    if task_env is None or task_env.shape[0] != n:
        task_env = _task_envelope(n, IMU_FS, task_id, profile, rng)

    # ROM 由 FMA + hand_function 共同驱动。
    rom = (0.15 + 1.1 * norm + 0.6 * hf_norm) * imu_gain

    n_a = len(IMU_AXES)
    out = np.zeros((n, len(MUSCLES) * n_a), dtype=np.float32)

    for m_idx in range(len(MUSCLES)):
        primary_freq = 0.4 + 0.18 * (task_id + m_idx) + rng.uniform(-0.05, 0.05)
        for a in range(n_a):
            col = m_idx * n_a + a
            axis_scale = 1.0 if a < 3 else 0.5   # ACC vs GYRO 量级
            phase = rng.uniform(0, 2 * np.pi) + a * (np.pi / 3)

            base = axis_scale * rom * task_env * np.sin(
                2 * np.pi * primary_freq * t + phase)

            tremor_f = rng.uniform(4.0, 7.0)
            tremor = axis_scale * tremor_amp * 0.5 * np.sin(
                2 * np.pi * tremor_f * t + phase) * (0.3 + 0.7 * task_env)

            # Jerk：直接由 hand_function 决定（低 hf → 大 jerk）。
            jerk = axis_scale * jerk_amp * rng.standard_normal(n) * task_env

            sensor_noise = axis_scale * (0.04 + 0.04 * (1.0 - norm)) * noise_gain \
                * rng.standard_normal(n)

            sig = base + tremor + jerk + sensor_noise
            out[:, col] = (sig * IMU_UNIT_SCALE).astype(np.float32)

    return out


# ─── CSV 文件写入 ─────────────────────────────────────────────────────────────
def write_eeg_csv(path: Path, eeg: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eeg, columns=EEG_CHANNELS).to_csv(path, index=False, float_format="%.6g")


def _imu_zoh_upsample(imu: np.ndarray, n_emg: int) -> np.ndarray:
    """Upsample IMU from N_IMU to N_EMG using zero-order hold.

    The real BioSignals Plux device writes one IMU reading per EMG row: the IMU
    counter ticks at IMU_FS rate but the device streams at EMG_FS rate, so each
    physical IMU sample is held for round(EMG_FS/IMU_FS) ≈ 8.5 EMG rows.
    """
    n_imu = imu.shape[0]
    out = np.empty((n_emg, imu.shape[1]), dtype=np.float32)
    for imu_i in range(n_imu):
        emg_start = round(imu_i * n_emg / n_imu)
        emg_end = round((imu_i + 1) * n_emg / n_imu)
        out[emg_start:emg_end] = imu[imu_i]
    return out


def write_emg_imu_csv(path: Path, emg: np.ndarray, imu: np.ndarray) -> None:
    n_emg = emg.shape[0]
    n_m = len(MUSCLES)
    n_a = len(IMU_AXES)

    hdr = []
    for mi, muscle in enumerate(MUSCLES):
        num = mi + 1
        hdr += ["X[s]", f"{muscle}: EMG {num}"]
        for axis in IMU_AXES:
            hdr += ["X[s]", f"{muscle}: {axis} {num}"]

    t_emg = np.arange(n_emg, dtype=np.float64) / EMG_FS
    # IMU timestamp mirrors real device: counter advances at 1/IMU_FS per row
    # regardless of EMG rate, so the timestamp reaches DURATION*(EMG_FS/IMU_FS).
    t_imu = np.arange(n_emg, dtype=np.float64) / IMU_FS
    # ZOH-upsample from N_IMU physical samples to N_EMG rows (matches real device).
    imu_up = _imu_zoh_upsample(imu, n_emg)

    n_cols = n_m * (2 + 2 * n_a)
    data = np.empty((n_emg, n_cols), dtype=np.float64)
    col = 0
    for mi in range(n_m):
        data[:, col] = t_emg; col += 1
        data[:, col] = emg[:, mi].astype(np.float64); col += 1
        for ai in range(n_a):
            data[:, col] = t_imu; col += 1
            data[:, col] = imu_up[:, mi * n_a + ai]; col += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(hdr) + "\n")
    with open(path, "ab") as f:
        np.savetxt(f, data, delimiter=",", fmt="%.8g")


# ─── 数据生成 ─────────────────────────────────────────────────────────────────
def task_and_trial_number(ti: int) -> tuple[int, int]:
    task_id = TASK_IDS[ti % len(TASK_IDS)]
    trial_number = ti // len(TASK_IDS) + 1
    return task_id, trial_number


def generate_data(root: Path, seed: int = GLOBAL_SEED) -> None:
    """Generate synthetic EEG/EMG/IMU CSV files for the new patients only.

    Real S1..S5 trial files are NOT touched.
    """
    eeg_dir = root / "BJH" / "EEG_new"
    emg_dir = root / "BJH" / "EMG_new"
    master_rng = np.random.default_rng(seed)

    for subj in NEW_SUBJECTS:
        sid = subj["id"]
        fma = int(subj["FMA_UE"])
        n_trials = int(subj["n_trials"])
        affected_side = str(subj["demographics"]["affected_side"])
        profile_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        profile = subject_profile(
            fma,
            profile_rng,
            affected_side=affected_side,
            hand_tone=subj["hand_tone"],
            hand_function=int(subj["hand_function"]),
        )

        print(f"  S{sid:03d} (FMA={fma:2d}, side={affected_side}, "
              f"tone={subj['hand_tone']}, hf={subj['hand_function']}): ",
              end="", flush=True)
        for ti in range(n_trials):
            task_id, trial_number = task_and_trial_number(ti)
            rng = np.random.default_rng(master_rng.integers(0, 2**31))
            env_seed = int(rng.integers(0, 2**31))
            env_eeg = _task_envelope(N_EEG, EEG_FS, task_id, profile,
                                     np.random.default_rng(env_seed))
            env_emg = _task_envelope(N_EMG, EMG_FS, task_id, profile,
                                     np.random.default_rng(env_seed))
            env_imu = _task_envelope(N_IMU, IMU_FS, task_id, profile,
                                     np.random.default_rng(env_seed))
            eeg = gen_eeg(N_EEG, fma, task_id, rng, profile, task_env=env_eeg)
            emg = gen_emg(N_EMG, fma, task_id, rng, profile, task_env=env_emg)
            imu = gen_imu(N_IMU, fma, task_id, rng, profile, task_env=env_imu)
            fname = f"S{sid}_{task_id}_{trial_number}.csv"
            write_eeg_csv(eeg_dir / fname, eeg)
            write_emg_imu_csv(emg_dir / fname, emg, imu)
            print(f"T{task_id}-{trial_number}", end=" ", flush=True)
        print()


# --------------------------------------------------------------------------- #
# Rehab suggestion text generation                                            #
# --------------------------------------------------------------------------- #
def _fma_band_text(fma: int) -> str:
    if fma <= 3:
        return "手严重运动功能障碍"
    if fma <= 10:
        return "手中度运动功能障碍"
    if fma <= 17:
        return "手轻-中度运动功能障碍"
    return "手轻度运动功能障碍"


def _bi_band_text(bi: int) -> str:
    if bi <= 20:
        return ("日常生活完全依赖，进食、转移、修饰、上厕所、洗澡、行走、"
                "上下楼梯、穿脱衣服需要帮助")
    if bi <= 40:
        return "日常生活重度依赖，多数活动需要帮助"
    if bi <= 60:
        return "日常需要帮助，转移、行走、上下楼梯需要帮助"
    if bi <= 80:
        return "日常生活需要帮助，进食、修饰、洗澡、穿脱衣服需要帮助"
    if bi <= 95:
        return "日常生活基本自理，洗澡、上下楼梯需要帮助"
    return "日常生活完全自理"


TONE_TEXT = {
    "0": "手肌张力正常",
    "1": "手肌张力轻度增加",
    "1+": "手肌张力轻度增加",
    "2": "手肌张力中度增加",
    "3": "手肌张力明显增加",
    "4": "手呈强直状态",
}

# Brunnstrom 期能力（Brunnstrom = hand_function 1..6）。
BRUNNSTROM_CAPABILITY = {
    1: "手呈完全弛缓状态",
    2: "可见轻微随意运动迹象",
    3: "可完成手指钩状抓握",
    4: "可完成部分分离动作",
    5: "可完成大部分分离运动与侧捏",
    6: "手所有抓握均能完成，但速度和准确性比健侧差",
}

BRUNNSTROM_LIMITATION = {
    1: "尚不能完成任何主动抓握",
    2: "随意运动幅度极小，无法独立完成抓握",
    3: "不能伸指，精细抓握不能",
    4: "对掌、各指独立运动仍不足",
    5: "球形/柱形抓握速度与协调仍不足",
    6: "灵活性与精确度尚未达到健侧水平",
}

TRAINING_PLAN = {
    1: ("继续增强主动运动，诱发分离运动", "手机器人", "提升手部主动运动"),
    2: ("继续增强主动运动，诱发分离运动", "手机器人", "提升手部主动运动"),
    3: ("继续增强主动运动，诱发分离运动", "手机器人", "提升手部主动运动"),
    4: ("强化分离动作训练，重点对掌与各指独立活动", "手机器人", "提升手部分离运动"),
    5: ("强化精细抓握与对掌训练", "手机器人", "提升手部精细动作"),
    6: ("增强手灵活性和协调性训练", "手机器人", "提升手实用功能"),
}

SIDE_ZH = {"L": "左", "R": "右"}


def gen_rehab_text(subject_id: int,
                   demographics: Dict[str, object],
                   labels: Dict[str, object]) -> str:
    """Generate a ~200-character rehab suggestion text (deterministic given inputs)."""
    fma = int(labels["FMA_UE"])
    bi = int(labels["BI"])
    tone = str(labels["hand_tone"])
    hf = int(labels["hand_function"])

    gender = str(demographics["gender"])
    age = int(demographics["age"])
    disease = str(demographics["disease"])
    days = int(demographics["days_post"])
    side = str(demographics["affected_side"])
    side_zh = SIDE_ZH.get(side, "右")

    primary, aux, goal = TRAINING_PLAN[hf]
    return (
        f"患者S{subject_id}，{gender}性，{age}岁，{disease}，病程{days}天，{side_zh}侧偏瘫。"
        f"当前FMA手{fma}分，提示{_fma_band_text(fma)}；"
        f"BI评分{bi}分，{_bi_band_text(bi)}；"
        f"手部肌张力{tone}级，手分期为Brunnstrom {hf}期，"
        f"表明{TONE_TEXT[tone]}，{BRUNNSTROM_CAPABILITY[hf]}，但{BRUNNSTROM_LIMITATION[hf]}。"
        f"建议{primary}，结合{aux}训练，{goal}。"
    )


def write_rehab_suggestions_json(root: Path) -> Path:
    """Write {subject_id: {rehab_text, demographics, source}} for all 150 patients.

    Real S1..S5 use the verbatim text from bjh_Rehabilitation_Suggestions.txt;
    synthetic S6..S150 are template-generated from labels + demographics.
    """
    out: Dict[str, Dict[str, object]] = {}
    for sid, demos in ORIG_PATIENT_DEMOGRAPHICS.items():
        out[str(sid)] = {
            "rehab_text": ORIG_PATIENT_REHAB_TEXTS[sid],
            "demographics": demos,
            "labels": {
                "FMA_UE": ORIG_PATIENT_LABELS[sid]["FMA_UE"],
                "BI": ORIG_PATIENT_LABELS[sid]["BI"],
                "hand_tone": ORIG_PATIENT_LABELS[sid]["hand_tone"],
                "hand_function": ORIG_PATIENT_LABELS[sid]["hand_function"],
            },
            "source": SOURCE_REAL,
        }
    for subj in NEW_SUBJECTS:
        sid = int(subj["id"])
        labels = {
            "FMA_UE": subj["FMA_UE"],
            "BI": subj["BI"],
            "hand_tone": subj["hand_tone"],
            "hand_function": subj["hand_function"],
        }
        out[str(sid)] = {
            "rehab_text": gen_rehab_text(sid, subj["demographics"], labels),
            "demographics": subj["demographics"],
            "labels": labels,
            "source": SOURCE_SYNTHETIC,
        }

    path = root / f"patient_rehab_suggestions_{TARGET_N_SUBJECTS}subjects.json"
    path.write_text(json.dumps(
        {"subjects": out, "n_subjects": len(out)},
        ensure_ascii=False, indent=2,
    ))
    n_real = sum(1 for v in out.values() if v["source"] == SOURCE_REAL)
    n_syn = sum(1 for v in out.values() if v["source"] == SOURCE_SYNTHETIC)
    print(f"  写入康复建议: {path.name} ({len(out)} 名，real={n_real}/synthetic={n_syn})")
    return path


# ─── Manifest 构建 ────────────────────────────────────────────────────────────
def _subject_id_as_int(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype("Int64")
    )


def _attach_real_labels(orig: pd.DataFrame) -> pd.DataFrame:
    """Add the 4-task labels and source='real' to the real-patient manifest rows.

    Rewrites real EEG paths to the on-disk ``.bdf`` files, tags rows with
    ``eeg_format='bdf'``, drops manifest rows whose BDF/EMG file isn't actually
    present (e.g., excluded trials like S1_5 / never-recorded ones like S2_3).
    Does NOT write any ``wrist_tone`` column.
    """
    out = orig.copy()
    sid_int = _subject_id_as_int(out["subject_id"])

    for col in ("fma_ue", "bi", "hand_tone", "hand_function"):
        if col not in out.columns:
            out[col] = None
    # Drop wrist_tone column if it exists in the legacy manifest.
    if "wrist_tone" in out.columns:
        out = out.drop(columns=["wrist_tone"])
    out["source"] = SOURCE_REAL
    if "eeg_format" not in out.columns:
        out["eeg_format"] = SOURCE_REAL  # placeholder; overwritten below

    out["eeg_path"] = out["eeg_path"].astype(str).str.replace(
        r"\.csv$", ".bdf", regex=True
    )
    out["eeg_format"] = "bdf"

    for sid, labels in ORIG_PATIENT_LABELS.items():
        mask = sid_int == sid
        if not mask.any():
            print(f"  [警告] 原始 manifest 中没有找到 S{sid}，无法覆盖其标签。")
            continue
        out.loc[mask, "fma_ue"] = labels["FMA_UE"]
        out.loc[mask, "bi"] = labels["BI"]
        out.loc[mask, "hand_tone"] = labels["hand_tone"]
        out.loc[mask, "hand_function"] = labels["hand_function"]
        if "fma_score" in out.columns:
            out.loc[mask, "fma_score"] = labels["FMA_UE"]
        if "notes" in out.columns:
            notes = out.loc[mask, "notes"].fillna("").astype(str)
            out.loc[mask, "notes"] = notes.apply(
                lambda x, f=labels["FMA_UE"]: (x + "; " if x else "") + f"real_4tasks_fma={f}"
            )

    def _exists(rel: str) -> bool:
        return (ROOT / rel).exists()

    keep_mask = (
        out["eeg_path"].astype(str).map(_exists)
        & out["emg_path"].astype(str).map(_exists)
    )
    dropped = out.loc[~keep_mask, ["subject_id", "trial_id", "eeg_path"]]
    if not dropped.empty:
        print(f"  [真实数据] 丢弃 {len(dropped)} 条磁盘上不存在的 trial：")
        for _, row in dropped.iterrows():
            print(f"    - S{row['subject_id']} {row['trial_id']} ({row['eeg_path']})")
    out = out.loc[keep_mask].reset_index(drop=True)

    return out


def _synthesize_real_manifest_rows() -> pd.DataFrame:
    """Build manifest rows for the 5 real patients by scanning BJH/EEG_new/*.bdf.

    The legacy ``samples_manifest_tri.csv`` only covers S1..S3 with CSV paths.
    Since the user added S4 and S5 as real BDF data, we generate fresh rows for
    all real BDFs found on disk (rather than relying on the legacy CSV).
    """
    eeg_dir = ROOT / "BJH" / "EEG_new"
    emg_dir = ROOT / "BJH" / "EMG_new"
    rows: list[dict] = []
    for sid, labels in ORIG_PATIENT_LABELS.items():
        bdfs = sorted(eeg_dir.glob(f"S{sid}_*.bdf"))
        for bdf in bdfs:
            stem = bdf.stem  # e.g. S2_4_1
            parts = stem.split("_")
            if len(parts) != 3:
                continue
            _, task_id_s, trial_n_s = parts
            try:
                task_id = int(task_id_s)
                trial_number = int(trial_n_s)
            except ValueError:
                continue
            emg = emg_dir / f"{stem}.csv"
            if not emg.exists():
                print(f"  [真实数据] 跳过 {stem}: 缺 EMG CSV")
                continue
            rows.append({
                "subject_id": sid,
                "trial_id": f"{task_id}_{trial_number}",
                "task_id": task_id,
                "trial_number": trial_number,
                "eeg_path": f"BJH/EEG_new/{bdf.name}",
                "emg_path": f"BJH/EMG_new/{emg.name}",
                "fma_score": labels["FMA_UE"],
                "fma_ue": labels["FMA_UE"],
                "bi": labels["BI"],
                "hand_tone": labels["hand_tone"],
                "hand_function": labels["hand_function"],
                "mm_score": "",
                "eeg_channels": 32,
                "emg_muscles": 4,
                "imu_axes": 24,
                "match_strategy": "real_5subjects",
                "notes": f"real_4tasks_fma={labels['FMA_UE']}",
                "source": SOURCE_REAL,
                "eeg_format": "bdf",
            })
    return pd.DataFrame(rows)


def build_manifest(root: Path) -> Path:
    """Combine real S1..S5 manifest rows (scanned from disk) with synthetic 145."""
    orig = _synthesize_real_manifest_rows()

    rows: list[dict] = []
    for subj in NEW_SUBJECTS:
        sid = subj["id"]
        n_trials = int(subj["n_trials"])
        side = str(subj["demographics"]["affected_side"])
        for ti in range(n_trials):
            task_id, trial_number = task_and_trial_number(ti)
            fname = f"S{sid}_{task_id}_{trial_number}.csv"
            rows.append({
                "subject_id": sid,
                "trial_id": f"{task_id}_{trial_number}",
                "task_id": task_id,
                "trial_number": trial_number,
                "eeg_path": f"BJH/EEG_new/{fname}",
                "emg_path": f"BJH/EMG_new/{fname}",
                "fma_score": subj["FMA_UE"],
                "fma_ue": subj["FMA_UE"],
                "bi": subj["BI"],
                "hand_tone": subj["hand_tone"],
                "hand_function": subj["hand_function"],
                "mm_score": "",
                "eeg_channels": 32,
                "emg_muscles": 4,
                "imu_axes": 24,
                "match_strategy": "synthetic_4tasks",
                "notes": f"sim_4tasks_fma={subj['FMA_UE']};side={side}",
                "source": SOURCE_SYNTHETIC,
                "eeg_format": "csv",
            })

    combined = pd.concat([orig, pd.DataFrame(rows)], ignore_index=True)
    out = root / f"samples_manifest_tri_4tasks_{TARGET_N_SUBJECTS}subjects.csv"
    combined.to_csv(out, index=False)

    print(f"  Manifest：{len(orig)} 真实行 + {len(rows)} 合成行 = {len(combined)} 行  ({out.name})")
    n_real = combined["source"].eq(SOURCE_REAL).sum()
    n_syn = combined["source"].eq(SOURCE_SYNTHETIC).sum()
    print(f"  source: real={n_real} 行 / synthetic={n_syn} 行")
    df_sum = (
        combined.groupby("subject_id", sort=False)
        .agg(FMA_UE=("fma_ue", "first"),
             BI=("bi", "first"),
             hand_tone=("hand_tone", "first"),
             hand_function=("hand_function", "first"),
             trials=("fma_ue", "count"),
             source=("source", "first"))
    )
    print(df_sum.to_string())
    return out


def write_subject_labels_json(root: Path) -> Path:
    """Write a flat JSON of all 150 patients' 4-task labels for predict.py / eval."""
    out: Dict[str, Dict[str, object]] = {}
    for sid, labels in ORIG_PATIENT_LABELS.items():
        out[str(sid)] = {**labels, "source": SOURCE_REAL}
    for subj in NEW_SUBJECTS:
        out[str(subj["id"])] = {
            "FMA_UE": subj["FMA_UE"],
            "BI": subj["BI"],
            "hand_tone": subj["hand_tone"],
            "hand_function": subj["hand_function"],
            "source": SOURCE_SYNTHETIC,
        }

    # Validate everything against task_config so we never let "1+" become a number.
    enc_h = get_encoder("hand_tone")
    enc_f = get_encoder("hand_function")
    for sid, lab in out.items():
        assert FMA_RANGE[0] <= int(lab["FMA_UE"]) <= FMA_RANGE[1], sid
        assert BI_RANGE[0] <= int(lab["BI"]) <= BI_RANGE[1], sid
        enc_h.encode(lab["hand_tone"])
        enc_f.encode(lab["hand_function"])

    path = root / f"patient_labels_4tasks_{TARGET_N_SUBJECTS}subjects.json"
    path.write_text(json.dumps(
        {"subjects": out, "n_subjects": len(out)},
        ensure_ascii=False, indent=2,
    ))
    print(f"  写入患者标签: {path.name} ({len(out)} 名患者)")
    return path


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="BJH 4 临床任务 + 康复建议 数据扩充（慢性期）。",
    )
    ap.add_argument(
        "--target-n",
        type=int,
        default=TARGET_N_SUBJECTS,
        help=(f"Total subjects = real ({len(ORIG_PATIENT_LABELS)}) + synthetic. "
              f"Default {TARGET_N_SUBJECTS} (full run). Use a smaller value "
              "(e.g. 15) for a quick verification run."),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global TARGET_N_SUBJECTS, NEW_SUBJECTS
    args = _parse_args(argv)
    if args.target_n != TARGET_N_SUBJECTS:
        TARGET_N_SUBJECTS = int(args.target_n)
        NEW_SUBJECTS = make_new_subjects(target_n=TARGET_N_SUBJECTS)

    print("=" * 60)
    print(f"BJH 4 临床任务 + 康复建议 数据扩充（慢性期，{TARGET_N_SUBJECTS} 例）")
    print("=" * 60)
    print(f"项目根目录：{ROOT}")
    real_info = ", ".join(
        f"S{sid}(FMA={lab['FMA_UE']},BI={lab['BI']},"
        f"手={lab['hand_tone']},功能={lab['hand_function']})"
        for sid, lab in ORIG_PATIENT_LABELS.items()
    )
    print(f"真实患者：  {real_info}")
    first_new = len(ORIG_PATIENT_LABELS) + 1
    print(f"模拟新增：  S{first_new}~S{TARGET_N_SUBJECTS}，共 {len(NEW_SUBJECTS)} 名 (synthetic)")
    n_sim_trials = sum(s["n_trials"] for s in NEW_SUBJECTS)
    print(f"合成 trial 数：{n_sim_trials}（真实 trial 数将在 manifest 构建时统计）")
    print()

    print("[1/4] 生成合成信号（EEG + EMG/IMU CSV 文件）...")
    print(f"      信号长度：{DURATION}s | EEG {N_EEG}行 | EMG {N_EMG}行 | IMU {N_IMU}行")
    generate_data(ROOT, seed=GLOBAL_SEED)

    n_real = len(ORIG_PATIENT_LABELS)
    print(f"\n[2/4] 构建 Manifest（{n_real} 真实 BDF + {len(NEW_SUBJECTS)} 合成 CSV）...")
    manifest_path = build_manifest(ROOT)

    print(f"\n[3/4] 导出 patient_labels_4tasks_{TARGET_N_SUBJECTS}subjects.json ...")
    write_subject_labels_json(ROOT)

    print(f"\n[4/4] 导出 patient_rehab_suggestions_{TARGET_N_SUBJECTS}subjects.json ...")
    write_rehab_suggestions_json(ROOT)

    print("\n完成。")
    print("⚠️  合成数据仅用于程序流程测试，不能作为临床有效性证据。")
    print(f"⚠️  Manifest: {manifest_path.name}")
    suf = f"4tasks_{TARGET_N_SUBJECTS}subjects"
    print(f"下一步：python src/patient_splits.py --manifest samples_manifest_tri_{suf}.csv "
          f"--output splits/3fold_patient_split_tri_{suf}.json --force")
    print("下一步：python src/train.py --task FMA_UE  (其他任务同理)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
