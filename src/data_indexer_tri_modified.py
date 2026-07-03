"""Tri-modal samples manifest builder for the BJH dataset.

Pairs EEG csv files (under BJH/EEG_new/) with the matching EMG/IMU csv files
(under BJH/EMG_new/) by trial key (Sx_t_n), validates schema by column name
(not by column count), and joins external FMA labels.

Output schema (CSV):
    subject_id, trial_id, task_id, trial_number,
    eeg_path, emg_path,
    fma_score, mm_score,
    eeg_channels, emg_muscles, imu_axes,
    match_strategy, notes,
    fma_ue, bi, hand_tone, hand_function, source

This is intentionally separate from the original data_indexer.py so the existing
MHH baseline pipeline keeps working unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from bjh_io.bjh_loader import EEG_CHANNELS, EMG_MUSCLES, IMU_AXES_PER_MUSCLE


DEFAULT_MANIFEST_NAME = "samples_manifest_tri-try.csv"
DEFAULT_LABELS_NAME = "bjh_labels.json"

EEG_DIR_TOKENS = {"eeg", "eegnew", "eeg_new"}
EMG_DIR_TOKENS = {"emg", "emgnew", "emg_new"}

TRIAL_RE = re.compile(r"^(?:s)?0*(\d+)[_\-\s]+0*(\d+)[_\-\s]+0*(\d+)$", re.IGNORECASE)

MANIFEST_FIELDS: Tuple[str, ...] = (
    "subject_id",
    "trial_id",
    "task_id",
    "trial_number",
    "eeg_path",
    "emg_path",
    "fma_score",
    "mm_score",
    "eeg_channels",
    "emg_muscles",
    "imu_axes",
    "match_strategy",
    "notes",
    "fma_ue",
    "bi",
    "hand_tone",
    "hand_function",
    "source",
)


# --------------------------------------------------------------------------- #
# Trial key parsing                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, order=True)
class TrialKey:
    subject_id: str
    task_id: str
    trial_number: str

    @property
    def trial_id(self) -> str:
        return f"{self.task_id}_{self.trial_number}"

    @property
    def label(self) -> str:
        return f"S{self.subject_id}_{self.task_id}_{self.trial_number}"


def _numeric_id(value: str) -> str:
    s = value.strip()
    if not s.isdigit():
        raise ValueError(f"Expected a numeric id, got {value!r}")
    return str(int(s))


def parse_trial_key(path: Path) -> Optional[TrialKey]:
    stem = path.stem.strip()
    if "static" in stem.lower():
        return None
    m = TRIAL_RE.match(stem)
    if not m:
        return None
    sid, tid, n = (_numeric_id(p) for p in m.groups())
    return TrialKey(subject_id=sid, task_id=tid, trial_number=n)


def _clean_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _path_has_token(path: Path, tokens: Iterable[str]) -> bool:
    cleaned = {_clean_token(t) for t in tokens}
    for part in path.parts:
        cleaned_part = _clean_token(part)
        if any(t in cleaned_part for t in cleaned):
            return True
    return False


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #
def _has_csv_files(directory: Path) -> bool:
    return any(directory.glob("*.csv"))


def locate_signal_dirs(root: Path) -> Tuple[List[Path], List[Path]]:
    """Find local BJH EEG and EMG/IMU directories under the project root.

    Prefers explicit BJH subtrees: if a `BJH/` directory exists with EEG_new
    and/or EMG_new subdirectories, those take precedence. This avoids the
    legacy MHH EMG/kinematic trees being mis-classified as BJH data.
    """
    root = root.resolve()
    bjh_root = root / "BJH"
    if bjh_root.is_dir():
        eeg_pref = [bjh_root / "EEG_new"] if (bjh_root / "EEG_new").is_dir() else []
        emg_pref = [bjh_root / "EMG_new"] if (bjh_root / "EMG_new").is_dir() else []
        if eeg_pref and emg_pref:
            return eeg_pref, emg_pref

    eeg_dirs: List[Path] = []
    emg_dirs: List[Path] = []
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: str(p)):
        if not _has_csv_files(directory):
            continue
        # Skip the legacy MHH tree to avoid name collisions with kinematic/EMG.
        if _path_has_token(directory, {"mhh"}):
            continue
        if _path_has_token(directory, EEG_DIR_TOKENS):
            eeg_dirs.append(directory)
        if _path_has_token(directory, EMG_DIR_TOKENS):
            emg_dirs.append(directory)
    # Strip overlap: a directory containing EMG_new doesn't also contain EEG_new.
    eeg_dirs = [d for d in eeg_dirs if not _path_has_token(d, EMG_DIR_TOKENS)]
    emg_dirs = [d for d in emg_dirs if not _path_has_token(d, EEG_DIR_TOKENS)]
    return eeg_dirs, emg_dirs


def _collect_csv_files(directories: Sequence[Path]) -> List[Path]:
    files: Dict[str, Path] = {}
    for d in directories:
        for fp in d.glob("*.csv"):
            files[str(fp.resolve())] = fp.resolve()
    return [files[k] for k in sorted(files)]


def _group_by_trial(files: Sequence[Path]) -> Tuple[Dict[TrialKey, List[Path]], List[Path]]:
    grouped: Dict[TrialKey, List[Path]] = {}
    skipped: List[Path] = []
    for f in files:
        k = parse_trial_key(f)
        if k is None:
            skipped.append(f)
            continue
        grouped.setdefault(k, []).append(f)
    return grouped, skipped


# --------------------------------------------------------------------------- #
# CSV inspection by column NAME (not index)                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EEGInspection:
    ok: bool
    channels: int
    missing: Tuple[str, ...]
    message: str


@dataclass(frozen=True)
class EMGInspection:
    ok: bool
    muscles: int
    imu_axes: int
    missing_emg: Tuple[str, ...]
    missing_imu: Tuple[str, ...]
    message: str


def inspect_eeg_csv(path: Path, expected: Sequence[str] = EEG_CHANNELS) -> EEGInspection:
    try:
        head = pd.read_csv(path, nrows=1)
    except Exception as exc:  # noqa: BLE001
        return EEGInspection(False, 0, tuple(expected), f"read_error:{exc}")
    cols = set(head.columns)
    missing = tuple(c for c in expected if c not in cols)
    if missing:
        return EEGInspection(False, len(head.columns), missing, f"eeg_missing:{','.join(missing[:3])}")
    return EEGInspection(True, len(expected), tuple(), "ok")


def inspect_emg_csv(
    path: Path,
    muscles: Sequence[str] = EMG_MUSCLES,
    imu_axes: Sequence[str] = IMU_AXES_PER_MUSCLE,
) -> EMGInspection:
    try:
        head = pd.read_csv(path, nrows=1)
    except Exception as exc:  # noqa: BLE001
        return EMGInspection(False, 0, 0, tuple(muscles), tuple(), f"read_error:{exc}")
    cols = list(head.columns)
    missing_emg = tuple(m for m in muscles if not any(m in c and ": EMG" in c for c in cols))
    missing_imu: List[str] = []
    for m in muscles:
        for ax in imu_axes:
            if not any(m in c and f": {ax}" in c for c in cols):
                missing_imu.append(f"{m}:{ax}")
    if missing_emg or missing_imu:
        return EMGInspection(
            False,
            len(muscles) - len(missing_emg),
            len(muscles) * len(imu_axes) - len(missing_imu),
            missing_emg,
            tuple(missing_imu),
            f"missing_emg={len(missing_emg)},missing_imu={len(missing_imu)}",
        )
    return EMGInspection(True, len(muscles), len(muscles) * len(imu_axes), tuple(), tuple(), "ok")


# --------------------------------------------------------------------------- #
# Labels                                                                      #
# --------------------------------------------------------------------------- #
def load_labels(path: Path) -> Dict[str, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("subjects", payload)
    labels: Dict[str, Dict[str, object]] = {}
    for sid, info in raw.items():
        if not str(sid).isdigit():
            continue
        if not isinstance(info, dict):
            continue
        labels[str(int(sid))] = dict(info)
    if not labels:
        raise ValueError(f"No valid subject labels in {path}")
    return labels


LABEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    # Keep fma_score compatible with the old FMA_UE label, while also emitting
    # the explicit fma_ue column requested by the downstream table schema.
    "fma_ue": (
        "fma_ue",
        "FMA_UE",
        "FMA UE",
        "FMA-UE",
        "FMAUE",
        "fma_score",
        "FMA_score",
        "FMA",
    ),
    "mm_score": ("mm_score", "MM_score", "MM Score", "MM", "mm"),
    "bi": ("bi", "BI", "Barthel Index", "Barthel_Index", "barthel_index", "Barthel"),
    "hand_tone": (
        "hand_tone",
        "Hand_tone",
        "Hand Tone",
        "hand tone",
        "hand_tonus",
        "hand_tension",
    ),
    "hand_function": (
        "hand_function",
        "Hand_function",
        "Hand Function",
        "hand function",
        "hand_func",
        "handfunction",
    ),
    "source": ("source", "Source", "data_source", "label_source", "clinical_source"),
}


def _normalize_label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _label_value(label: Dict[str, object], field: str) -> object:
    """Return a clinical label value using case/spacing tolerant aliases.

    The BJH label file may come from spreadsheets or hand-built JSON, so column
    names are not always identical. This helper first checks exact aliases and
    then retries using a normalized key, allowing values such as `FMA_UE`,
    `FMA-UE`, `FMA UE`, or `fma_ue` to populate the same output column.
    """
    aliases = LABEL_ALIASES[field]

    for key in aliases:
        value = label.get(key)
        if value is not None:
            return value

    normalized = {_normalize_label_key(str(k)): v for k, v in label.items()}
    for key in aliases:
        value = normalized.get(_normalize_label_key(key))
        if value is not None:
            return value

    return ""


# --------------------------------------------------------------------------- #
# Manifest assembly                                                           #
# --------------------------------------------------------------------------- #
def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sorted_keys(keys: Iterable[TrialKey]) -> List[TrialKey]:
    return sorted(keys, key=lambda k: (int(k.subject_id), int(k.task_id), int(k.trial_number)))


def build_tri_manifest(
    root: Path | str = ".",
    output: Path | str = DEFAULT_MANIFEST_NAME,
    labels_path: Path | str = DEFAULT_LABELS_NAME,
    eeg_dirs: Optional[Sequence[Path | str]] = None,
    emg_dirs: Optional[Sequence[Path | str]] = None,
    strict: bool = False,
) -> Dict[str, object]:
    root_p = Path(root).resolve()
    out_p = Path(output)
    if not out_p.is_absolute():
        out_p = root_p / out_p
    labels_p = Path(labels_path)
    if not labels_p.is_absolute():
        labels_p = root_p / labels_p

    if eeg_dirs is None or emg_dirs is None:
        located_eeg, located_emg = locate_signal_dirs(root_p)
        eeg_paths = located_eeg if eeg_dirs is None else [Path(p).resolve() for p in eeg_dirs]
        emg_paths = located_emg if emg_dirs is None else [Path(p).resolve() for p in emg_dirs]
    else:
        eeg_paths = [Path(p).resolve() for p in eeg_dirs]
        emg_paths = [Path(p).resolve() for p in emg_dirs]

    if not eeg_paths:
        raise FileNotFoundError("No BJH EEG csv directory found.")
    if not emg_paths:
        raise FileNotFoundError("No BJH EMG/IMU csv directory found.")

    labels = load_labels(labels_p)

    eeg_files = _collect_csv_files(eeg_paths)
    emg_files = _collect_csv_files(emg_paths)
    eeg_groups, skipped_eeg = _group_by_trial(eeg_files)
    emg_groups, skipped_emg = _group_by_trial(emg_files)

    eeg_dups = {k: v for k, v in eeg_groups.items() if len(v) > 1}
    emg_dups = {k: v for k, v in emg_groups.items() if len(v) > 1}
    duplicate_keys = set(eeg_dups) | set(emg_dups)

    rows: List[Dict[str, object]] = []
    invalid: List[str] = []
    missing_eeg: List[str] = []
    missing_emg: List[str] = []
    match_counter: Counter[str] = Counter()

    for key in _sorted_keys(set(eeg_groups) | set(emg_groups)):
        if key in duplicate_keys:
            invalid.append(f"{key.label}:duplicate_key")
            continue

        eeg_paths_for_key = eeg_groups.get(key, [])
        emg_paths_for_key = emg_groups.get(key, [])
        if not eeg_paths_for_key:
            missing_eeg.append(key.label)
            continue
        if not emg_paths_for_key:
            missing_emg.append(key.label)
            continue

        eeg_path = eeg_paths_for_key[0]
        emg_path = emg_paths_for_key[0]

        eeg_info = inspect_eeg_csv(eeg_path)
        emg_info = inspect_emg_csv(emg_path)
        label = labels.get(key.subject_id)

        notes: List[str] = []
        if label is None:
            notes.append("missing_patient_label")
        if not eeg_info.ok:
            notes.append(eeg_info.message)
        if not emg_info.ok:
            notes.append(emg_info.message)

        if label is None or not eeg_info.ok or not emg_info.ok:
            invalid.append(f"{key.label}:{'|'.join(notes)}")
            continue

        match_strategy = "exact_filename" if eeg_path.name == emg_path.name else "parsed_trial_key"
        match_counter[match_strategy] += 1

        rows.append(
            {
                "subject_id": key.subject_id,
                "trial_id": key.trial_id,
                "task_id": key.task_id,
                "trial_number": key.trial_number,
                "eeg_path": _relative_path(eeg_path, root_p),
                "emg_path": _relative_path(emg_path, root_p),
                "fma_score": _label_value(label, "fma_ue"),
                "mm_score": _label_value(label, "mm_score"),
                "eeg_channels": eeg_info.channels,
                "emg_muscles": emg_info.muscles,
                "imu_axes": emg_info.imu_axes,
                "match_strategy": match_strategy,
                "notes": ";".join(notes),
                "fma_ue": _label_value(label, "fma_ue"),
                "bi": _label_value(label, "bi"),
                "hand_tone": _label_value(label, "hand_tone"),
                "hand_function": _label_value(label, "hand_function"),
                "source": _label_value(label, "source") or labels_p.name,
            }
        )

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
        w.writeheader()
        w.writerows(rows)

    data_subjects = sorted({r["subject_id"] for r in rows}, key=lambda v: int(v))
    label_subjects = set(labels.keys())
    stats = {
        "root": str(root_p),
        "manifest_path": str(out_p),
        "labels_path": str(labels_p),
        "eeg_dirs": [_relative_path(p, root_p) for p in eeg_paths],
        "emg_dirs": [_relative_path(p, root_p) for p in emg_paths],
        "eeg_files": len(eeg_files),
        "emg_files": len(emg_files),
        "matched_samples": len(rows),
        "subjects": data_subjects,
        "skipped_eeg": [_relative_path(p, root_p) for p in skipped_eeg[:8]],
        "skipped_emg": [_relative_path(p, root_p) for p in skipped_emg[:8]],
        "missing_eeg_examples": missing_eeg[:12],
        "missing_emg_examples": missing_emg[:12],
        "invalid_examples": invalid[:12],
        "labels_missing_for_data": sorted(set(data_subjects) - label_subjects, key=lambda v: int(v)),
        "labels_without_samples": sorted(label_subjects - set(data_subjects), key=lambda v: int(v)),
        "match_strategy_counts": dict(match_counter),
        "duplicate_keys": len(duplicate_keys),
    }

    if strict:
        errors = []
        if stats["matched_samples"] == 0:
            errors.append("matched_samples=0")
        if stats["duplicate_keys"]:
            errors.append(f"duplicate_keys={stats['duplicate_keys']}")
        if stats["labels_missing_for_data"]:
            errors.append(f"labels_missing_for_data={stats['labels_missing_for_data']}")
        if errors:
            raise RuntimeError("Tri-modal manifest integrity check failed: " + ", ".join(errors))

    return stats


def print_stats(stats: Dict[str, object]) -> None:
    print("\nTri-modal samples manifest built")
    print(f"  root: {stats['root']}")
    print(f"  manifest: {stats['manifest_path']}")
    print(f"  labels:   {stats['labels_path']}")
    print(f"  EEG dirs: {stats['eeg_dirs']}")
    print(f"  EMG dirs: {stats['emg_dirs']}")
    print(f"  raw files: EEG={stats['eeg_files']} EMG={stats['emg_files']}")
    print(f"  matched samples: {stats['matched_samples']}")
    print(f"  subjects ({len(stats['subjects'])}): {stats['subjects']}")
    print(f"  match strategies: {stats['match_strategy_counts']}")
    print(f"  duplicate keys: {stats['duplicate_keys']}")
    if stats["invalid_examples"]:
        print(f"  invalid: {stats['invalid_examples']}")
    if stats["missing_eeg_examples"]:
        print(f"  missing EEG: {stats['missing_eeg_examples']}")
    if stats["missing_emg_examples"]:
        print(f"  missing EMG: {stats['missing_emg_examples']}")
    if stats["labels_missing_for_data"]:
        print(f"  labels missing for data subjects: {stats['labels_missing_for_data']}")
    if stats["labels_without_samples"]:
        print(f"  label subjects without samples: {stats['labels_without_samples']}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the BJH tri-modal samples manifest.")
    p.add_argument("--root", default=".")
    p.add_argument("--output", default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--labels", default=DEFAULT_LABELS_NAME)
    p.add_argument("--eeg-dir", action="append", default=None)
    p.add_argument("--emg-dir", action="append", default=None)
    p.add_argument("--strict", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stats = build_tri_manifest(
        root=args.root,
        output=args.output,
        labels_path=args.labels,
        eeg_dirs=args.eeg_dir,
        emg_dirs=args.emg_dir,
        strict=args.strict,
    )
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
