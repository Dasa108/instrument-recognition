"""Phase 2 dataset — IRMAS's official Testing set, used as intended: multi-label detection.

See DECISIONS.md, "Phase 2 dataset: IRMAS Testing set" entry for why this dataset (not
OpenMIC-2018/Slakh2100, spec.md's originally-scoped alternatives) — same 11-class taxonomy as
Phase 1, already downloaded, real recordings.

Verified directly against the real files (not assumed):
- 2,874 clips, 5-20s each, 11-class taxonomy (same as Phase 1's IRMAS_CLASSES).
- One label file per clip (`<name>.txt` next to `<name>.wav`), one 3-letter instrument code per
  line (e.g. "gel\\tab\\n"), 1+ lines per file — this IS the multi-label ground truth.
- Clips are excerpts of larger songs, same numbered-suffix convention as Training data
  (e.g. "...I_M_NOT_IN_LOVE-11.wav", "...-13.wav", "...-15.wav" are different excerpts of the same
  song) — song-grouped splitting is just as necessary here as it was for Phase 1's build_split(),
  to avoid the same recording appearing in both train and eval.
- Directory structure is inconsistent across the 3 downloaded parts (Part1/Part3 nest under
  "Part1"/"Part3", Part2 nests under "IRTestingData-Part2") — handled with a recursive glob
  instead of hardcoding per-part folder names.

Because "the annotated instruments are the same in the whole excerpt" (IRMAS's own README), every
3s window of a given clip shares that clip's label vector — so each window becomes its own
training example (not one example per whole 5-20s clip), which also means the effective dataset
size is larger than 2,874: longer clips contribute more windows.
"""

import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.datasets.irmas_dataset import IRMAS_CLASSES
from src.preprocessing.audio_to_logmel import CLIP_SECONDS, SAMPLE_RATE, load_audio, to_logmel

TESTING_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "IRMAS-TestingData"

_LABEL_INDEX = {code: i for i, code in enumerate(IRMAS_CLASSES)}
_SONG_TITLE_RE = re.compile(r"^(.*)-\d+$")  # strip the trailing "-<excerpt number>"


def parse_labels(txt_path: Path) -> list[str]:
    """One 3-letter instrument code per line (whitespace-padded) -> list of codes."""
    codes = [line.strip() for line in txt_path.read_text().splitlines()]
    return [c for c in codes if c]


def _song_title(wav_path: Path) -> str:
    stem = wav_path.stem  # filename without ".wav"
    m = _SONG_TITLE_RE.match(stem)
    return m.group(1) if m else stem  # fallback: treat as its own singleton group


def _multi_hot(codes: list[str]) -> list[int]:
    vec = [0] * len(IRMAS_CLASSES)
    for c in codes:
        if c in _LABEL_INDEX:
            vec[_LABEL_INDEX[c]] = 1
    return vec


def build_multilabel_split(
    root: Path = TESTING_ROOT,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[Path, list[int]]]]:
    """Song-grouped split, same leakage-safety logic as Phase 1's build_split() — but assigned by
    overall clip-count proportion rather than per-class (a clip can count toward several classes
    at once in a multi-label setting, so Phase 1's per-class-balanced assignment doesn't directly
    generalize; not attempted here, see DECISIONS.md)."""
    import random

    wav_paths = sorted(root.rglob("*.wav"))
    groups: dict[str, list[tuple[Path, list[int]]]] = {}
    for wav_path in wav_paths:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue  # shouldn't happen — every real clip has a label file
        labels = _multi_hot(parse_labels(txt_path))
        groups.setdefault(_song_title(wav_path), []).append((wav_path, labels))

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    n_total = len(wav_paths)
    n_val_target = round(n_total * val_frac)
    n_test_target = round(n_total * test_frac)

    splits: dict[str, list[tuple[Path, list[int]]]] = {"train": [], "val": [], "test": []}
    val_count = test_count = 0
    for key in group_keys:
        clips = groups[key]
        if val_count < n_val_target:
            splits["val"].extend(clips)
            val_count += len(clips)
        elif test_count < n_test_target:
            splits["test"].extend(clips)
            test_count += len(clips)
        else:
            splits["train"].extend(clips)

    return splits


class IRMASMultilabelDataset(Dataset):
    """Yields (log-mel window, multi-hot label vector) pairs. One entry per 3s window of every
    clip in the split — see module docstring for why."""

    def __init__(
        self,
        split: str = "train",
        root: Path = TESTING_ROOT,
        seed: int = 42,
        clip_seconds: float = CLIP_SECONDS,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.clip_seconds = clip_seconds
        clips = build_multilabel_split(root=root, seed=seed)[split]

        # Expand each clip into (path, window_index, label) — one example per 3s window.
        # sf.info() reads only the file header (fast), not the audio itself.
        self.samples: list[tuple[Path, int, list[int]]] = []
        for path, labels in clips:
            duration = sf.info(str(path)).duration
            n_windows = max(1, math.ceil(duration / clip_seconds))
            for w in range(n_windows):
                self.samples.append((path, w, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, window_idx, labels = self.samples[idx]
        offset = window_idx * self.clip_seconds
        waveform = load_audio(str(path), sample_rate=SAMPLE_RATE, offset=offset,
                               duration=self.clip_seconds)

        target_len = int(round(self.clip_seconds * SAMPLE_RATE))
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]

        logmel = to_logmel(waveform)
        tensor = torch.from_numpy(np.ascontiguousarray(logmel)).unsqueeze(0)  # (1, 128, 301)
        return tensor, torch.tensor(labels, dtype=torch.float32)
