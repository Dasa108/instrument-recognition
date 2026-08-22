"""PyTorch Dataset for IRMAS — yields (log-mel spectrogram, instrument label) pairs.

Spec: spec.md Section 3 (Dataset) + Section 4 (Preprocessing Pipeline).

Split strategy: val/test are carved out of Training data, grouped by *song* (not by clip) so the
same source recording never appears in both train and eval. IRMAS training filenames encode this:
a numeric id shared across excerpts of the same recording, e.g. `[cel][cla]0207__1.wav` and
`[cel][cla]0207__3.wav` are both excerpt 1 and 3 of song 0207. Verified against the real download:
2,250 of 2,261 (class, song-id) groups have more than one clip (up to 6) — grouping by clip alone
would leak the same recording across splits. See DECISIONS.md, "Song-grouped IRMAS split" entry.
"""

import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.audio_to_logmel import process_clip

IRMAS_CLASSES = [
    "cel",  # cello
    "cla",  # clarinet
    "flu",  # flute
    "gac",  # acoustic guitar
    "gel",  # electric guitar
    "org",  # organ
    "pia",  # piano
    "sax",  # saxophone
    "tru",  # trumpet
    "vio",  # violin
    "voi",  # voice
]

TRAIN_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "IRMAS-TrainingData" / "IRMAS-TrainingData"
)

_SONG_ID_RE = re.compile(r"(\d+)__\d+\.wav$")


def _song_id(filename: str) -> str:
    """Extract the shared-recording id from an IRMAS filename; falls back to the filename
    itself (treated as its own singleton group) if the pattern doesn't match."""
    m = _SONG_ID_RE.search(filename)
    return m.group(1) if m else filename


def build_split(
    root: Path = TRAIN_DATA_ROOT,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[Path, int]]]:
    """Group clips by (class, song id), shuffle groups, and assign whole groups to
    train/val/test per class so proportions stay balanced across classes without ever
    splitting a song across sets."""
    rng = random.Random(seed)
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}

    for label, cls in enumerate(IRMAS_CLASSES):
        files = sorted((root / cls).glob("*.wav"))
        groups: dict[str, list[Path]] = {}
        for f in files:
            groups.setdefault(_song_id(f.name), []).append(f)

        group_ids = list(groups.keys())
        rng.shuffle(group_ids)

        n_total = len(files)
        n_val_target = round(n_total * val_frac)
        n_test_target = round(n_total * test_frac)

        val_count = test_count = 0
        for gid in group_ids:
            clips = [(f, label) for f in groups[gid]]
            if val_count < n_val_target:
                splits["val"].extend(clips)
                val_count += len(clips)
            elif test_count < n_test_target:
                splits["test"].extend(clips)
                test_count += len(clips)
            else:
                splits["train"].extend(clips)

    return splits


class IRMASDataset(Dataset):
    """Single-label predominant-instrument dataset (Phase 1). See spec.md Section 3."""

    def __init__(self, split: str = "train", root: Path = TRAIN_DATA_ROOT, seed: int = 42):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.samples = build_split(root=root, seed=seed)[split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        logmel = process_clip(str(path))[0]  # (n_mels, n_frames); IRMAS clips are 1 window
        tensor = torch.from_numpy(np.ascontiguousarray(logmel)).unsqueeze(0)  # (1, n_mels, n_frames)
        return tensor, label
