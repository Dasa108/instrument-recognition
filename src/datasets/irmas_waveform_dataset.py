"""Raw-waveform variant of IRMASDataset, for pretrained models (PANNs, AST) that compute their own
spectrogram features internally rather than consuming this project's audio_to_logmel.py output.

Reuses build_split() from irmas_dataset.py (same song-grouped split, same seed=42 default), so
Phase B results stay directly comparable to Runs 1-6 — same songs held out, same test set.

See notes/improv_cnn.md "Phase B" section for why raw waveform is needed here: PANNs' Cnn14
computes its own internal log-mel (32kHz, 64 mel bins) and AST's own feature extractor computes
its own fbank (16kHz, 128 mel bins, its own padding) — feeding either this project's existing
128-bin/16kHz log-mel output would be feeding the wrong representation entirely, not just a
slightly different one.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.datasets.irmas_dataset import TRAIN_DATA_ROOT, build_split
from src.preprocessing.audio_to_logmel import load_audio, window


class IRMASWaveformDataset(Dataset):
    """Yields (waveform, label) pairs at a caller-specified sample rate, instead of log-mel.

    IRMAS training clips are ~3s each; `window()` is reused purely for its exact-length
    trim/pad-to-one-window behavior (silence-pads the rare clip that's fractionally short), not
    for its multi-window splitting (IRMAS training clips are always exactly one window).
    """

    def __init__(
        self,
        split: str = "train",
        root: Path = TRAIN_DATA_ROOT,
        seed: int = 42,
        sample_rate: int = 32000,
        clip_seconds: float = 3.0,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.samples = build_split(root=root, seed=seed)[split]
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        waveform = load_audio(str(path), sample_rate=self.sample_rate)
        windows = window(waveform, clip_seconds=self.clip_seconds, sample_rate=self.sample_rate)
        return torch.from_numpy(windows[0]), label  # (num_samples,), int
