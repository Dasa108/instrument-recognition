"""Audio -> log-mel spectrogram pipeline.

Spec: spec.md Section 4 (Preprocessing Pipeline):
    raw audio -> resample to 16 kHz mono -> fixed-length windows (3s, silence-pad leftovers)
        -> STFT -> mel filterbank (n_mels=128) -> log scale -> log-mel spectrogram (per window)

IRMAS Training clips are already fixed 3s single-label files (verified: 44.1kHz stereo, ~3.0000s,
6,705 files — see DECISIONS.md) so window() is a no-op safeguard for them (returns 1 window). It
does real work on longer/variable-length audio (IRMAS Testing clips, or real-world input later) —
splits into non-overlapping 3s windows and silence-pads the leftover tail, per the windowing
sidebar in notes/THEORY_NOTES.md.
"""

import random

import librosa
import numpy as np
import torch

SAMPLE_RATE = 16_000
CLIP_SECONDS = 3.0
N_MELS = 128
N_FFT = 400        # 25ms at 16kHz (0.025 * 16000)
HOP_LENGTH = 160    # 10ms at 16kHz (0.01 * 16000)


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file, resampled to `sample_rate` and downmixed to mono."""
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return waveform


def window(
    waveform: np.ndarray, clip_seconds: float = CLIP_SECONDS, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Split into non-overlapping fixed-length windows, silence-padding the leftover tail.

    Returns shape (num_windows, win_len). A waveform already exactly one window long (e.g. IRMAS
    Training clips) returns shape (1, win_len) unchanged (aside from exact-length trimming/padding
    to correct for float-duration rounding).
    """
    win_len = int(round(clip_seconds * sample_rate))
    if win_len <= 0:
        raise ValueError("clip_seconds * sample_rate must be positive")

    n = len(waveform)
    num_windows = max(1, -(-n // win_len))  # ceil division, at least one window
    total_len = num_windows * win_len
    if total_len > n:
        waveform = np.pad(waveform, (0, total_len - n))
    else:
        waveform = waveform[:total_len]
    return waveform.reshape(num_windows, win_len)


def to_logmel(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Single 1D window -> log-mel spectrogram, shape (n_mels, n_frames)."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window="hann",
    )
    return librosa.power_to_db(mel, ref=np.max)


def process_clip(path: str) -> np.ndarray:
    """Full pipeline: audio file -> stacked log-mel spectrograms, shape
    (num_windows, n_mels, n_frames)."""
    waveform = load_audio(path)
    windows = window(waveform)
    return np.stack([to_logmel(w) for w in windows])


def spec_augment(
    x: torch.Tensor,
    freq_mask_param: int = 16,
    time_mask_param: int = 30,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
) -> torch.Tensor:
    """SpecAugment (Park et al. 2019), time/frequency masking only (no time warping).

    Applied per-sample to a batch of log-mel spectrograms, shape (B, 1, n_mels, n_frames).
    Masked regions are filled with that sample's own mean (roughly "average energy", not silence
    — log-mel is a dB scale where 0 is loudest, not 0 energy, so zero-filling would read as an
    unrealistic loud artifact rather than a gap). See notes/improv_cnn.md, section 2.

    Training-time only — never call this on eval/val/test batches.
    """
    x = x.clone()
    _, _, n_mels, n_frames = x.shape
    for b in range(x.shape[0]):
        fill = x[b].mean()
        for _ in range(num_freq_masks):
            f = random.randint(0, min(freq_mask_param, n_mels))
            f0 = random.randint(0, n_mels - f)
            x[b, :, f0 : f0 + f, :] = fill
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, n_frames))
            t0 = random.randint(0, n_frames - t)
            x[b, :, :, t0 : t0 + t] = fill
    return x
