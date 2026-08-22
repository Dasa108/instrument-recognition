# Instrument Recognition — Project Spec (v1.0)

**Status:** Finalized (2026-08-21), fully self-paced (no fixed deadline). See `DECISIONS.md` for
the reasoning behind each choice below.
**Background:** theory covered in `notes/THEORY_NOTES.md` (7 modules: sound → sampling → FFT →
spectrograms → mel scale → timbre → ML bridging). This spec turns that theory into an execution
plan.

## 1. Objective

Build a model that identifies which musical instrument(s) are present in an audio clip.

## 2. Scope — phased

- **Phase 1 (MVP):** predominant-instrument recognition — one label per 3s clip, single-label
  classification.
- **Phase 2:** multi-label recognition on full songs — predict *all* instruments present, not
  just the dominant one.
- **Out of scope (for now):** instrument *separation* (isolating each instrument's audio), genre
  classification, real-time/streaming inference.

Datasets for each phase: see Section 3.

## 3. Dataset

| Phase | Dataset | Size | Labels | Notes |
|---|---|---|---|---|
| 1 | IRMAS | ~9k × 3s clips | 11 instruments, single-label (predominant) | Standard benchmark, download via Zenodo |
| 2 | OpenMIC-2018 | ~20k × 10s clips | 20 instruments, multi-label (partial) | Weak/partial supervision |
| 2 (alt) | Slakh2100 | 2100 full songs | multi-label, synthesized (MIDI-rendered) | Larger scale, not real recordings |

Train/val/test split: IRMAS's "Testing" files are *not* a single-label test set (multi-labeled,
variable-length — built for a different, detection-style task) and aren't used as Phase 1's held-out
set. Instead, val/test are carved out of the Training data itself, grouped by song, to avoid the
same song appearing in both train and eval. See `DECISIONS.md`, "IRMAS download scope" entry.

## 4. Preprocessing Pipeline

```
raw audio → resample to 16 kHz mono → fixed-length windows (3s, silence-pad leftovers)
   → STFT → mel filterbank (n_mels=128) → log scale → log-mel spectrogram (per window)
```

- Sample rate: 16 kHz (standard ML choice, sufficient for instrument energy range — see Mod 2).
- Window: 3s clips, hop matching IRMAS convention.
- STFT: ~25ms frame, ~10ms hop, Hann window (see Mod 4).
- Output shape: `128 mel bins × 301 time frames` per 3s clip — verified against real IRMAS audio
  (`src/preprocessing/audio_to_logmel.py`), not just calculated.

## 5. Model

- **Baseline:** small CNN from scratch (4–6 conv/pool blocks + dense head) on log-mel input.
- **Phase 1 upgrade path:** transfer learning via pretrained audio embeddings (PANNs / YAMNet /
  VGGish) + small classifier head, if baseline underperforms.
- **Phase 2:** same backbone, swap output head to per-class sigmoid + BCE loss (multi-label).

## 6. Evaluation

- Phase 1 (single-label): accuracy, per-class precision/recall/F1, confusion matrix.
- Phase 2 (multi-label): macro-F1 and micro-F1 (primary metrics), per-class precision/recall.
- Track a fixed validation set throughout; no test-set peeking until final report.

## 7. Tech Stack

- Python, PyTorch (model + training loop).
- `librosa` or `torchaudio` for audio I/O and mel-spectrogram extraction.
- `scikit-learn` for metrics.
- Experiment tracking: TensorBoard (`torch.utils.tensorboard`).
- Compute: local RTX 3050 (6GB VRAM, driver 595.84, CUDA 13.2) — confirmed working
  (`torch.cuda.is_available() == True`). Batch size 32–128, `torch.cuda.amp` mixed precision.
- Environment: conda env named `Sound` (Python 3.13) — see `README.md` for setup. All project
  commands (installs, preprocessing, training, evaluation) run inside it, not the system Python.
  Exact pinned package versions in `requirements.txt`.

## 8. Repo Structure (proposed)

```
instrument-recognition/
├── notes/                # theory notes (existing)
├── spec.md               # this file
├── data/                 # raw + processed datasets (gitignored)
├── src/
│   ├── datasets/          # download + Dataset/DataLoader classes (code, not data)
│   ├── preprocessing/     # audio → log-mel spectrogram
│   ├── models/             # CNN architectures
│   ├── train.py
│   └── evaluate.py
├── notebooks/             # exploration / visualization
└── configs/               # training configs (yaml)
```

## 9. Milestones

1. Repo scaffold + IRMAS download script.
2. Preprocessing pipeline (audio → log-mel spectrogram), verified visually on a few samples.
3. Baseline CNN training on IRMAS (Phase 1), report accuracy/F1.
4. Iterate: pretrained-embedding upgrade if baseline is weak.
5. Phase 2: multi-label pipeline on OpenMIC/Slakh2100.
