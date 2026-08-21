# Instrument Recognition — Project Spec (Draft v0.1)

**Status:** Draft for review — not yet finalized. Open questions flagged at the bottom.
**Background:** theory covered in `notes/THEORY_NOTES.md` (7 modules: sound → sampling → FFT →
spectrograms → mel scale → timbre → ML bridging). This spec turns that theory into an execution
plan.

## 1. Objective

Build a model that identifies which musical instrument(s) are present in an audio clip.

## 2. Scope — phased

- **Phase 1 (MVP):** predominant-instrument recognition — one label per 3s clip, single-label
  classification. Dataset: **IRMAS** (11 classes).
- **Phase 2:** multi-label recognition on full songs — predict *all* instruments present, not
  just the dominant one. Dataset: **OpenMIC-2018** or **Slakh2100**.
- **Out of scope (for now):** instrument *separation* (isolating each instrument's audio), genre
  classification, real-time/streaming inference.

## 3. Dataset

| Phase | Dataset | Size | Labels | Notes |
|---|---|---|---|---|
| 1 | IRMAS | ~9k × 3s clips | 11 instruments, single-label (predominant) | Standard benchmark, download via Zenodo |
| 2 | OpenMIC-2018 | ~20k × 10s clips | 20 instruments, multi-label (partial) | Weak/partial supervision |
| 2 (alt) | Slakh2100 | 2100 full songs | multi-label, synthesized (MIDI-rendered) | Larger scale, not real recordings |

Train/val/test split: use each dataset's official split where provided (IRMAS ships one); avoid
splitting the same song across train/test to prevent leakage.

## 4. Preprocessing Pipeline

```
raw audio → resample to 16 kHz mono → fixed-length windows (3s, silence-pad leftovers)
   → STFT → mel filterbank (n_mels=128) → log scale → log-mel spectrogram (per window)
```

- Sample rate: 16 kHz (standard ML choice, sufficient for instrument energy range — see Mod 2).
- Window: 3s clips, hop matching IRMAS convention.
- STFT: ~25ms frame, ~10ms hop, Hann window (see Mod 4).
- Output shape target: e.g. `128 mel bins × ~130 time frames` per clip (exact framing TBD once
  implemented).

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
- Experiment tracking: TBD (plain logging vs. Weights & Biases) — see open questions.

## 8. Repo Structure (proposed)

```
instrument_recognition/
├── notes/                # theory notes (existing)
├── spec.md               # this file
├── data/                 # raw + processed datasets (gitignored)
├── src/
│   ├── data/              # download + dataset classes
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

## 10. Open Questions (need your input before finalizing)

- [ ] Confirm Phase 1 dataset: IRMAS okay as the starting point?
- [ ] Framework preference: PyTorch assumed above — any objection / preference for TensorFlow?
- [ ] Compute available: local GPU, Colab, or CPU-only? Affects model size / training time
      expectations.
- [ ] Experiment tracking: plain logs/CSV fine for now, or set up W&B/TensorBoard from the start?
- [ ] Timeline/pace: any target deadline, or fully self-paced?
