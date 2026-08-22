# Instrument Recognition — Project Spec (v1.1)

**Status:** Finalized (2026-08-21), fully self-paced (no fixed deadline). See `DECISIONS.md` for
the reasoning behind each choice below.
**Background:** theory covered in `notes/THEORY_NOTES.md` (7 modules: sound → sampling → FFT →
spectrograms → mel scale → timbre → ML bridging). This spec turns that theory into an execution
plan.

**Progress (updated 2026-08-22): Phase 1 is functionally complete.** Every section below now
reflects what was actually built, not just planned — the whole pipeline (download →
preprocessing → model → training → evaluation) is implemented and has run end-to-end multiple
times. Best result: **78% test accuracy / 0.76 macro F1**, tied between a pretrained CNN (PANNs)
and a pretrained transformer (AST) — see Section 5 and Section 9. Full numbers, curves, and
confusion matrices for all 8 training runs + 1 ensemble: `results.md`. Reasoning behind every
non-obvious choice made along the way: `DECISIONS.md`. Phase 2 (multi-label) has not started.

## 1. Objective

Build a model that identifies which musical instrument(s) are present in an audio clip.

## 2. Scope — phased

- **Phase 1 (MVP):** predominant-instrument recognition — one label per 3s clip, single-label
  classification. **Status: functionally complete.** Best result 78% test accuracy / 0.76 macro
  F1 (Section 5, Section 9).
- **Phase 2:** multi-label recognition on full songs — predict *all* instruments present, not
  just the dominant one. **Status: not started.** IRMAS's Testing data (2,874 multi-labeled
  clips, downloaded — Section 3) is the natural entry point once this begins.
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

**Status: both fully downloaded and verified.** Training data (6,705 clips, `data/raw/
IRMAS-TrainingData/`) and Testing data (2,874 clips, `data/raw/IRMAS-TestingData/`, all 3 parts —
hit and fixed a real extraction bug along the way, see `DECISIONS.md`). Split logic
(`src/datasets/irmas_dataset.py`, `build_split()`) verified to produce zero song-group overlap
across train/val/test.

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
- **Status: implemented and verified.** Visually confirmed against real audio (harmonic bands on
  pitched instruments, transients on guitars, chord structure on piano — matches Module 6 theory).
  Also includes `spec_augment()` (SpecAugment, training-time input augmentation — Section 5).
  Only used by the from-scratch CNN — PANNs/AST (Section 5) compute their own internal features
  from raw waveform instead (`src/datasets/irmas_waveform_dataset.py`).

## 5. Model

**Status: both the baseline and the upgrade path have been built and evaluated. The upgrade path
won decisively.**

- **Baseline** (`src/models/cnn.py`, `BaselineCNN`): small CNN from scratch, as planned — 5
  conv/pool blocks + global-avg-pool + dense head, ~0.43M params, log-mel input. Iterated through
  6 training runs (regularization, SpecAugment, extended training, focal loss, and an ensemble —
  full detail in `results.md`) topping out at **65% test accuracy / 0.64 macro F1** (Run 5). The
  from-scratch ceiling turned out to be a genuine data-scarcity limit (~5,300 training clips),
  confirmed by how much the upgrade path below improved on it, not a fixable regularization or
  architecture problem.
- **Phase 1 upgrade path** (triggered — baseline's ceiling justified it): transfer learning via
  pretrained audio embeddings. Originally scoped as PANNs / YAMNet / VGGish; in practice **PANNs**
  and **AST** (Audio Spectrogram Transformer) were the two implemented, since `torchaudio`'s
  pretrained-model bundles include no AudioSet-pretrained classifier at all (verified directly,
  not assumed — see `DECISIONS.md`, "PANNs input pipeline" entry) and AST was added specifically
  to give a direct, pretrained answer to "should this be a transformer instead" (see
  `notes/improv_cnn.md`, "Should we use a transformer?" section). Both frozen-backbone (only a
  small new head trained) + AudioSet-pretrained:
  - `src/models/panns_classifier.py` — PANNs CNN14, raw 32kHz waveform input.
  - `src/models/ast_classifier.py` — AST, 16kHz/128-mel input (loop-padded to match its
    pretrained 10s-clip positional embeddings).
  - Both reach **78% test accuracy / 0.76 macro F1**, tied — pretraining is what mattered, not
    the architecture family. **This is Phase 1's current best result.**
  - `src/models/registry.py` dispatches model class + Dataset variant from
    `configs/*.yaml`'s `model.name` (`baseline_cnn` / `panns_cnn14` / `ast`).
- **Phase 2:** same backbone, swap output head to per-class sigmoid + BCE loss (multi-label).
  **Not started.**

## 6. Evaluation

- Phase 1 (single-label): accuracy, per-class precision/recall/F1, confusion matrix.
  **Implemented** (`src/evaluate.py`, `src/ensemble_evaluate.py`) and run for every training run
  so far. Best result: 78% accuracy, macro F1 0.76 (PANNs and AST, tied) on the 684-clip held-out
  test split, up from 56%/0.53 for the from-scratch baseline. Full per-run tables and confusion
  matrices: `results.md`.
- Phase 2 (multi-label): macro-F1 and micro-F1 (primary metrics), per-class precision/recall.
  **Not started.**
- Track a fixed validation set throughout; no test-set peeking until final report. **Honored** —
  the held-out `test` split was only evaluated after each run's training was complete, never used
  to pick hyperparameters (that's what `val` is for, per the song-grouped split in
  `src/datasets/irmas_dataset.py`).

## 7. Tech Stack

- Python, PyTorch (model + training loop).
- `librosa` or `torchaudio` for audio I/O and mel-spectrogram extraction.
- `scikit-learn` for metrics.
- Experiment tracking: TensorBoard (`torch.utils.tensorboard`).
- **Pretrained models (added for Section 5's upgrade path):** `panns-inference` (PANNs CNN14,
  AudioSet-pretrained) and `transformers` (HuggingFace — AST, AudioSet-pretrained,
  `MIT/ast-finetuned-audioset-10-10-0.4593`). Both auto-download their pretrained checkpoints on
  first use (`~/panns_data/` and the HuggingFace Hub cache respectively).
- Compute: local RTX 3050 (6GB VRAM, driver 595.84, CUDA 13.2) — confirmed working
  (`torch.cuda.is_available() == True`). Batch size 32–128, `torch.cuda.amp` mixed precision.
- Environment: conda env named `Sound` (Python 3.13) — see `README.md` for setup. All project
  commands (installs, preprocessing, training, evaluation) run inside it, not the system Python.
  Exact pinned package versions in `requirements.txt`.

## 8. Repo Structure (as built — see `README.md` for the same map with descriptions)

```
instrument-recognition/
├── README.md, spec.md, HANDOFF.md      # entry point, this file, session-continuity notes
├── DECISIONS.md, EXPERIMENTS.md, results.md   # reasoning / run-artifact index / all numbers
├── requirements.txt
├── notes/
│   ├── THEORY_NOTES.md    # audio/DSP primer (7 modules)
│   └── improv_cnn.md      # model-improvement technique menu + Phase A/B outcomes
├── data/                  # raw datasets (gitignored)
├── checkpoints/, runs/    # trained weights, TensorBoard logs (gitignored — see EXPERIMENTS.md)
├── src/
│   ├── datasets/
│   │   ├── download_irmas.py         # IRMAS download + extract
│   │   ├── irmas_dataset.py          # log-mel Dataset + song-grouped split (build_split)
│   │   └── irmas_waveform_dataset.py # raw-waveform Dataset (for PANNs/AST)
│   ├── preprocessing/
│   │   └── audio_to_logmel.py        # audio -> log-mel pipeline + SpecAugment
│   ├── models/
│   │   ├── cnn.py                    # BaselineCNN (from-scratch)
│   │   ├── panns_classifier.py       # PANNs CNN14 (pretrained, frozen)
│   │   ├── ast_classifier.py         # AST transformer (pretrained, frozen)
│   │   └── registry.py               # model/dataset dispatch from config
│   ├── losses.py                     # FocalLoss, class-weighting
│   ├── train.py, evaluate.py, ensemble_evaluate.py
├── configs/               # one yaml per training run (9 so far — see EXPERIMENTS.md)
└── notebooks/              # exploration / visualization
```

## 9. Milestones

1. ✅ Repo scaffold + IRMAS download script. Both training (6,705 clips) and testing (2,874
   clips) data downloaded and verified.
2. ✅ Preprocessing pipeline (audio → log-mel spectrogram), verified visually on a few samples.
3. ✅ Baseline CNN training on IRMAS (Phase 1), report accuracy/F1. Run 1 (from-scratch, no
   tuning): 56% / macro F1 0.53. Iterated (regularization, SpecAugment, extended training,
   ensembling, focal loss — 6 runs total): best from-scratch result **65% / macro F1 0.64**
   (Run 5). Full breakdown: `results.md`.
4. ✅ Iterate: pretrained-embedding upgrade — triggered given the from-scratch ceiling, and it
   worked decisively. PANNs (pretrained CNN) and AST (pretrained transformer), both frozen-
   backbone: **78% / macro F1 0.76, tied.** This is Phase 1's current best result and where the
   project currently stands. Full breakdown: `results.md`, Runs 7-8.
5. ⬜ Phase 2: multi-label pipeline on OpenMIC/Slakh2100. **Not started.** IRMAS Testing data
   (2,874 multi-labeled clips) is already downloaded and ready for this once it begins.

**Open, not yet decided** (see `HANDOFF.md` for full context): accept 78% as Phase 1's final
result and move to Phase 2, or push further on Phase 1 first (fine-tuning PANNs/AST rather than
frozen-backbone-only, AST embedding interpolation instead of loop-padding, or ensembling Run 7 +
Run 8 the way Run 2 + Run 3 were ensembled in Phase A).
