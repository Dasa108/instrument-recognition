# Instrument Recognition

ML model to recognize musical instruments in audio. Learning project — see the docs below for the
full story, not just the code.

- **`HANDOFF.md`** — start here. Where the project stands and what's next.
- **`INFERENCE.md`** — **run the trained model on your own audio file.** Start here if you just
  want a prediction, not the full story.
- **`notes/THEORY_NOTES.md`** — the audio/DSP theory primer this project is built on.
- **`spec.md`** — the project spec (scope, dataset, pipeline, model, evaluation) — kept up to date
  with what's actually been built, not just the original plan.
- **`DECISIONS.md`** — why each non-obvious choice (dataset, framework, compute, tooling, model
  architecture) was made, and what alternatives were considered.
- **`results.md`** — every training run's actual numbers: curves, per-class metrics, confusion
  matrices.
- **`EXPERIMENTS.md`** — which config/checkpoint/TensorBoard-log belongs to which run, plus a
  short behavioral summary of how each run's training actually looked.
- **`notes/improv_cnn.md`** — the model-improvement menu and how the from-scratch CNN was iterated
  toward, then surpassed by, pretrained embeddings.
- **`REPO_GUIDE.md`** — what every folder and file in this repo is for, including the gitignored
  ones (data, checkpoints, logs).

## Layout

```
src/
├── datasets/       # download + Dataset/DataLoader classes (code, not data)
├── preprocessing/  # audio -> log-mel spectrogram + SpecAugment
├── models/         # BaselineCNN, PANNs, AST wrappers + the model/dataset registry
├── losses.py       # focal loss, class weighting
├── train.py
├── evaluate.py
├── ensemble_evaluate.py
└── predict.py      # run a trained model on any audio file — see INFERENCE.md
data/               # raw datasets (gitignored — not in version control)
checkpoints/, runs/ # trained weights, TensorBoard logs (gitignored — see EXPERIMENTS.md)
notebooks/          # exploration / visualization
configs/            # one yaml per training run
```

## Setup

Uses a conda environment named `Sound` (Python 3.13) — already created; all project commands run
inside it.

```bash
conda activate Sound
pip install -r requirements.txt
# torch/torchaudio are pinned in requirements.txt too, matched to this machine's CUDA version —
# see DECISIONS.md if setting this up on different hardware.
```

## Status

**Phase 1 (single-label predominant-instrument classification) is closed (2026-08-22).** Best
result: **78% test accuracy / 0.76 macro F1**, tied between a pretrained CNN (PANNs) and a
pretrained transformer (AST) — see `results.md`. Want a prediction on your own audio right now?
See `INFERENCE.md`. Phase 2 (multi-label) has not started — see `HANDOFF.md` for what's next.
