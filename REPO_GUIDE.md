# Repo Guide — What Every File and Folder Is For

**What this file is:** a complete map of the repository as it stands at the close of Phase 1
(2026-08-22) — every folder and every file, one line on what it's for. Read this when you need
"where does X live" rather than "why was X built that way" (`DECISIONS.md`) or "what did X
achieve" (`results.md`). Git-tracked vs. gitignored is called out per item, since a chunk of the
repo (data, checkpoints, logs) exists on disk but deliberately isn't in version control.

---

## Root-level documentation

| File | Purpose |
|---|---|
| `README.md` | Entry point — points to every other doc below, quick setup instructions. |
| `spec.md` | The project spec: objective, scope/phases, dataset, pipeline, model, evaluation, tech stack, repo structure, milestones. Kept up to date with what's actually built, not just the original plan. |
| `HANDOFF.md` | Session-continuity notes — where the project stands, for picking it back up later (by a person or a future Claude session). The most "current status" of any doc. |
| `DECISIONS.md` | ADR-style decision log — every non-obvious choice (dataset, framework, compute, architecture, bug fixes), what alternatives were considered, and why. Most-recent-first. |
| `EXPERIMENTS.md` | Index of every training run: which config/checkpoint/TensorBoard-log belongs to which run, plus a short behavioral summary of how each run's training actually looked. |
| `results.md` | The actual numbers — per-run training curves, per-class precision/recall/F1, confusion matrices, verdicts. Chronological (unlike `DECISIONS.md`), since each run's story builds on the last. |
| `INFERENCE.md` | User-facing instructions for running the trained model on your own audio file (`src/predict.py`). Start here if you just want a prediction. |
| `REPO_GUIDE.md` | This file. |
| `requirements.txt` | Exact pinned package versions for the `Sound` conda environment (see `DECISIONS.md`, "Environment"/"Python version" entries for why these exact versions). `torch`/`torchaudio` included, matched to this machine's CUDA build. |
| `.gitignore` | Excludes `data/`, `checkpoints/`, `runs/`, `notebooks/dataset_visualisation/`, Python cache files, etc. — see the "Gitignored" section below for what that actually holds. |

## `notes/` — theory and technique reference (git-tracked)

| File | Purpose |
|---|---|
| `THEORY_NOTES.md` | The audio/DSP primer this whole project is built on — 7 modules from "what sound physically is" through spectrograms, mel scale, timbre, and bridging to ML. Written before any code existed. |
| `improv_cnn.md` | Menu of model-improvement techniques (regularization, augmentation, architecture, transfer learning, etc.), written *before* choosing what to try each round — plus the running account of what actually happened when each was tried, ending in a full Run 1-8 summary. |

## `src/` — all source code (git-tracked)

**`src/datasets/`** — data loading and the IRMAS download:

| File | Purpose |
|---|---|
| `download_irmas.py` | Downloads and extracts all 4 IRMAS Zenodo archives (Training + Testing Parts 1-3), verifies MD5 checksums. Also exports `download_file()`, reused by `panns_classifier.py` for its own checkpoint download. |
| `irmas_dataset.py` | `IRMASDataset` (log-mel input, for `BaselineCNN`) + `build_split()` — the song-grouped 80/10/10 train/val/test split every run uses, and `IRMAS_CLASSES`, the canonical 11-class label list/order. |
| `irmas_waveform_dataset.py` | `IRMASWaveformDataset` — same split (via `build_split()`), but returns raw waveform instead of log-mel, for PANNs/AST which compute their own internal features. |

**`src/preprocessing/`**:

| File | Purpose |
|---|---|
| `audio_to_logmel.py` | The audio → log-mel spectrogram pipeline (`load_audio`, `window`, `to_logmel`, `process_clip`) plus `spec_augment()` (SpecAugment, training-time input masking). `window()` is also reused directly by `predict.py` and `irmas_waveform_dataset.py` for its windowing/silence-padding logic, independent of log-mel conversion. |

**`src/models/`**:

| File | Purpose |
|---|---|
| `cnn.py` | `BaselineCNN` — the from-scratch model (5 conv/pool blocks, ~0.43M params), used by Runs 1-6. |
| `panns_classifier.py` | `PANNsClassifier` — wraps `panns_inference`'s `Cnn14` (AudioSet-pretrained) + a fresh linear head. Used by Run 7. |
| `ast_classifier.py` | `ASTClassifier` — wraps HuggingFace's `ASTForAudioClassification` (AudioSet-pretrained transformer) + loop-padding logic for the input-length mismatch. Used by Run 8. |
| `registry.py` | `build_model()` / `build_dataset()` / `build_collate_fn()` / `load_checkpoint()` — dispatches the right model class, Dataset variant, and collate function from a config's (or checkpoint's) `model.name`. The single place that knows about all three architectures; everything else (`train.py`, `evaluate.py`, `ensemble_evaluate.py`, `predict.py`) goes through this rather than hardcoding a model type. |

**`src/` top level**:

| File | Purpose |
|---|---|
| `losses.py` | `FocalLoss` and `compute_class_weights()` — the loss-function alternatives to plain cross-entropy, used by Run 6 (and the untried `class_weighted` config). |
| `train.py` | The training loop — used for every run. Reads a config, builds model/data/loss/optimizer via `registry.py`, handles AMP, SpecAugment, early stopping, checkpointing, and TensorBoard logging. |
| `evaluate.py` | Loads a checkpoint, runs it on the held-out `test` split, prints accuracy/per-class precision-recall-F1/confusion matrix — the source of every number in `results.md`. |
| `ensemble_evaluate.py` | Same as `evaluate.py`, but for 2+ checkpoints at once — averages their softmax probabilities (soft-vote). Produced the "Ensemble (2+3)" result in `results.md`. |
| `predict.py` | Inference on **any** audio file, any length, any trained checkpoint — see `INFERENCE.md`. The only script here meant for a real audio file rather than IRMAS's own pre-split clips. |
| `__init__.py` (in `src/`, `src/datasets/`, `src/models/`, `src/preprocessing/`) | Empty — makes each directory an importable Python package. Nothing to read here. |

## `configs/` — one YAML per training run (git-tracked)

Each file is a complete, self-contained training recipe; `train.py --config configs/<name>.yaml`
reproduces that run. See `EXPERIMENTS.md` for the full run→config→checkpoint mapping.

| Config | Run | What it tests |
|---|---|---|
| `base.yaml` | 1 | No regularization/augmentation — the raw baseline. |
| `reg.yaml` | 2 | + weight decay + conv-block dropout. |
| `specaug.yaml` | 3 | + SpecAugment. |
| `combined.yaml` | 4 | Run 2 + Run 3 together, 30 epochs. |
| `combined_extended.yaml` | 5 | Same as `combined.yaml`, 60 epochs instead of 30. |
| `focal.yaml` | 6 | Run 3's recipe + focal loss instead of cross-entropy. |
| `class_weighted.yaml` | 6b *(not run)* | Run 3's recipe + inverse-frequency class weighting — an untried control, see `DECISIONS.md`. |
| `panns.yaml` | 7 | PANNs, frozen backbone. |
| `ast.yaml` | 8 | AST, frozen backbone. |

## Gitignored — exists on disk, not in version control

These are excluded via `.gitignore` because they're either too large for a git repo, regenerable
from code + data, or pure local output. Listed here because "every folder and file" should
include them even though `git status` won't show them.

**`data/`** — the actual audio:
- `data/raw/_archives/` (11GB) — the 4 downloaded IRMAS zip files + `.extracted` marker files
  (per-archive, not per-directory — see `DECISIONS.md`, "Bug fix: IRMAS Testing extraction
  marker" entry for why that distinction matters).
- `data/raw/IRMAS-TrainingData/` (3.4GB) — 6,705 extracted training clips, 11 class subfolders.
- `data/raw/IRMAS-TestingData/` (8.0GB) — 2,874 extracted testing clips (multi-labeled,
  variable-length — not used as Phase 1's test set, see `spec.md` §3), across 3 part-subfolders.
- `data/processed/` — **currently empty, unused.** Preprocessing runs on-the-fly
  (`IRMASDataset.__getitem__` calls `process_clip()` live) rather than caching log-mel
  spectrograms to disk first — this directory is a placeholder from the original repo scaffold
  (`spec.md` §8) that nothing ended up writing to.

**`checkpoints/`** — one `.pt` file per trained run (model weights + epoch/val-acc/config):
`best.pt` (Run 1, legacy filename — predates the per-run-name convention), `run2_regularization.pt`
through `run6_focal_specaugment.pt` (~1.7MB each — `BaselineCNN` is small), `run7_panns_frozen.pt`
(313MB) and `run8_ast_frozen.pt` (329MB) — much larger since these store the full pretrained
backbone alongside the small trained head, not just the head. `run6b`/others not run don't exist.

**`runs/`** — one TensorBoard event-log directory per run (`baseline_cnn_2026-08-21T18-22-50/` for
Run 1 — legacy timestamp name — through `run8_ast_frozen/`). View with `tensorboard --logdir runs/`.

**`notebooks/dataset_visualisation/`** — 11 PNGs, one log-mel spectrogram plot per instrument
class, generated during preprocessing verification (visual sanity check that the pipeline
produces real harmonic structure, not noise). Regenerable from `audio_to_logmel.py` + real IRMAS
audio; not checkpointed against a specific script since it was a one-off visual check, not a
pipeline stage.

## `notebooks/` (git-tracked, mostly empty)

Only `.gitkeep` is tracked (keeps the empty directory present in git) — the actual exploration
artifact (`dataset_visualisation/`) is gitignored, see above. No `.ipynb` notebooks exist; despite
the directory name, no Jupyter work happened in this project — verification was done via
one-off scripts instead.
