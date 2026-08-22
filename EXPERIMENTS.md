# Experiments Log

**Purpose:** tracks *what was run and where its artifacts live* — config file, model code, saved
checkpoint, TensorBoard logs — so any run can be found, inspected, or reproduced later. This file
is the index; it doesn't duplicate the other docs:
- **Result numbers** (accuracy, F1, confusion matrix): `results.md`.
- **Why each variant was chosen**: `DECISIONS.md` ("Runs 2-4" entry) and `notes/improv_cnn.md`.
- **Overall project status**: `HANDOFF.md`.

## Note on code versioning

**Correction (2026-08-22):** this section previously claimed "no commits since the initial
scaffold" — that's now stale. `git log` shows 2 commits (`95c040c` initial scaffold, `53a058e`
"Implement Phase 1 pipeline..." covering the download script through Runs 1-4), working tree
clean as of that commit. Runs 5+ below (and any further code changes) postdate `53a058e` and will
get their own commit once logged.

One consequence still worth knowing regardless of commit state: `src/train.py`/`src/models/cnn.py`
were extended for Runs 2-4 (added `weight_decay`, conv-block `dropout`, SpecAugment, early
stopping) *after* Run 1 finished — Run 1 predates that code within the working tree, though both
ended up captured in the same commit (`53a058e`) once Runs 1-4 were all done. Its config,
`configs/base.yaml`, sets `dropout: 0.0`, `weight_decay: 0.0`, `augmentation.specaugment: false`
specifically so that running it again with the *current* code reproduces Run 1's original
architecture/behavior exactly (those knobs are no-ops at their default values) — so Run 1 stays
reproducible from current code despite predating the code that added those options. Same logic
applies to Runs 1-4 vs. the `loss:` dispatch added for Run 6 (absent `loss:` block → today's plain
cross-entropy, unchanged).

## Runs

| Run | What it tests | Config | Checkpoint | TensorBoard logs | Status |
|---|---|---|---|---|---|
| 1 — baseline | No regularization/augmentation beyond what the model always had (BN, 0.3 dropout in the dense head only) | `configs/base.yaml` | `checkpoints/best.pt` *(legacy name — predates the per-run-name convention below)* | `runs/baseline_cnn_2026-08-21T18-22-50/` *(legacy timestamp name)* | Done — see `results.md` Run 1 |
| 2 — regularization | + weight decay (1e-4) + conv-block dropout (0.2), isolated | `configs/reg.yaml` | `checkpoints/run2_regularization.pt` | `runs/run2_regularization/` | Done — see `results.md` Run 2 |
| 3 — SpecAugment | + time/frequency masking on input, isolated | `configs/specaug.yaml` | `checkpoints/run3_specaugment.pt` | `runs/run3_specaugment/` | Done — see `results.md` Run 3 |
| 4 — combined | Run 2 + Run 3 together | `configs/combined.yaml` | `checkpoints/run4_combined.pt` | `runs/run4_combined/` | Done — see `results.md` Run 4 |
| Ensemble (2+3) | Soft-vote average of Run 2 + Run 3's softmax outputs, no retraining | *(none — see `src/ensemble_evaluate.py`)* | *(none)* | *(none)* | Done — see `results.md` "Ensemble" section. **Best result so far.** |
| 5 — combined, extended | Run 4's recipe, more epochs (60 vs 30) — tests Run 4's under-training hypothesis | `configs/combined_extended.yaml` | `checkpoints/run5_combined_extended.pt` | `runs/run5_combined_extended/` | Done — see `results.md` Run 5. **Best single-model result.** |
| 6 — focal + SpecAugment | Run 3's recipe + focal loss (targets the confirmed confusions instead of overfitting) | `configs/focal.yaml` | `checkpoints/run6_focal_specaugment.pt` | `runs/run6_focal_specaugment/` | Done — see `results.md` Run 6. Early-stopped at epoch 20 (best epoch 13). Fixed clarinet collapse, caused a worse organ collapse — net worse than Run 3. |
| 6b — class-weighted + SpecAugment *(optional control)* | Run 3's recipe + inverse-frequency class weighting | `configs/class_weighted.yaml` | `checkpoints/run6b_class_weighted_specaugment.pt` | `runs/run6b_class_weighted_specaugment/` | Not run — deprioritized in favor of Phase B, see `results.md` Run 6 verdict |

Reproduce the ensemble result: `python -m src.ensemble_evaluate --checkpoints checkpoints/run2_regularization.pt checkpoints/run3_specaugment.pt`.

| 7 — PANNs (frozen) | AudioSet-pretrained CNN14 backbone, frozen, fresh linear head | `configs/panns.yaml` | `checkpoints/run7_panns_frozen.pt` | `runs/run7_panns_frozen/` | Done — see `results.md` Run 7. **Best result (tied with Run 8).** |
| 8 — AST (frozen) | AudioSet-pretrained Audio Spectrogram Transformer, frozen, fresh head | `configs/ast.yaml` | `checkpoints/run8_ast_frozen.pt` | `runs/run8_ast_frozen/` | Done — see `results.md` Run 8. **Best result (tied with Run 7).** |

**Phases A + B both complete (2026-08-22).** Best results: Run 7 (PANNs) and Run 8 (AST), tied at
78% accuracy / 0.76 macro F1 — both ~13 points ahead of the best from-scratch result (Run 5, 65%).
Full comparison table: `results.md`, "Summary — Phases A + B complete".

Reproduce either: `python -m src.train --config configs/panns.yaml` (auto-downloads the PANNs
checkpoint to `~/panns_data/` on first run) or `python -m src.train --config configs/ast.yaml`
(auto-downloads via HuggingFace Hub on first run).

## How to reproduce or inspect a run

```bash
conda activate Sound
python -m src.train --config configs/<name>.yaml      # re-run training
python -m src.evaluate --checkpoint checkpoints/<name>.pt   # eval an existing checkpoint
tensorboard --logdir runs/                              # browse all runs' curves at once
```

Every checkpoint's `.pt` file also stores its own full config dict (`ckpt["config"]`) alongside the
weights, so a checkpoint is self-describing even without cross-referencing this file.
