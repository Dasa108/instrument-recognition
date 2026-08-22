# Experiments Log

**Purpose:** tracks *what was run and where its artifacts live* — config file, model code, saved
checkpoint, TensorBoard logs — so any run can be found, inspected, or reproduced later, plus a
short *behavioral* summary per run (how training actually looked, not just the final score) so
the personality of each run is visible at a glance. This file is the index; it doesn't duplicate
the other docs:
- **Full numbers** (per-epoch curves, per-class metrics, confusion matrices): `results.md`.
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

## Model behavior per run

Not just final scores — how each run's training actually *looked*. Full curves/confusion
matrices for all of this: `results.md`.

**Run 1 (baseline)** — textbook overfitting. Train accuracy climbed smoothly to 93%+, but val
was erratic throughout (briefly crashed to 17-20% around epochs 3-4) and trended flat-to-worse
after epoch 15 while train kept climbing — ~42-point train/val gap at the end. "pia" acted as a
dumping ground for other classes' errors.

**Run 2 (+ weight decay + dropout)** — a different personality entirely: train and val tracked
each other almost exactly for all 30 epochs, val even nudging *ahead* of train by the end. No
crashes, no divergence, just a slower, honest climb — fully eliminated Run 1's overfitting
signature, at the cost of a much higher absolute training loss (the task was deliberately made
harder).

**Run 3 (+ SpecAugment only)** — best peak accuracy of the individual techniques, but volatile:
val accuracy swung as wildly as 61%→31% between adjacent epochs late in training. Also developed a
specific pathology — clarinet recall collapsed to 0.16 (model got extremely conservative, only
guessing clarinet when very confident). Stronger peak, less predictable.

**Run 4 (Run 2 + Run 3 combined)** — stable like Run 2 (no oscillation, no overfitting) but slow —
still climbing gradually at epoch 30, never catching up to either individual technique. Looked at
first like "combining regularizers backfires"; the curve shape (steady, unfinished climb, not a
peak-and-decline) was the tell it was actually just under-trained.

**Ensemble (Run 2 + Run 3, no retraining)** — simply averaging softmax outputs beat both inputs
(65% vs. 60%/63%). Confirmed the two models' mistakes were genuinely complementary rather than the
same mistakes twice — Run 2's steadiness partially rescued some of Run 3's clarinet blind spot.

**Run 5 (Run 4's recipe, 60 epochs instead of 30)** — vindicated the under-training theory
decisively. Same stable, monotonic climb as Run 4, just given twice the runway — kept climbing
cleanly the whole time (train/val gap stayed ~1-2 points even at epoch 60) and nearly doubled
Run 4's final val accuracy. Best-balanced result of Phase A — no single class collapsed.

**Run 6 (focal loss, built on Run 3)** — the most interesting behavior of the batch. It *did* fix
what it targeted — clarinet recall jumped from 0.16 to 0.63 — but manufactured a new, worse
problem: organ recall hit a literal 1.00, meaning the model started defaulting to "organ" as a
catch-all guess whenever uncertain, dragging its precision down to 0.32. First run where early
stopping actually triggered (val stalled after epoch 13). Net: fixed one failure mode by trading
it for a bigger one.

**Run 6b (class-weighted, optional control)** — not run (deprioritized in favor of Phase B once
Run 6 already showed the confusions weren't frequency-driven — see `DECISIONS.md`).

**Run 7 (PANNs, frozen) and Run 8 (AST, frozen)** — behaved almost nothing like the Phase A runs.
Both hit ~70-73% val accuracy in the *first epoch* — something no from-scratch run got close to
even after 30-60 epochs — then climbed gently and plateaued smoothly with no oscillation at all.
Makes sense mechanically: only a small linear head (~22K params) was actually training, a far
easier optimization problem than tuning a full CNN. AST converged even faster than PANNs
(plateaued by epoch 8 vs. epoch 14) and had the smoothest curve of any run in the whole project.
Both landed at 78%/0.76 — tied overall, but not identical under the hood: AST notably better on
organ and voice, PANNs slightly better on clarinet and trumpet, and AST's single worst weak spot
was cello↔violin confusion (12 misclassifications, the largest single off-diagonal entry across
every run in the project).

**Thread running through every single run, regardless of technique:** clarinet↔sax/trumpet,
cello↔violin, and guitar→piano confusions never fully disappeared — they shrank a lot under
pretraining (roughly a third the magnitude of the worst from-scratch runs) but never zeroed out,
suggesting some of these instrument pairs are genuinely acoustically similar, not purely an
artifact of too little training data.

## How to reproduce or inspect a run

```bash
conda activate Sound
python -m src.train --config configs/<name>.yaml      # re-run training
python -m src.evaluate --checkpoint checkpoints/<name>.pt   # eval an existing checkpoint
tensorboard --logdir runs/                              # browse all runs' curves at once
```

Every checkpoint's `.pt` file also stores its own full config dict (`ckpt["config"]`) alongside the
weights, so a checkpoint is self-describing even without cross-referencing this file.
