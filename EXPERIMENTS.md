# Experiments Log

**Purpose:** tracks *what was run and where its artifacts live* — config file, model code, saved
checkpoint, TensorBoard logs — so any run can be found, inspected, or reproduced later. This file
is the index; it doesn't duplicate the other docs:
- **Result numbers** (accuracy, F1, confusion matrix): `results.md`.
- **Why each variant was chosen**: `DECISIONS.md` ("Runs 2-4" entry) and `notes/improv_cnn.md`.
- **Overall project status**: `HANDOFF.md`.

## Note on code versioning

The repo has had **no commits since the initial scaffold** (`git log`: one commit, "Initial
project setup") — everything since (repo scaffold, download script, preprocessing, model,
training loop, and all four runs below) has been done directly in the working tree, uncommitted.
So "code version" per run below means *config file* + *current shared code in `src/`*, not a git
SHA — there's no commit history to pin an exact snapshot to. One consequence worth knowing:
`src/train.py`/`src/models/cnn.py` were extended for Runs 2-4 (added `weight_decay`, conv-block
`dropout`, SpecAugment, early stopping) *after* Run 1 finished — Run 1 predates that code. Its
config, `configs/base.yaml`, sets `dropout: 0.0`, `weight_decay: 0.0`, `augmentation.specaugment:
false` specifically so that running it again with the *current* code reproduces Run 1's original
architecture/behavior exactly (those knobs are no-ops at their default values) — so Run 1 stays
reproducible from current code despite predating the code that added those options.

## Runs

| Run | What it tests | Config | Checkpoint | TensorBoard logs | Status |
|---|---|---|---|---|---|
| 1 — baseline | No regularization/augmentation beyond what the model always had (BN, 0.3 dropout in the dense head only) | `configs/base.yaml` | `checkpoints/best.pt` *(legacy name — predates the per-run-name convention below)* | `runs/baseline_cnn_2026-08-21T18-22-50/` *(legacy timestamp name)* | Done — see `results.md` Run 1 |
| 2 — regularization | + weight decay (1e-4) + conv-block dropout (0.2), isolated | `configs/reg.yaml` | `checkpoints/run2_regularization.pt` | `runs/run2_regularization/` | Done — see `results.md` Run 2 |
| 3 — SpecAugment | + time/frequency masking on input, isolated | `configs/specaug.yaml` | `checkpoints/run3_specaugment.pt` | `runs/run3_specaugment/` | Done — see `results.md` Run 3 |
| 4 — combined | Run 2 + Run 3 together | `configs/combined.yaml` | `checkpoints/run4_combined.pt` | `runs/run4_combined/` | Done — see `results.md` Run 4 |

All four runs are complete — see `results.md` for the full breakdown and cross-run summary table.
**Best result: Run 3 (SpecAugment alone).** None of the three triggered early stopping (patience 7
on val accuracy) — all ran the full 30-epoch cap in their configs.

## How to reproduce or inspect a run

```bash
conda activate Sound
python -m src.train --config configs/<name>.yaml      # re-run training
python -m src.evaluate --checkpoint checkpoints/<name>.pt   # eval an existing checkpoint
tensorboard --logdir runs/                              # browse all runs' curves at once
```

Every checkpoint's `.pt` file also stores its own full config dict (`ckpt["config"]`) alongside the
weights, so a checkpoint is self-describing even without cross-referencing this file.
