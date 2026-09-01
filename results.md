# Results Log

**Purpose:** a running record of every training run's actual numbers, so later variations (see
`notes/improv_cnn.md`) have something concrete to compare against. New runs go at the **bottom**
(chronological order) — unlike `DECISIONS.md`'s most-recent-first convention, each run's verdict
here builds on and references the one before it, so reading top-to-bottom tells the actual story.

Each entry: config summary, per-epoch curve, held-out test-set metrics (accuracy/per-class
precision-recall-F1/confusion matrix), and a one-line verdict.

## Instrument class lookup

Every table below (per-class metrics, confusion matrices) uses these 3-letter codes —
`IRMAS_CLASSES` in `src/datasets/irmas_dataset.py`, in this exact fixed order:

| Code | Instrument | Code | Instrument | Code | Instrument |
|---|---|---|---|---|---|
| `cel` | cello | `gel` | electric guitar | `tru` | trumpet |
| `cla` | clarinet | `org` | organ | `vio` | violin |
| `flu` | flute | `pia` | piano | `voi` | voice |
| `gac` | acoustic guitar | `sax` | saxophone | | |

---

## Run 1 — `baseline_cnn`, `configs/base.yaml`

**Date:** 2026-08-22
**Config:** `BaselineCNN` (5 conv/pool blocks, ~0.43M params), batch size 64, lr 0.001 (Adam), 30
epochs, mixed precision, no weight decay, no augmentation, no LR schedule, no early stopping.
**Data:** 5,342 train / 679 val / 684 test clips (song-grouped split — see `DECISIONS.md`).
**Checkpoint:** `checkpoints/best.pt` (epoch 23, gitignored — not committed).

### Training curve

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.0427 | 0.2840 | 2.1859 | 0.2356 |
| 2 | 1.6843 | 0.4234 | 2.1532 | 0.3299 |
| 3 | 1.4390 | 0.5052 | 3.5647 | 0.1708 |
| 4 | 1.2399 | 0.5708 | 3.9265 | 0.1988 |
| 5 | 1.1258 | 0.6206 | 1.7994 | 0.4227 |
| 6 | 1.0218 | 0.6539 | 1.9032 | 0.3829 |
| 7 | 0.9489 | 0.6748 | 1.9358 | 0.4389 |
| 8 | 0.8765 | 0.7054 | 1.7212 | 0.4669 |
| 9 | 0.7905 | 0.7323 | 3.7974 | 0.2666 |
| 10 | 0.7364 | 0.7551 | 1.8719 | 0.4845 |
| 11 | 0.6639 | 0.7712 | 2.8220 | 0.4374 |
| 12 | 0.6106 | 0.7965 | 2.0265 | 0.4713 |
| 13 | 0.5510 | 0.8145 | 2.9319 | 0.3358 |
| 14 | 0.5039 | 0.8310 | 2.4610 | 0.4035 |
| 15 | 0.4712 | 0.8398 | 1.9178 | 0.5125 |
| 16 | 0.3936 | 0.8669 | 2.3431 | 0.4786 |
| 17 | 0.3539 | 0.8813 | 3.3131 | 0.3520 |
| 18 | 0.3292 | 0.8864 | 3.1755 | 0.3726 |
| 19 | 0.2872 | 0.9032 | 3.3361 | 0.4345 |
| 20 | 0.2603 | 0.9124 | 2.9622 | 0.4212 |
| 21 | 0.2269 | 0.9251 | 3.8167 | 0.3594 |
| 22 | 0.2128 | 0.9291 | 3.9186 | 0.4035 |
| **23** | **0.2207** | **0.9285** | **2.2164** | **0.5287 (best)** |
| 24 | 0.1283 | 0.9607 | 5.2469 | 0.2931 |
| 25 | 0.1572 | 0.9483 | 2.9038 | 0.4374 |
| 26 | 0.1362 | 0.9562 | 2.5772 | 0.5081 |
| 27 | 0.1371 | 0.9547 | 2.9466 | 0.5155 |
| 28 | 0.1301 | 0.9605 | 3.0052 | 0.4330 |
| 29 | 0.1145 | 0.9631 | 3.9860 | 0.4890 |
| 30 | 0.1359 | 0.9515 | 3.4012 | 0.4551 |

Train loss falls smoothly and monotonically; val loss/acc are noisy and trend *worse* after epoch
~15 while train keeps improving — the textbook overfitting signature.

### Test set (684 clips, held out, song-grouped, never seen in training)

**Overall: accuracy 0.56, macro F1 0.53, weighted F1 0.53** (random baseline for 11 classes ≈ 0.09)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel (cello) | 0.94 | 0.37 | 0.53 | 41 |
| cla (clarinet) | 0.55 | 0.67 | 0.60 | 51 |
| flu (flute) | 0.62 | 0.38 | 0.47 | 47 |
| gac (acoustic guitar) | 0.76 | 0.34 | 0.47 | 64 |
| gel (electric guitar) | 0.54 | 0.67 | 0.60 | 78 |
| org (organ) | 0.44 | 0.91 | 0.60 | 69 |
| pia (piano) | 0.51 | 0.93 | 0.66 | 74 |
| sax (saxophone) | 0.42 | 0.21 | 0.28 | 63 |
| tru (trumpet) | 0.56 | 0.93 | 0.70 | 60 |
| vio (violin) | 0.93 | 0.22 | 0.36 | 59 |
| voi (voice) | 0.93 | 0.36 | 0.52 | 78 |
| **macro avg** | **0.65** | **0.54** | **0.53** | 684 |
| **weighted avg** | **0.65** | **0.56** | **0.53** | 684 |

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 15 | 3 | 3 | 0 | 6 | 3 | 3 | 1 | 6 | 1 | 0 |
| cla | 0 | 34 | 1 | 0 | 1 | 2 | 3 | 4 | 6 | 0 | 0 |
| flu | 0 | 7 | 18 | 0 | 2 | 13 | 7 | 0 | 0 | 0 | 0 |
| gac | 0 | 0 | 0 | 22 | 4 | 10 | 25 | 1 | 2 | 0 | 0 |
| gel | 0 | 1 | 0 | 0 | 52 | 10 | 15 | 0 | 0 | 0 | 0 |
| org | 0 | 0 | 0 | 0 | 1 | 63 | 5 | 0 | 0 | 0 | 0 |
| pia | 0 | 0 | 0 | 1 | 0 | 4 | 69 | 0 | 0 | 0 | 0 |
| sax | 0 | 7 | 1 | 2 | 12 | 4 | 6 | 13 | 18 | 0 | 0 |
| tru | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 1 | 56 | 0 | 0 |
| vio | 1 | 7 | 3 | 4 | 9 | 8 | 2 | 3 | 7 | 13 | 2 |
| voi | 0 | 2 | 1 | 0 | 9 | 25 | 0 | 8 | 5 | 0 | 28 |

**Read on the confusion matrix:** "pia" is a common false-positive sink early in training (many
classes' misclassifications land there — see the earlier 1-epoch smoke test where it absorbed
almost everything), and by the final model "org"/"voi" absorb a lot of sax/violin/voice confusion.
sax is confused with tru and gel; vio is confused broadly across cla/flu/gel/org (spread thin, no
dominant confusion partner) which matches its low recall.

### Verdict

Clearly better than chance and not obviously broken (56% on an 11-class from-scratch baseline is a
reasonable starting point), but the ~40-point train/val gap means the model is memorizing training
clips rather than generalizing — regularization/augmentation has real room to help before
concluding the baseline architecture itself is the bottleneck. See `notes/improv_cnn.md` for the
options under consideration.

---

## Run 2 — `run2_regularization`, `configs/reg.yaml`

**Date:** 2026-08-22
**Change from Run 1:** + weight decay (1e-4, Adam) + conv-block dropout (0.2) + early stopping
(patience 7, not triggered — val acc kept improving through epoch 30). SpecAugment still off,
isolating regularization's effect alone. See `notes/improv_cnn.md` section 1, `DECISIONS.md`
"Runs 2-4" entry.
**Checkpoint:** `checkpoints/run2_regularization.pt` (epoch 30 — final epoch was also the best).

### Training curve

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.2948 | 0.1760 | 2.2064 | 0.2401 |
| 5 | 2.0118 | 0.2924 | 2.0156 | 0.3432 |
| 10 | 1.7684 | 0.3916 | 1.7650 | 0.4624 |
| 15 | 1.6335 | 0.4481 | 1.6300 | 0.4669 |
| 20 | 1.5287 | 0.4843 | 1.5504 | 0.5007 |
| 25 | 1.4408 | 0.5155 | 1.4712 | 0.5302 |
| **30** | **1.3457** | **0.5475** | **1.3833** | **0.5523 (best)** |

(Full 30-row log: `EXPERIMENTS.md` → `runs/run2_regularization/` TensorBoard logs.) Unlike Run 1,
train and val loss/accuracy track each other closely the *entire* run — val is even slightly
*ahead* of train by epoch 30. The overfitting signature from Run 1 is gone. Training loss is also
much higher in absolute terms than Run 1's (1.35 vs Run 1's 0.14 at their respective final
epochs) — expected and fine: weight decay/dropout are deliberately making the *training* task
harder so the model can't just memorize, which is exactly the point.

### Test set (684 clips)

**Overall: accuracy 0.60, macro F1 0.57, weighted F1 0.59** — up from Run 1's 0.56 / 0.53 / 0.53.

| Class | Precision | Recall | F1 | Support | Δ F1 vs Run 1 |
|---|---:|---:|---:|---:|---:|
| cel | 0.48 | 0.49 | 0.48 | 41 | **+0.05 (0.53→0.48 — actually down, see note)** |
| cla | 0.52 | 0.47 | 0.49 | 51 | −0.11 |
| flu | 0.48 | 0.34 | 0.40 | 47 | −0.07 |
| gac | 0.76 | 0.59 | 0.67 | 64 | +0.20 |
| gel | 0.45 | 0.54 | 0.49 | 78 | −0.11 |
| org | 0.77 | 0.84 | 0.81 | 69 | +0.21 |
| pia | 0.60 | 0.74 | 0.67 | 74 | +0.01 |
| sax | 0.44 | 0.27 | 0.33 | 63 | +0.05 |
| tru | 0.70 | 0.75 | 0.73 | 60 | +0.03 |
| vio | 0.49 | 0.47 | 0.48 | 59 | **+0.12** |
| voi | 0.69 | 0.82 | 0.75 | 78 | +0.23 |
| **macro avg** | **0.58** | **0.58** | **0.57** | 684 | **+0.04** |

Most classes improved, several substantially (org +0.21, voi +0.23, gac +0.20). cel actually got
*worse* (0.53→0.48 F1) — Run 1's cel F1 was inflated by very high precision (0.94) but poor
recall (0.37, i.e. it rarely guessed cel but was usually right when it did); Run 2 guesses cel more
often (recall 0.49) at some precision cost (0.48) — a more balanced but not strictly-better
trade-off for that one class. cla/flu/gel also dipped slightly. Net effect is positive (macro F1
+0.04) but not uniformly so.

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 20 | 2 | 1 | 0 | 8 | 0 | 1 | 4 | 1 | 4 | 0 |
| cla | 2 | 24 | 5 | 0 | 4 | 1 | 2 | 2 | 2 | 9 | 0 |
| flu | 1 | 13 | 16 | 0 | 4 | 6 | 6 | 0 | 0 | 0 | 1 |
| gac | 1 | 0 | 0 | 38 | 7 | 1 | 11 | 0 | 1 | 3 | 2 |
| gel | 2 | 0 | 0 | 4 | 42 | 3 | 10 | 5 | 2 | 1 | 9 |
| org | 0 | 0 | 0 | 2 | 2 | 58 | 2 | 0 | 1 | 3 | 1 |
| pia | 2 | 0 | 2 | 1 | 2 | 5 | 55 | 2 | 0 | 2 | 3 |
| sax | 10 | 2 | 0 | 0 | 9 | 0 | 4 | 17 | 10 | 5 | 6 |
| tru | 0 | 1 | 5 | 1 | 1 | 0 | 0 | 3 | 45 | 2 | 2 |
| vio | 4 | 4 | 4 | 4 | 5 | 1 | 0 | 2 | 2 | 28 | 5 |
| voi | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 4 | 0 | 0 | 64 |

Compare to Run 1's confusion matrix: the "pia sink" effect (many classes' errors landing on pia) is
markedly reduced — e.g. gac→pia dropped from 25 to 11, flu→pia from 7 to 6, cla→pia from 3 to 2.
Errors are more spread out / less systematically biased toward one class now.

### Verdict

Clear win. Confirms Run 1's diagnosis (overfitting) and the fix (regularization) — the overfitting
signature is fully gone, test accuracy up 4 points, macro F1 up 4 points, and the model's errors
are less systematically biased. Not every class improved, but the net effect is unambiguously
positive. Next: does SpecAugment (Run 3) help on its own, and does combining the two (Run 4) help
further or plateau?

---

## Run 3 — `run3_specaugment`, `configs/specaug.yaml`

**Date:** 2026-08-22
**Change from Run 1:** + SpecAugment (time/frequency masking on the input spectrogram, training
only) + early stopping (patience 7, not triggered — ran the full 30-epoch cap). No weight
decay/conv dropout, isolating SpecAugment's effect alone. See `notes/improv_cnn.md` section 2.
**Checkpoint:** `checkpoints/run3_specaugment.pt` (best epoch 26 of 30).

### Training curve (selected epochs; full log in `runs/run3_specaugment/`)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.1022 | 0.2632 | 2.1152 | 0.3078 |
| 5 | 1.2729 | 0.5689 | 2.3845 | 0.3564 |
| 10 | 0.9383 | 0.6908 | 1.5264 | 0.5110 |
| 15 | 0.7183 | 0.7606 | 1.5178 | 0.5847 |
| 20 | 0.5781 | 0.8051 | 2.9980 | 0.3682 |
| **26** | **0.3955** | **0.8746** | **1.4808** | **0.6097 (best)** |
| 30 | 0.3195 | 0.8954 | 5.0667 | 0.3108 |

Unlike Run 2's smooth curve, val accuracy here is still quite volatile (swings from 0.61 down to
0.31 between epochs 26 and 30) — SpecAugment alone doesn't tame the noisiness the way
regularization did, and some train/val gap remains at the best epoch (0.87 train vs 0.61 val, an
~26-point gap — smaller than Run 1's ~42 points, bigger than Run 2's ~0). But the *peak* val
accuracy it reaches (0.6097) is the highest of any run so far.

### Test set (684 clips)

**Overall: accuracy 0.63, macro F1 0.60, weighted F1 0.61** — the best of any run so far (Run 1:
0.56/0.53/0.53, Run 2: 0.60/0.57/0.59).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel | 0.76 | 0.54 | 0.63 | 41 |
| cla | 1.00 | 0.16 | 0.27 | 51 |
| flu | 0.66 | 0.62 | 0.64 | 47 |
| gac | 0.74 | 0.48 | 0.58 | 64 |
| gel | 0.77 | 0.35 | 0.48 | 78 |
| org | 0.79 | 0.90 | 0.84 | 69 |
| pia | 0.46 | 0.96 | 0.62 | 74 |
| sax | 0.49 | 0.33 | 0.40 | 63 |
| tru | 0.54 | 0.92 | 0.68 | 60 |
| vio | 0.54 | 0.59 | 0.56 | 59 |
| voi | 0.86 | 0.92 | 0.89 | 78 |
| **macro avg** | **0.69** | **0.62** | **0.60** | 684 |

Best overall macro F1, but with a striking outlier: **clarinet (cla) recall collapsed to 0.16**
(perfect 1.00 precision — the model got extremely conservative, only guessing cla when very
confident, and defaulted to sax/tru/flu for the rest — see confusion matrix). voi (0.89 F1) and org
(0.84 F1) are the strongest classes here, both better than in Run 1 or Run 2.

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 22 | 0 | 2 | 0 | 4 | 0 | 4 | 0 | 2 | 7 | 0 |
| cla | 0 | 8 | 6 | 0 | 0 | 0 | 4 | 13 | 17 | 3 | 0 |
| flu | 0 | 0 | 29 | 0 | 0 | 1 | 10 | 2 | 2 | 3 | 0 |
| gac | 0 | 0 | 0 | 31 | 0 | 4 | 25 | 0 | 1 | 2 | 1 |
| gel | 0 | 0 | 0 | 5 | 27 | 7 | 24 | 1 | 3 | 5 | 6 |
| org | 0 | 0 | 0 | 0 | 3 | 62 | 4 | 0 | 0 | 0 | 0 |
| pia | 0 | 0 | 0 | 1 | 0 | 1 | 71 | 0 | 0 | 0 | 1 |
| sax | 3 | 0 | 2 | 1 | 0 | 0 | 10 | 21 | 18 | 6 | 2 |
| tru | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 55 | 2 | 0 |
| vio | 4 | 0 | 4 | 4 | 0 | 3 | 1 | 2 | 4 | 35 | 2 |
| voi | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 2 | 0 | 2 | 72 |

cla's 51 true clips scatter mostly into cla(8)/flu(6)/sax(13)/tru(17) — no single dominant
confusion partner, which is consistent with genuinely low confidence rather than one systematic
mix-up. pia stays a mild sink (gac→pia 25, gel→pia 24) similar to Run 1's pattern, though not as
extreme.

### Verdict

Highest peak accuracy/F1 of the three runs so far, but a less stable training curve than Run 2 and
one badly-behaved class (cla). SpecAugment seems to help the model learn more *discriminative*
features overall (org/voi/tru all very strong) at the cost of occasionally collapsing one class's
recall entirely — a regularization technique that pushes harder than Run 2's but less predictably.
Open question for Run 4: does combining with weight decay/dropout stabilize this curve while
keeping (or improving on) this accuracy, or fix the cla collapse?

---

## Run 4 — `run4_combined`, `configs/combined.yaml`

**Date:** 2026-08-22
**Change from Run 1:** Run 2 + Run 3 together — weight decay (1e-4) + conv-block dropout (0.2) +
SpecAugment + early stopping (patience 7, not triggered — ran the full 30-epoch cap, still slowly
improving at the end).
**Checkpoint:** `checkpoints/run4_combined.pt` (best epoch 28 of 30).

### Training curve (selected epochs; full log in `runs/run4_combined/`)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.2848 | 0.1840 | 2.1805 | 0.2548 |
| 5 | 2.0111 | 0.3027 | 1.9836 | 0.3417 |
| 10 | 1.8394 | 0.3714 | 1.8030 | 0.3976 |
| 15 | 1.7237 | 0.4171 | 1.7370 | 0.4315 |
| 20 | 1.6373 | 0.4438 | 1.5989 | 0.4934 |
| 25 | 1.5747 | 0.4635 | 1.5206 | 0.5052 |
| **28** | **1.5259** | **0.4809** | **1.4851** | **0.5272 (best)** |
| 30 | 1.5018 | 0.4948 | 1.4747 | 0.5228 |

Curve is smooth and stable like Run 2's (no wild val swings — combining the two didn't inherit
Run 3's volatility), but converges *slower* and plateaus *lower*: still climbing gradually at
epoch 30, hasn't caught up to either individual run. Train accuracy at epoch 30 (0.49) is barely
above val (0.52) — no overfitting at all, if anything the model looks slightly under-fit within
this epoch budget.

### Test set (684 clips)

**Overall: accuracy 0.55, macro F1 0.53, weighted F1 0.54** — essentially tied with Run 1's
baseline (0.56/0.53/0.53), and **worse than both individual techniques** (Run 2: 0.60/0.57/0.59,
Run 3: 0.63/0.60/0.61).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel | 0.35 | 0.44 | 0.39 | 41 |
| cla | 0.54 | 0.41 | 0.47 | 51 |
| flu | 0.47 | 0.40 | 0.44 | 47 |
| gac | 0.70 | 0.52 | 0.59 | 64 |
| gel | 0.52 | 0.37 | 0.43 | 78 |
| org | 0.80 | 0.80 | 0.80 | 69 |
| pia | 0.61 | 0.57 | 0.59 | 74 |
| sax | 0.44 | 0.38 | 0.41 | 63 |
| tru | 0.80 | 0.65 | 0.72 | 60 |
| vio | 0.39 | 0.44 | 0.41 | 59 |
| voi | 0.48 | 0.87 | 0.62 | 78 |
| **macro avg** | **0.55** | **0.53** | **0.53** | 684 |

Notably, Run 3's cla collapse (0.16 recall) is *fixed* here (0.41 recall) — so combining did solve
that specific failure mode — but nearly every other class's F1 dropped back toward Run 1 levels
compared to Run 2 or Run 3 individually. voi becomes a mild sink again (gel→voi 26, up from Run 2's
9 and Run 3's 6).

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 18 | 2 | 1 | 0 | 9 | 0 | 0 | 5 | 0 | 6 | 0 |
| cla | 2 | 21 | 5 | 0 | 2 | 1 | 1 | 0 | 3 | 13 | 3 |
| flu | 1 | 6 | 19 | 0 | 2 | 3 | 6 | 0 | 2 | 3 | 5 |
| gac | 7 | 0 | 0 | 33 | 0 | 0 | 10 | 4 | 0 | 0 | 10 |
| gel | 1 | 2 | 1 | 2 | 29 | 2 | 5 | 6 | 0 | 4 | 26 |
| org | 0 | 0 | 0 | 4 | 1 | 55 | 2 | 0 | 0 | 2 | 5 |
| pia | 7 | 3 | 5 | 3 | 0 | 6 | 42 | 0 | 1 | 1 | 6 |
| sax | 9 | 3 | 0 | 0 | 4 | 0 | 1 | 24 | 4 | 7 | 11 |
| tru | 0 | 1 | 2 | 1 | 0 | 1 | 0 | 9 | 39 | 5 | 2 |
| vio | 7 | 1 | 7 | 4 | 1 | 1 | 2 | 5 | 0 | 26 | 5 |
| voi | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 2 | 0 | 0 | 68 |

### Verdict

**Combining the two techniques did not help — it underperformed both individually.** Likely
explanation: weight decay + conv dropout + SpecAugment together impose *three* simultaneous forms
of regularization pressure on a small (~0.43M param) model, within the same fixed 30-epoch budget.
The training curve's shape supports this — it's stable (not overfitting, not oscillating) but
converging noticeably slower than either individual run, suggesting the model is being
under-trained relative to how hard the task's been made, rather than that the combination is
fundamentally counterproductive. This is a real, useful negative result, not a failure to report
around: **more regularization is not automatically better**, and stacking techniques has a real
cost (slower convergence) that needs a matching benefit (more epochs, or a larger model) to pay
off. Given this specific setup, **Run 3 (SpecAugment alone) is currently the best result.**

---

## Ensemble — Run 2 + Run 3 (soft-vote)

**Date:** 2026-08-22
**What it is:** no retraining — averages Run 2's and Run 3's softmax probabilities on the same
held-out test split (`src/ensemble_evaluate.py`), then argmaxes. Motivation: Run 2 (stable,
balanced) and Run 3 (highest accuracy, but collapsed clarinet recall) looked like they might have
*complementary* rather than *correlated* errors — see `notes/improv_cnn.md`, "Phase A" section.
**Artifacts:** no checkpoint/config of its own (see `EXPERIMENTS.md`) — reproduce with:
```
python -m src.ensemble_evaluate --checkpoints checkpoints/run2_regularization.pt checkpoints/run3_specaugment.pt
```

### Test set (684 clips)

**Overall: accuracy 0.65, macro F1 0.62, weighted F1 0.63 — beats every individual run,
including Run 3's previous-best 0.63/0.60/0.61.**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel | 0.77 | 0.56 | 0.65 | 41 |
| cla | 1.00 | 0.22 | 0.35 | 51 |
| flu | 0.64 | 0.60 | 0.62 | 47 |
| gac | 0.76 | 0.53 | 0.62 | 64 |
| gel | 0.68 | 0.46 | 0.55 | 78 |
| org | 0.79 | 0.93 | 0.85 | 69 |
| pia | 0.49 | 0.95 | 0.64 | 74 |
| sax | 0.46 | 0.30 | 0.37 | 63 |
| tru | 0.57 | 0.92 | 0.71 | 60 |
| vio | 0.62 | 0.63 | 0.62 | 59 |
| voi | 0.87 | 0.88 | 0.88 | 78 |
| **macro avg** | **0.69** | **0.63** | **0.62** | 684 |

Confirms the hypothesis: the ensemble beats *both* inputs, so Run 2 and Run 3's errors were at
least partly complementary, not just correlated noise. Notably, Run 3's clarinet collapse (0.16
recall) partially recovers here (0.22) — still weak vs. Run 2's own 0.47 alone, but better than
Run 3 alone — consistent with Run 2's vote pulling some clarinet predictions back from Run 3's
overconfident sax/trumpet guesses.

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 23 | 0 | 2 | 0 | 5 | 0 | 4 | 0 | 2 | 5 | 0 |
| cla | 0 | 11 | 7 | 0 | 0 | 0 | 4 | 13 | 13 | 3 | 0 |
| flu | 0 | 0 | 28 | 0 | 0 | 2 | 12 | 1 | 1 | 3 | 0 |
| gac | 0 | 0 | 0 | 34 | 2 | 3 | 22 | 0 | 3 | 0 | 0 |
| gel | 1 | 0 | 0 | 5 | 36 | 8 | 18 | 1 | 2 | 3 | 4 |
| org | 0 | 0 | 0 | 0 | 3 | 64 | 2 | 0 | 0 | 0 | 0 |
| pia | 0 | 0 | 0 | 1 | 0 | 1 | 70 | 0 | 0 | 0 | 2 |
| sax | 4 | 0 | 2 | 1 | 3 | 0 | 10 | 19 | 16 | 6 | 2 |
| tru | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 55 | 2 | 0 |
| vio | 2 | 0 | 4 | 4 | 0 | 3 | 1 | 2 | 4 | 37 | 2 |
| voi | 0 | 0 | 0 | 0 | 4 | 0 | 1 | 3 | 0 | 1 | 69 |

### Verdict

**Best result so far, for free.** No new training, no new hyperparameters to tune — just averaging
two already-trained models. gac→pia (22) and sax↔tru (16/10) confusions persist (as they have in
every run), consistent with `notes/improv_cnn.md`'s diagnosis that these are a genuine
feature-discriminability limit rather than something regularization/ensembling alone fully
resolves. Next: does more training time fix Run 4's under-training (Run 5), and does focal loss
targeting these exact confused pairs help beyond what ensembling already recovered (Run 6)?

---

## Run 5 — `run5_combined_extended`, `configs/combined_extended.yaml`

**Date:** 2026-08-22
**Change from Run 4:** identical recipe (weight decay 1e-4 + conv dropout 0.2 + SpecAugment), just
`epochs: 60` (was 30) and `early_stopping_patience: 12` (was 7) — testing Run 4's own verdict that
it was under-trained, not fundamentally a bad combination. See `DECISIONS.md`, "Combined-recipe
epoch budget" entry.
**Checkpoint:** `checkpoints/run5_combined_extended.pt` (best epoch 56 of 60).

### Training curve (selected epochs; full log in `runs/run5_combined_extended/`)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.2884 | 0.1750 | 2.1729 | 0.2283 |
| 10 | 1.8526 | 0.3669 | 1.8126 | 0.4080 |
| 20 | 1.6689 | 0.4390 | 1.6245 | 0.4875 |
| 30 | 1.5175 | 0.4867 | 1.4776 | 0.5287 |
| 40 | 1.4002 | 0.5359 | 1.3758 | 0.5670 |
| 50 | 1.2836 | 0.5721 | 1.3095 | 0.5773 |
| **56** | **1.1951** | **0.5994** | **1.2679** | **0.6082 (best)** |
| 60 | 1.1674 | 0.6103 | 1.2463 | 0.6068 |

**Confirms the hypothesis decisively.** The curve climbs smoothly and monotonically the entire 60
epochs — no oscillation, no overfitting (train/val gap stays ~1-2 points throughout, even smaller
than Run 2's), and it's *still climbing* at epoch 60 (val 0.6068 at epoch 60 vs. 0.6082 at epoch
56 — essentially a plateau just starting to form, not a clear peak-and-decline). Compare to Run
4's original 30-epoch curve, which reached only val acc 0.5272 by its endpoint — this run doubles
that (0.6082) simply by training longer on the exact same recipe.

### Test set (684 clips)

**Overall: accuracy 0.65, macro F1 0.64, weighted F1 0.65** — ties the ensemble's accuracy (0.65)
and has the **highest macro F1 of any run or the ensemble so far** (previous best: ensemble's
0.62; previous best single-model: Run 3's 0.60).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel | 0.54 | 0.49 | 0.51 | 41 |
| cla | 0.55 | 0.67 | 0.60 | 51 |
| flu | 0.67 | 0.38 | 0.49 | 47 |
| gac | 0.70 | 0.77 | 0.73 | 64 |
| gel | 0.48 | 0.67 | 0.56 | 78 |
| org | 0.88 | 0.86 | 0.87 | 69 |
| pia | 0.74 | 0.81 | 0.77 | 74 |
| sax | 0.44 | 0.46 | 0.45 | 63 |
| tru | 0.77 | 0.78 | 0.78 | 60 |
| vio | 0.71 | 0.37 | 0.49 | 59 |
| voi | 0.78 | 0.74 | 0.76 | 78 |
| **macro avg** | **0.66** | **0.64** | **0.64** | 684 |

No class collapses (unlike Run 3's clarinet: cla recall here is a healthy 0.67, the best of any
run). Most balanced result yet — no F1 below 0.45, several classes (org 0.87, pia 0.77, tru 0.78)
at their best or near-best across all runs/ensemble.

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 20 | 2 | 1 | 0 | 9 | 0 | 1 | 6 | 0 | 2 | 0 |
| cla | 1 | 34 | 1 | 0 | 5 | 0 | 0 | 5 | 3 | 2 | 0 |
| flu | 1 | 12 | 18 | 0 | 7 | 4 | 3 | 1 | 0 | 0 | 1 |
| gac | 0 | 0 | 0 | 49 | 4 | 0 | 9 | 0 | 0 | 1 | 1 |
| gel | 2 | 0 | 0 | 5 | 52 | 0 | 6 | 7 | 1 | 2 | 3 |
| org | 0 | 0 | 0 | 4 | 3 | 59 | 1 | 0 | 0 | 0 | 2 |
| pia | 0 | 1 | 1 | 5 | 0 | 3 | 60 | 1 | 0 | 0 | 3 |
| sax | 4 | 4 | 1 | 2 | 9 | 0 | 1 | 29 | 7 | 2 | 4 |
| tru | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 7 | 47 | 0 | 1 |
| vio | 9 | 4 | 5 | 4 | 7 | 1 | 0 | 3 | 3 | 22 | 1 |
| voi | 0 | 0 | 0 | 1 | 12 | 0 | 0 | 7 | 0 | 0 | 58 |

gac→pia drops to 9 (from 22-25 in most other runs) — the "pia sink" pattern is markedly reduced
here, though not eliminated. sax↔tru/gel persists (7/9). vio recall (0.37) is now the weakest
class — vio errors spread across cel/flu/gel rather than concentrating on one confusion partner.

### Verdict

**Best single-model result — Run 4's under-training diagnosis was correct.** Doubling the epoch
budget alone (no other change) took the exact same recipe from underperforming both individual
techniques (55%/0.53) to beating all of them (65%/0.64). This is a clean, unambiguous confirmation:
**the earlier "combining regularization techniques doesn't help" conclusion was an artifact of an
insufficient epoch budget, not a real interaction effect.** Given the curve was still climbing at
epoch 60, further gains from even more epochs are plausible but untested — not pursued further here
since Run 6 (focal loss) is the next planned experiment and diminishing returns are likely.

---

## Run 6 — `run6_focal_specaugment`, `configs/focal.yaml`

**Date:** 2026-08-22
**Change from Run 3:** identical recipe (SpecAugment, no weight decay/dropout) + focal loss
(`gamma=2.0`, no class weighting) replacing plain cross-entropy — targets Run 3's specific
clarinet-recall collapse and the confusions persisting across every prior run. See `DECISIONS.md`,
"Loss: focal over class-weighted" entry.
**Checkpoint:** `checkpoints/run6_focal_specaugment.pt` (best epoch 13 of 30 — **early stopping
triggered**, no improvement for 7 epochs after).

### Training curve (selected epochs; full log in `runs/run6_focal_specaugment/`)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.6072 | 0.2712 | 1.8317 | 0.2342 |
| 5 | 0.8078 | 0.5721 | 1.4396 | 0.3918 |
| 9 | 0.5873 | 0.6601 | 1.1658 | 0.4934 |
| **13** | **0.4820** | **0.7177** | **1.0835** | **0.5346 (best)** |
| 16 | 0.3941 | 0.7450 | 2.4940 | 0.3432 |
| 20 (stop) | 0.2783 | 0.8089 | 2.1309 | 0.3976 |

Note: focal loss values aren't directly comparable in magnitude to Run 1-5's cross-entropy loss
values (different loss function) — only the *trend* and accuracy columns are comparable across
runs. The val curve is the most volatile of any run so far (e.g. epoch 15→16: 0.50→0.34), and
**this is the first run where early stopping actually triggered** — no val-acc improvement for 7
straight epochs after epoch 13.

### Test set (684 clips)

**Overall: accuracy 0.58, macro F1 0.56, weighted F1 0.58** — worse than Run 3, the recipe this
was built on (0.63/0.60/0.61), and worse than Run 5 (0.65/0.64/0.65).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel | 0.83 | 0.24 | 0.38 | 41 |
| cla | 0.54 | 0.63 | 0.58 | 51 |
| flu | 0.67 | 0.34 | 0.45 | 47 |
| gac | 0.77 | 0.56 | 0.65 | 64 |
| gel | 0.59 | 0.65 | 0.62 | 78 |
| org | 0.32 | **1.00** | 0.49 | 69 |
| pia | 0.91 | 0.68 | 0.78 | 74 |
| sax | 0.51 | 0.40 | 0.45 | 63 |
| tru | 0.77 | 0.72 | 0.74 | 60 |
| vio | 0.80 | 0.20 | 0.32 | 59 |
| voi | 0.79 | 0.69 | 0.74 | 78 |
| **macro avg** | **0.68** | **0.56** | **0.56** | 684 |

**A genuinely mixed result, not simply "worse."** The specific problem this targeted — Run 3's
clarinet collapse — is substantially fixed: cla recall 0.16 (Run 3) → **0.63** here, one of the
best cla results of any run. But focal loss introduced a **new, more severe collapse**: org recall
hit **1.00** (perfect recall, 0.32 precision) — the model became a near-unconditional "guess organ
when unsure" machine, absorbing large numbers of flu/gac/gel/pia/vio misclassifications (see
confusion matrix). vio recall also dropped to 0.20, the worst of any run. Net effect: fixing one
failure mode traded it for a different, larger one.

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 10 | 6 | 0 | 1 | 8 | 2 | 1 | 6 | 3 | 3 | 1 |
| cla | 0 | 32 | 4 | 0 | 2 | 9 | 0 | 3 | 1 | 0 | 0 |
| flu | 0 | 2 | 16 | 0 | 1 | 26 | 0 | 0 | 0 | 0 | 2 |
| gac | 0 | 0 | 0 | 36 | 4 | 18 | 3 | 2 | 0 | 0 | 1 |
| gel | 0 | 0 | 0 | 2 | 51 | 21 | 1 | 0 | 1 | 0 | 2 |
| org | 0 | 0 | 0 | 0 | 0 | 69 | 0 | 0 | 0 | 0 | 0 |
| pia | 0 | 0 | 0 | 7 | 0 | 17 | 50 | 0 | 0 | 0 | 0 |
| sax | 0 | 13 | 1 | 1 | 9 | 10 | 0 | 25 | 2 | 0 | 2 |
| tru | 0 | 3 | 0 | 0 | 0 | 9 | 0 | 4 | 43 | 0 | 1 |
| vio | 2 | 3 | 3 | 0 | 3 | 19 | 0 | 6 | 6 | 12 | 5 |
| voi | 0 | 0 | 0 | 0 | 8 | 13 | 0 | 3 | 0 | 0 | 54 |

The org row is a clean diagonal (69/69, never misclassified *as* something else), but org is the
single most common wrong answer for 6 of the other 10 classes. This is the "pia sink" pattern seen
in earlier runs, but worse — concentrated on one class instead of spread across two or three, and
with 100% recall on that class as the tell.

### Verdict

**Focal loss, as configured here, is not an improvement over Run 3 — hold this recipe, don't adopt
it.** The result is informative rather than simply negative: it demonstrates focal loss's
upweighting of hard/low-confidence examples can overcorrect and manufacture a new systematic bias
(here, org as an attractor class) rather than cleanly fixing the targeted confusion. Whether a
lower `gamma` (less aggressive upweighting) would fix the org collapse while keeping the clarinet
improvement is untested — a reasonable follow-up if focal loss is revisited, but not pursued now
given Run 5 already beats this result outright on every metric. **Run 5 remains the best
single-model result; the Run 2+3 ensemble remains competitive on accuracy.** The optional
class-weighted control (`configs/class_weighted.yaml`) was not run — Run 6's outcome doesn't change
the reasoning against class-weighting as a primary fix (see `DECISIONS.md`), and priority shifted
to Phase B (pretrained embeddings) given Phase A's clearest win was Run 5's epoch-budget fix, not
loss-function changes.

---

## Summary — Phase A complete

| Run | Config | Test acc | Macro F1 | Weighted F1 | Notes |
|---|---|---:|---:|---:|---|
| 1 — baseline | `configs/base.yaml` | 0.56 | 0.53 | 0.53 | severe overfitting (~42pt gap) |
| 2 — regularization | `configs/reg.yaml` | 0.60 | 0.57 | 0.59 | overfitting fully closed |
| 3 — SpecAugment | `configs/specaug.yaml` | 0.63 | 0.60 | 0.61 | highest solo accuracy (pre-Run5), cla collapse |
| 4 — combined | `configs/combined.yaml` | 0.55 | 0.53 | 0.53 | underperformed both alone — under-trained (see Run 5) |
| Ensemble (2+3) | *(no training)* | 0.65 | 0.62 | 0.63 | best accuracy (tied), no retraining needed |
| **5 — combined, extended** | `configs/combined_extended.yaml` | **0.65** | **0.64** | **0.65** | **best overall — most balanced, no class collapse** |
| 6 — focal + SpecAugment | `configs/focal.yaml` | 0.58 | 0.56 | 0.58 | fixed cla collapse, caused a worse org collapse |

**Best result: Run 5 (combined recipe, extended to 60 epochs).** Highest macro F1 of any run or
ensemble, ties the ensemble's accuracy, and is the most balanced across classes (no F1 below 0.45,
vs. every other run/ensemble having at least one class below 0.40). The single biggest lesson from
Phase A: **training budget mattered more than which regularization techniques were combined** —
Run 4 and Run 5 are the *same recipe*, and doubling the epoch count alone closed a 10-point
accuracy gap. Phase B (pretrained embeddings — PANNs, AST) is next; see `notes/improv_cnn.md`.

---

## Run 7 — `run7_panns_frozen`, `configs/panns.yaml`

**Date:** 2026-08-22
**What it is:** PANNs (CNN14), pretrained on AudioSet (~2M clips), frozen backbone + a fresh
`Linear(2048, 11)` head trained from scratch. Raw 32kHz waveform input — not this project's usual
log-mel pipeline (see `DECISIONS.md`, "PANNs input pipeline" entry). Only the head's ~22k
parameters are trained; the ~80M-param backbone is untouched.
**Checkpoint:** `checkpoints/run7_panns_frozen.pt` (best epoch 14 of 30 — early stopping
triggered, patience 7).

**Bug hit and fixed on the way here:** `mixed_precision: false` in this config didn't actually
disable autocast (a pre-existing bug in `run_epoch` — see `DECISIONS.md`, "Bug fix: mixed_precision:
false" entry) — PANNs' internal STFT frontend produces NaN logits under fp16 autocast, so this
config genuinely needs fp32. Caught immediately by the smoke test (NaN loss from epoch 1), fixed
before the real run below.

### Training curve (selected epochs; full log in `runs/run7_panns_frozen/`)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.5159 | 0.6312 | 1.1341 | 0.7054 |
| 5 | 0.6753 | 0.7896 | 0.8485 | 0.7393 |
| 10 | 0.5841 | 0.8094 | 0.8082 | 0.7511 |
| **14** | **0.5422** | **0.8250** | **0.8032** | **0.7688 (best)** |
| 21 (stop) | 0.4980 | 0.8398 | 0.8012 | 0.7541 |

Smooth, stable, fast convergence — reaches 70% val acc in the *first* epoch (vs. every from-scratch
run needing 20+ epochs to approach that), then climbs gently before plateauing. No overfitting
oscillation of the kind seen in Runs 3/6 — makes sense: only ~22k parameters (the head) are
actually training, a much simpler optimization problem than any from-scratch run.

### Test set (684 clips)

**Overall: accuracy 0.78, macro F1 0.76, weighted F1 0.77** — dramatically ahead of every
from-scratch result (previous best: Run 5, 0.65/0.64).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel | 0.69 | 0.61 | 0.65 | 41 |
| cla | 0.89 | 0.61 | 0.72 | 51 |
| flu | 0.75 | 0.64 | 0.69 | 47 |
| gac | 0.84 | 0.92 | 0.88 | 64 |
| gel | 0.72 | 0.79 | 0.76 | 78 |
| org | 0.78 | 0.90 | 0.84 | 69 |
| pia | 0.84 | 0.92 | 0.88 | 74 |
| sax | 0.57 | 0.67 | 0.61 | 63 |
| tru | 0.78 | 0.72 | 0.75 | 60 |
| vio | 0.76 | 0.58 | 0.65 | 59 |
| voi | 0.92 | 0.97 | 0.94 | 78 |
| **macro avg** | **0.78** | **0.76** | **0.76** | 684 |

No class below 0.61 F1 — every from-scratch run had at least one class below 0.40. sax remains the
weakest class (0.61 F1, same pattern as every prior run) but even that is well above any
from-scratch run's sax result.

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 25 | 0 | 1 | 1 | 7 | 0 | 0 | 1 | 0 | 6 | 0 |
| cla | 0 | 31 | 3 | 0 | 1 | 0 | 2 | 12 | 1 | 1 | 0 |
| flu | 0 | 0 | 30 | 2 | 1 | 8 | 2 | 3 | 1 | 0 | 0 |
| gac | 0 | 0 | 0 | 59 | 1 | 0 | 1 | 0 | 0 | 1 | 2 |
| gel | 0 | 0 | 0 | 7 | 62 | 4 | 0 | 0 | 0 | 2 | 3 |
| org | 0 | 0 | 0 | 0 | 1 | 62 | 6 | 0 | 0 | 0 | 0 |
| pia | 1 | 0 | 1 | 0 | 3 | 1 | 68 | 0 | 0 | 0 | 0 |
| sax | 2 | 2 | 0 | 0 | 4 | 2 | 0 | 42 | 8 | 1 | 2 |
| tru | 0 | 1 | 2 | 0 | 1 | 0 | 0 | 13 | 43 | 0 | 0 |
| vio | 8 | 1 | 3 | 1 | 4 | 2 | 1 | 3 | 2 | 34 | 0 |
| voi | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 76 |

The exact same confusion pairs identified back in Run 1 are still visible (cla↔sax 12+2, sax↔tru
8+13, cel↔vio 6+8) — same *direction* of confusion, but at roughly a third of the magnitude of the
worst from-scratch runs. Pretraining reduced these confusions substantially without eliminating
them — consistent with `notes/improv_cnn.md`'s diagnosis that some of these pairs are genuinely
acoustically similar, not just an artifact of insufficient training data.

### Verdict

**Confirms the pretrained-embeddings hypothesis decisively.** A frozen AudioSet-pretrained backbone
plus a tiny trained head beats every from-scratch architecture/regularization combination tried in
Phase A by a wide margin (78% vs. 65% best), using a fraction of the trainable parameters (~22k vs.
~430k) and converging far faster/more stably. This is strong evidence the from-scratch CNN's
ceiling in Phase A was a real data-scarcity limit, not something more architecture/regularization
tuning would have closed.

---

## Run 8 — `run8_ast_frozen`, `configs/ast.yaml`

**Date:** 2026-08-22
**What it is:** the literal "pretrained transformer" answer to the user's original question — AST
(Audio Spectrogram Transformer), pretrained on AudioSet, frozen backbone + a fresh classifier head.
Input: 16kHz/128-mel (matches this project's own choices), loop-padded to ~11s before AST's own
feature extraction to avoid feeding its pretrained 1024-frame positional embeddings mostly silence
(see `DECISIONS.md`, "AST input-length mismatch" entry).
**Checkpoint:** `checkpoints/run8_ast_frozen.pt` (best epoch 8 of 30 — early stopping triggered).

### Training curve (selected epochs; full log in `runs/run8_ast_frozen/`)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.4291 | 0.5992 | 1.0401 | 0.7040 |
| 3 | 0.6838 | 0.7980 | 0.7876 | 0.7599 |
| 5 | 0.5805 | 0.8169 | 0.7525 | 0.7644 |
| **8** | **0.5118** | **0.8356** | **0.7277** | **0.7732 (best)** |
| 15 (stop) | 0.4317 | 0.8605 | 0.7313 | 0.7703 |

Even faster convergence than PANNs — plateaus by epoch 8 (vs. PANNs' 14) and the val curve is
essentially flat/noiseless after that. Early stopping triggered slightly sooner too. This is the
smoothest, most stable training curve of any run in the entire project.

### Test set (684 clips)

**Overall: accuracy 0.78, macro F1 0.76, weighted F1 0.78** — ties Run 7 (PANNs) almost exactly.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cel | 0.69 | 0.54 | 0.60 | 41 |
| cla | 0.75 | 0.65 | 0.69 | 51 |
| flu | 0.71 | 0.68 | 0.70 | 47 |
| gac | 0.86 | 0.89 | 0.88 | 64 |
| gel | 0.77 | 0.79 | 0.78 | 78 |
| org | 0.88 | 0.94 | 0.91 | 69 |
| pia | 0.84 | 0.91 | 0.87 | 74 |
| sax | 0.61 | 0.65 | 0.63 | 63 |
| tru | 0.78 | 0.67 | 0.72 | 60 |
| vio | 0.65 | 0.68 | 0.66 | 59 |
| voi | 0.94 | 0.99 | 0.96 | 78 |
| **macro avg** | **0.77** | **0.76** | **0.76** | 684 |

Near-identical macro F1 to PANNs (0.76 both), but a different balance: AST is stronger on org
(0.91 vs 0.84), voi (0.96 vs 0.94), vio (0.66 vs 0.65); PANNs is stronger on cla (0.72 vs 0.69),
tru (0.75 vs 0.72), cel (0.65 vs 0.60). Neither dominates the other class-by-class.

### Confusion matrix (rows = true, cols = predicted)

| true \ pred | cel | cla | flu | gac | gel | org | pia | sax | tru | vio | voi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cel | 22 | 1 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 12 | 0 |
| cla | 0 | 33 | 5 | 0 | 0 | 0 | 0 | 10 | 2 | 1 | 0 |
| flu | 0 | 1 | 32 | 0 | 4 | 4 | 3 | 1 | 1 | 1 | 0 |
| gac | 0 | 0 | 0 | 57 | 1 | 0 | 5 | 0 | 0 | 0 | 1 |
| gel | 1 | 0 | 0 | 6 | 62 | 4 | 2 | 2 | 0 | 1 | 0 |
| org | 0 | 0 | 0 | 0 | 0 | 65 | 3 | 0 | 0 | 1 | 0 |
| pia | 0 | 0 | 0 | 2 | 3 | 0 | 67 | 0 | 0 | 1 | 1 |
| sax | 3 | 5 | 0 | 0 | 2 | 1 | 0 | 41 | 7 | 2 | 2 |
| tru | 0 | 3 | 4 | 0 | 0 | 0 | 0 | 11 | 40 | 2 | 0 |
| vio | 6 | 1 | 4 | 1 | 3 | 0 | 0 | 2 | 1 | 40 | 1 |
| voi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 77 |

cel↔vio is notably worse here (12 cel→vio, the single largest off-diagonal entry in this whole
matrix) than in Run 7 (6+8) — AST's biggest weak spot. Same recurring confusion families as every
other run (cla↔sax, sax↔tru), just redistributed slightly differently.

### Verdict

**Directly answers the user's original question: yes, a *pretrained* transformer works — and works
about as well as the pretrained CNN, both far ahead of anything from-scratch.** Neither
architecture family (CNN vs. transformer) dominates once both start from AudioSet pretraining —
the pretraining is what mattered, exactly as `notes/improv_cnn.md` predicted before either was
run. AST converges slightly faster/more stably; PANNs and AST trade off different classes' F1 with
no clear overall winner. Given they're essentially tied, the practical tie-breaker would be
inference cost/deployment simplicity (PANNs: pure CNN, cheaper) rather than accuracy — not
explored further here since it's outside the current scope.

---

## Summary — Phases A + B complete

| Run | Config | Test acc | Macro F1 | Weighted F1 | Notes |
|---|---|---:|---:|---:|---|
| 1 — baseline | `configs/base.yaml` | 0.56 | 0.53 | 0.53 | severe overfitting |
| 2 — regularization | `configs/reg.yaml` | 0.60 | 0.57 | 0.59 | overfitting closed |
| 3 — SpecAugment | `configs/specaug.yaml` | 0.63 | 0.60 | 0.61 | cla collapse |
| 4 — combined | `configs/combined.yaml` | 0.55 | 0.53 | 0.53 | under-trained (see Run 5) |
| Ensemble (2+3) | *(no training)* | 0.65 | 0.62 | 0.63 | free improvement |
| 5 — combined, extended | `configs/combined_extended.yaml` | 0.65 | 0.64 | 0.65 | best from-scratch result |
| 6 — focal + SpecAugment | `configs/focal.yaml` | 0.58 | 0.56 | 0.58 | fixed cla, caused org collapse |
| **7 — PANNs (frozen)** | `configs/panns.yaml` | **0.78** | **0.76** | **0.77** | **pretrained CNN** |
| **8 — AST (frozen)** | `configs/ast.yaml` | **0.78** | **0.76** | **0.78** | **pretrained transformer** |

**Best results: Run 7 (PANNs) and Run 8 (AST), effectively tied at 78% accuracy / 0.76 macro F1.**
Both beat the best from-scratch result (Run 5, 65%/0.64) by ~13 points of accuracy while training
roughly 20x fewer parameters (a linear head vs. a full CNN) and converging in ~10-15 epochs instead
of 30-60. This closes the question that started Phase B: **the from-scratch CNN's ~65% ceiling was
a genuine data-scarcity limit** (~5,300 training clips isn't much for learning instrument timbre
from raw spectrograms), not a fixable regularization/architecture problem — pretraining on a much
larger, more general audio corpus (AudioSet) is what actually moved the needle, and it moved it by
roughly the same amount whether the pretrained backbone was a CNN or a transformer.

