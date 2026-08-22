# Results Log

**Purpose:** a running record of every training run's actual numbers, so later variations (see
`notes/improv_cnn.md`) have something concrete to compare against. New runs go at the top.

Each entry: config summary, per-epoch curve, held-out test-set metrics (accuracy/per-class
precision-recall-F1/confusion matrix), and a one-line verdict.

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

## Summary across all four runs

| Run | Config | Test acc | Macro F1 | Weighted F1 | Train/val gap at best epoch |
|---|---|---:|---:|---:|---|
| 1 — baseline | `configs/base.yaml` | 0.56 | 0.53 | 0.53 | ~42 pts (severe overfitting) |
| 2 — regularization | `configs/reg.yaml` | 0.60 | 0.57 | 0.59 | ~0 pts (none) |
| **3 — SpecAugment** | `configs/specaug.yaml` | **0.63** | **0.60** | **0.61** | ~26 pts (reduced) |
| 4 — combined | `configs/combined.yaml` | 0.55 | 0.53 | 0.53 | ~4 pts (none, but under-trained) |

**Best result: Run 3 (SpecAugment alone), 63% test accuracy / 0.60 macro F1.** Run 2 is the most
*stable* (zero overfitting, smoothest curve) but Run 3 reaches higher peak accuracy despite more
volatility. Run 4's failure to beat either individual technique is itself the most informative
result for future iteration: don't assume stacking regularization techniques helps without
budgeting more training time for the harder optimization problem it creates.

