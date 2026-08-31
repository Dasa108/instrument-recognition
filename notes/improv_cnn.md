# Improving the Baseline CNN — Options

**What this file is:** a menu of techniques for improving on `results.md`'s Run 1, explained with
intuition first (same style as `THEORY_NOTES.md`). No code here, no decisions made — this is the
reference to read before we pick what to try next. Each option ends with a one-line verdict on
whether it's likely to help *our specific problem*.

**Our specific problem, restated:** Run 1 hit 95%+ train accuracy but only ~53% val / 56% test
accuracy — a ~40-point gap. The model is **overfitting**: memorizing details of the 5,342 training
clips rather than learning general instrument-timbre features. That diagnosis should drive which
of the options below are worth trying first — techniques that fight overfitting are the highest
priority; techniques that add model capacity are *not*, since the model already has more than
enough capacity to memorize the training set.

---

## 1. Regularization — techniques that directly fight overfitting

**Intuition:** overfitting happens when the model has an easy path to "cheat" — memorize specific
training examples — instead of the harder path of learning general patterns. Regularization makes
the cheating path harder, forcing the model toward genuinely general features.

- **Weight decay (L2 regularization).** Adds a penalty for large weight values during training.
  Analogy: it's like a tax on complexity — the model can still fit the data, but only if the fit is
  "worth" the tax, which discourages the kind of large, sharply-tuned weights that memorization
  needs. Run 1 uses **zero** weight decay — this is usually the first knob to turn.
- **More dropout.** Run 1 only has dropout (0.3) in the final dense layer, none in the
  convolutional blocks. Dropout randomly zeroes out some activations during training, forcing the
  network to not rely too heavily on any single feature detector. Adding light dropout between
  conv blocks (not just the head) is a common baseline improvement.
- **Early stopping.** Run 1 already tracks the *best* val-accuracy checkpoint (epoch 23), but kept
  training to epoch 30 anyway — wasted compute, and the noisy late-training val curve (epoch 24
  dropped to 29% val acc) suggests training past the best point isn't helping. Stopping training
  once val accuracy hasn't improved for N epochs (a "patience" of e.g. 5) would save time with no
  downside, though it doesn't fix the underlying overfitting by itself.
- **Label smoothing.** Instead of training the model to output 100% confidence on the correct
  class, train it toward e.g. 90% — discourages the extremely confident (and often overfit)
  predictions that come from chasing the last few percent of training accuracy.

**Verdict: high priority.** Directly targets the diagnosed problem, cheap to try (weight decay and
dropout are one-line config changes), and doesn't require new data or architecture work.

---

## 2. Data augmentation — make the training set effectively bigger/more varied

**Intuition:** overfitting is partly a *data* problem, not just a model problem — 5,342 clips is
small by deep-learning standards. Augmentation creates realistic variations of existing clips on
the fly, so the model sees a different-looking version of roughly the same sound each epoch instead
of memorizing the exact same 5,342 spectrograms 30 times over.

- **SpecAugment (time/frequency masking).** The standard technique for spectrogram inputs
  specifically: randomly blank out a few horizontal (frequency) or vertical (time) strips of the
  log-mel spectrogram before feeding it to the model. Forces the model to not over-rely on any one
  narrow frequency band or time window — directly analogous to dropout, but applied to the input
  rather than internal activations. Cheap (pure numpy/tensor ops on the spectrogram, no audio
  re-processing needed) and purpose-built for exactly this input type.
- **Waveform-level augmentation** (applied before the log-mel conversion): pitch shift, time
  stretch, additive background noise, gain/volume jitter. More "realistic" than SpecAugment in one
  sense (these are things that actually happen to real recordings), but costlier — each requires
  re-running the STFT/mel pipeline, so it slows down data loading more than SpecAugment does.
- **Mixup.** Blend two training clips (and their labels) together at a random ratio, training the
  model on the blend. Popular in image classification; less obviously suited to single-label
  predominant-instrument classification (blending two different instruments' spectrograms and
  asking for one label is a slightly awkward fit) — would need care in how it's applied here.

**Verdict: high priority.** SpecAugment specifically is close to a free win — cheap to implement,
purpose-built for spectrograms, directly increases effective training-set diversity. Waveform-level
augmentation is a reasonable second step if SpecAugment alone isn't enough.

---

## 3. Learning rate schedule / optimizer tuning

**Intuition:** Run 1 uses a flat learning rate (0.001, Adam) for all 30 epochs. Early in training a
higher rate helps the model move quickly toward a good region; late in training a lower rate helps
it settle precisely instead of bouncing around — which may explain some of the val-accuracy
noisiness in Run 1's later epochs (e.g. epoch 24's val-acc crash to 29%).

- **LR scheduling** (cosine annealing, step decay, or `ReduceLROnPlateau` which lowers the rate
  automatically once val loss stops improving) — smooths out the back half of training.
- **Warmup** — start the LR low and ramp up over the first few epochs, before decaying. More
  common for larger/deeper models than this one; likely lower-value here.
- **Gradient clipping** — caps how large a single update step can be, preventing occasional
  unstable jumps. Could help explain/prevent the val-acc spikes seen in Run 1.

**Verdict: medium priority.** Won't fix overfitting on its own (a well-scheduled LR can still
memorize the training set), but likely to smooth out the noisy val curve and modestly improve the
final number — reasonable to combine with #1/#2 rather than as a standalone fix.

---

## 4. Architecture changes

**Intuition:** these change *how much* and *what kind* of capacity the model has. Since Run 1 is
already overfitting (i.e. has *more* capacity than the data can currently support), most changes
in this category are lower priority right now — adding capacity to an already-overfitting model
usually makes the gap worse, not better. Two exceptions are called out below.

- **Fewer/narrower layers.** Directly reduces capacity — a legitimate anti-overfitting lever, but
  a blunter instrument than regularization/augmentation (#1/#2 target the *behavior* without
  giving up representational power; shrinking the network gives up power outright).
- **Residual (skip) connections.** Help gradient flow in deep networks — more relevant if the
  network were made significantly deeper; at 5 blocks, Run 1 isn't deep enough for this to matter
  much yet.
- **Squeeze-and-excitation / attention blocks.** Let the network learn to weight some feature
  channels more than others. Adds capacity and complexity — same caveat as above, lower priority
  while still overfitting.
- **Pooling strategy.** Run 1 uses global *average* pooling before the head. Global *max* pooling
  (or combining both) can help when the discriminative signal is concentrated in a short, sharp
  moment (e.g. a percussive onset) rather than spread across the whole clip — plausibly relevant
  for the guitar/piano confusion seen in Run 1's confusion matrix, worth a quick try even at
  low-medium priority.

**Verdict: low priority for now** (aside from the pooling-strategy note), until #1/#2 show whether
they've closed the gap. Revisit if the gap closes but accuracy is still capped too low.

---

## 5. Transfer learning (pretrained audio embeddings)

**Intuition:** instead of learning timbre features from scratch on 5,342 clips, start from a model
already trained on a much larger, more general audio dataset (e.g. AudioSet, millions of clips) and
fine-tune just a small classifier head on top. The pretrained model has already learned generally
useful audio features; we're just teaching it to map those features onto our 11 instrument labels.

- **PANNs, YAMNet, VGGish** — the three options `spec.md` Section 5 already named as the escalation
  path if the from-scratch baseline underperforms.
- Two ways to use one: **frozen feature extraction** (pretrained weights untouched, only train the
  new classifier head — fast, low risk of overfitting since very few parameters are trained) or
  **fine-tuning** (also update the pretrained weights, usually at a low learning rate — more
  potential accuracy, more risk of overfitting again since more parameters are trainable).

**Verdict: held in reserve.** This is `spec.md`'s explicitly planned escalation path, but the
overfitting diagnosis suggests trying the cheaper fixes (#1/#2) first — jumping straight to a much
larger pretrained model without addressing the overfitting problem risks just overfitting *that*
model instead, and it's a bigger step (new dependency, different input requirements, longer
iteration cycle) than adjusting the current pipeline.

---

## 6. Split / evaluation mechanics

**Intuition:** some of Run 1's val-accuracy noise might not be true overfitting-in-progress at all
— it could partly be *measurement* noise from a small validation set (679 clips, so each clip is
~0.15% of the val score) combined with class-imbalanced per-epoch swings. Worth separating "the
model is genuinely overfitting" from "our val-accuracy signal is noisy" before over-reacting to any
single epoch's number.

- **k-fold cross-validation** (song-grouped, same leakage-safety logic as the current split) —
  average results over multiple train/val splits instead of trusting one fixed 679-clip val set.
  More robust signal, but ~k× the training time.
- **Class-weighted loss** — Run 1's classes range from 306–622 clips in training; not wildly
  imbalanced, but weighting the loss by inverse class frequency could help the weaker classes
  (sax, violin) without much downside.

**Verdict: medium priority.** Cross-validation is a bigger time investment better suited to
confirming a final result than to fast iteration; class-weighting is cheap and worth trying
alongside #1/#2 given sax/violin are the weakest classes in Run 1.

---

## Summary — priority order given Run 1's specific overfitting gap (superseded below)

1. **Regularization** (weight decay, more dropout, early stopping) — cheap, directly targets the
   diagnosis.
2. **SpecAugment** — cheap, purpose-built, directly increases effective data diversity.
3. **LR scheduling** + **class-weighted loss** — cheap, smooths training and helps weak classes.
4. **Architecture changes** (aside from pooling strategy) and **transfer learning** — held in
   reserve until 1–3 are tried and we can see whether the train/val gap actually closes.

---

## Where things actually stand after Runs 1–6 (2026-08-22)

**What happened:** #1 (regularization, Run 2) and #2 (SpecAugment, Run 3) both worked — Run 2
closed the overfitting gap completely, Run 3 reached the highest single-technique accuracy (63%).
Combining them (Run 4) *underperformed* both individually (55%) — but that turned out to be a red
herring: the same recipe given twice the epoch budget (Run 5, 60 vs 30) jumped to 65%/macro F1
0.64, the best single-model result of the project so far. **The lesson wasn't "don't combine
techniques" — it was "combining techniques makes optimization harder, and that needs a matching
training-time budget, not just the right hyperparameters."**

**A finding this priority list didn't anticipate:** the confusion matrices across every one of
Runs 1–5 show the *same* family-structured mix-ups — clarinet↔sax/trumpet, cello↔violin,
acoustic-guitar→piano — regardless of which regularization approach was used, including Run 2
which has zero overfitting. That's a signal these specific pairs are hard for the *model's
features* to tell apart, not just a symptom of overfitting that regularization was expected to
clean up. Two responses were tried:

- **Ensembling Run 2 + Run 3** (soft-vote, no retraining) — worked: 65% accuracy, 0.62 macro F1,
  beating both inputs. Confirms their errors were at least partly complementary.
- **Focal loss** (Run 6, built on Run 3's recipe) — targeted the confusion pattern directly by
  upweighting hard/low-confidence predictions. Partially worked (clarinet recall 0.16→0.63) but
  manufactured a *new*, worse collapse (organ recall hit 1.00, absorbing errors from 6 other
  classes) — net worse than Run 3. The class-confusion diagnosis was right; this particular fix
  overcorrected. A lower `gamma` is a plausible untested follow-up, not pursued yet.

**Current best: Run 5** (`checkpoints/run5_combined_extended.pt`, 65% accuracy, 0.64 macro F1, no
class F1 below 0.45 — the most balanced result of any run). Full numbers: `results.md`.

### Is this "confidently usable"? A realistic ceiling check

Before reaching for something bigger, it's worth asking whether 65% is actually underperforming or
close to what's achievable given the data. Two things make this task inherently harder than plain
accuracy suggests: real mixes often have more than one audible instrument, and "predominant" was a
judgment call by the original annotators, not an objective ground truth — plus some instrument
pairs are acoustically very similar regardless of model quality (violin/cello, sax/clarinet/flute,
acoustic/electric guitar). Run 5's 65% is a reasonable, not obviously broken, result for the task's
actual difficulty — but "the model's own features may be near a ceiling" is exactly the case for
trying features learned from a much larger, more general audio corpus than IRMAS provides. (See
the next section for verified published numbers, rather than the general impression above.)

### External benchmark comparison — how does this stack up against published IRMAS results?

*(Added 2026-09-01, after the student asked directly how our numbers compare to the wider field —
verified against actual papers, not recalled from general impression.)*

**Published results, both evaluated on IRMAS's official Testing set (2,874 clips):**

| System | Micro F1 | Macro F1 |
|---|---:|---:|
| Han et al. 2017 — the standard CNN baseline this task is measured against | 0.619 | 0.513 |
| Cross-attentive CNN, Dec 2025 — current published SOTA for this task, 0.878M params | 0.64 | 0.57 |

**These numbers are not directly comparable to our 78% test accuracy / 0.76 macro F1** (Runs 7-8) —
not because of model quality, but because the *task itself* is measured differently in three ways:

1. **Ground truth is a set, not a label.** IRMAS's official Testing clips are annotated with *all*
   predominant instruments present (often 2-3) — published systems do multi-label detection
   (independent sigmoid + per-class threshold), not single-label classification (softmax + argmax,
   what we did throughout Phase 1).
2. **Genuinely polyphonic, uncurated recordings.** The Testing set is real songs, not clips
   selected for having one obvious dominant instrument — which is exactly why `DECISIONS.md`'s
   "IRMAS download scope" entry deliberately kept it out of Phase 1's evaluation in the first
   place. Our test split was carved from the Training data instead, which *was* curated that way.
3. **Window-aggregation.** Testing clips run 5-20s; published systems predict per-window and then
   combine across a clip's windows before scoring — a step our fixed-3s-clip evaluation never
   needed.

Read correctly: our 78% is a real result on a real, if easier-by-design, benchmark. The
literature's ~0.51-0.64 macro F1 is measuring a genuinely harder task (detect *every* instrument in
a real mix, not name the one obvious one). Neither number "beats" the other — a fair head-to-head
would mean re-evaluating our trained models against the official Testing set with a proper
multi-label adaptation, which is untried and doubles as the natural on-ramp into Phase 2.

**Where the frontier actually is, beyond PANNs/AST:** both are themselves no longer cutting-edge —
PANNs is from 2019 (~80M params), AST from 2021 (~86M params). Current general-purpose audio
foundation models are larger-scale: **BEATs** (Microsoft, masked acoustic-token pretraining, up to
300M params) and **Dasheng** (1.2B params, 272,356 hours of pretraining audio). No IRMAS-specific
published numbers were found for either, so no clean head-to-head claim can be made here — but if
further gains were wanted beyond Run 7/8, one of these is the natural next escalation, on the same
"bigger pretraining corpus → better features" logic that made PANNs/AST beat the from-scratch CNN
in the first place.

Sources: [Han et al. 2017](https://arxiv.org/pdf/1605.09507), [cross-attentive CNN (2025)](https://doi.org/10.3390/technologies14010003), [PANNs](https://arxiv.org/pdf/1912.10211), [Dasheng](https://arxiv.org/pdf/2406.06992), [pretrained-embedding sensitivity study](https://arxiv.org/pdf/2501.15900).

### Should we use a transformer? (asked directly)

**Not from scratch.** Transformers lack a CNN's built-in inductive biases (translation invariance,
locality) — well-documented in the ViT/AST literature that transformers underperform CNNs when
trained from scratch on small datasets, and only pull ahead once pretrained on a large corpus (e.g.
AudioSet, ~2M clips) or given far more data than IRMAS's ~5,300 training clips. Training a
transformer from scratch here would very likely perform *worse* than the current CNN, not better.

**The version that could help: a *pretrained* transformer.** That's functionally the same move as
pretrained CNN embeddings (PANNs) — the lever is *pretraining*, not the architecture family. This
project already flagged pretrained embeddings (PANNs/YAMNet/VGGish) as the escalation path
(`spec.md` §5) — a pretrained transformer (AST, AudioSet-pretrained) is one more concrete option
within that same category, not a different strategy.

**Verdict:** transformer-from-scratch — no. Pretrained transformer or pretrained CNN embedding —
yes, this is the strongest remaining lever, and Phase B (below) does both, compared directly.

---

## Phase A — cheap tier, done (2026-08-22)

Recap (see `results.md`/`DECISIONS.md` for full numbers):

1. **Ensemble Run 2 + Run 3** — worked, no retraining, 65%/0.62.
2. **Rerun combined recipe with more epochs** (Run 5) — worked decisively, 65%/0.64, best result.
3. **Focal loss** (Run 6) — mixed/net-negative, fixed clarinet but caused a worse organ collapse.

## Phase B — pretrained embeddings, done (2026-08-22)

Both tried, both frozen-backbone (only a small new head trained):

- **PANNs (CNN14, AudioSet-pretrained)** — 78% test accuracy, macro F1 0.76. Raw 32kHz waveform
  input (its own internal log-mel, 64 bins — incompatible with this project's usual 128-bin/16kHz
  pipeline, needed a parallel raw-waveform data path). Hit and fixed a real bug on the way: its
  internal STFT frontend produces NaN under fp16 autocast, which also surfaced a pre-existing bug
  where `mixed_precision: false` never actually disabled autocast — see `DECISIONS.md`.
- **AST (Audio Spectrogram Transformer, AudioSet-pretrained)** — 78% test accuracy, macro F1 0.76.
  The literal "pretrained transformer" answer to the question that started this. Input-length
  mismatch (pretrained for 10s clips, ours are 3s) handled via loop-padding to ~11s before feature
  extraction — worked cleanly, no embedding-interpolation follow-up needed.

**Both behaved almost nothing like the Phase A runs.** Every from-scratch run needed 15-30+ epochs
to find its footing; PANNs and AST both hit ~70-73% val accuracy in the *first* epoch, then
climbed gently and plateaued with no oscillation at all — no equivalent of Run 1's val-accuracy
crashes or Run 6's organ-collapse instability. Makes sense mechanically: only a small linear head
(~22K params) was actually training, a far easier optimization problem than tuning a full ~0.43M-
param CNN. AST converged even faster than PANNs (plateaued by epoch 8 vs. epoch 14) — the smoothest
curve of any run in the project. The two aren't identical under the hood, though: AST was notably
stronger on organ/voice, PANNs slightly stronger on clarinet/trumpet, and AST's single worst weak
spot was cello↔violin confusion (its largest off-diagonal confusion-matrix entry of any run).

**Result: both tied, ~13 points ahead of the best from-scratch attempt (Run 5, 65%).** Neither
architecture family won — pretraining is what mattered, exactly as predicted above before either
was run. Full curves/confusion matrices/verdicts: `results.md`, Runs 7-8.

**Where this leaves the original question:** the from-scratch CNN's ~65% ceiling was a genuine
data-scarcity limit (~5,300 clips isn't much to learn instrument timbre from scratch), not a
fixable regularization/architecture problem — Phase A's best efforts (ensembling, more training
time) closed real gaps but couldn't reach what one epoch of a *pretrained* model's frozen features
already achieved. "Confidently usable" is a judgment call, but 78% with no class below 0.60 F1 is a
different tier of result than anything Phase A produced.

---

## Summary — full journey, Run 1 through Run 8

**How the original priority list (top of this file) actually held up**, checked against what was
tried: #1 (regularization) and #2 (SpecAugment) were correctly ranked highest — both worked,
became Runs 2 and 3. #4 (transfer learning) was correctly identified as the strongest remaining
lever, appropriately held in reserve until cheaper options were tried first — it ended up being
*the* decisive move (Phase B). #3 (LR scheduling, class-weighted loss) was never actually tried —
overtaken by better-motivated options as evidence accumulated (extended training instead of LR
scheduling for Run 5; focal loss instead of class-weighting for Run 6, since the confusion pattern
turned out not to be frequency-driven — see `DECISIONS.md`). Architecture changes (section 4) were
never revisited — Phase A's gains came from training dynamics and data, not architecture size,
and Phase B changed the model entirely rather than tweaking `BaselineCNN`.

**The throughline connecting every run's behavior, from-scratch or pretrained:** every single run
in the project — all 6 Phase A configs, the ensemble, and both Phase B models — shows the same
family-structured confusions (clarinet↔sax/trumpet, cello↔violin, guitar→piano) in some form.
Regularization, augmentation, more training time, and ensembling all *reduced* these; pretraining
reduced them the most (roughly a third the magnitude of the worst from-scratch runs) but never
eliminated them. That consistency — the same specific pairs, every time, regardless of technique —
is itself evidence some of these pairs are genuinely acoustically similar, not simply an artifact
of too little data or an undertrained model.

**Final state:** best result is Run 7 (PANNs) / Run 8 (AST), tied at 78% accuracy / 0.76 macro F1
— see `results.md` for the full run-by-run numbers and `EXPERIMENTS.md` for a behavioral summary
of each run's training curve. Open, not yet decided (per `HANDOFF.md`): accept this as Phase 1's
result and move to Phase 2 (multi-label), or push further (fine-tuning instead of frozen-backbone,
AST embedding interpolation, or ensembling Run 7 + Run 8 the way Run 2 + Run 3 were ensembled).
