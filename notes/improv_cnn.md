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

## Summary — priority order given Run 1's specific overfitting gap

1. **Regularization** (weight decay, more dropout, early stopping) — cheap, directly targets the
   diagnosis.
2. **SpecAugment** — cheap, purpose-built, directly increases effective data diversity.
3. **LR scheduling** + **class-weighted loss** — cheap, smooths training and helps weak classes.
4. **Architecture changes** (aside from pooling strategy) and **transfer learning** — held in
   reserve until 1–3 are tried and we can see whether the train/val gap actually closes.
