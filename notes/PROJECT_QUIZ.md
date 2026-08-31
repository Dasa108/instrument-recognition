# Project Comprehension Quiz — Questions, Answers, Explanations

**What this file is:** a running record of the comprehension quiz on this project — every
question asked, the answer given, the correct answer, and a full explanation (not just for the
ones missed — this is meant as revision material, read top to bottom like `THEORY_NOTES.md`).
New rounds go at the bottom.

---

## Round 1 (2026-09-01) — 4/4 correct

### Q1. What is the primary task the Phase 1 model was actually trained to do?
- A. Detect all instruments present in a clip (multi-label)
- **B. Identify the single predominant instrument in a 3s clip** ✅ *(answered correctly)*
- C. Separate each instrument's audio track
- D. Classify the genre of the music

**Explanation:** Phase 1 (`spec.md` §2) is single-label predominant-instrument classification —
one label per 3-second clip, for whichever instrument was judged (by IRMAS's original annotators)
as most prominent. Multi-label detection (A) is Phase 2, not started. Source separation (C) and
genre classification (D) are both explicitly out of scope (`spec.md` §2).

### Q2. Why couldn't we use IRMAS's official "Testing" data as our held-out test set in Phase 1?
- **A. It's multi-labeled/variable-length, built for a different task** ✅ *(correct)*
- B. The download never completed successfully
- C. It only contains 4 of the 11 instrument classes
- D. It's literally the same audio as the Training set (would leak)

**Explanation:** IRMAS ships two different things under one DOI: Training data (6,705 clips,
single-label, fixed 3s) and Testing data (2,874 clips, multi-labeled, variable-length 5-20s, built
for a multi-label *detection* task). Both were fully downloaded and verified (`DECISIONS.md`,
"IRMAS download scope" entry), but only Training data is usable for single-label evaluation —
Testing data's ground truth is a *set* of instruments per clip, not one label, so it can't be
scored the same way. Phase 1's val/test split is instead carved from the Training data itself
(song-grouped — see Round 2, Q1).

### Q3. Which two models tied for Phase 1's best result (78% accuracy / macro F1 0.76)?
- **A. PANNs and AST** ✅ *(correct)*
- B. BaselineCNN and PANNs
- C. Run 5 and Run 6
- D. AST and the Run 2+3 ensemble

**Explanation:** Run 7 (PANNs, frozen AudioSet-pretrained CNN) and Run 8 (AST, frozen
AudioSet-pretrained transformer) both landed at 78% test accuracy / 0.76 macro F1 — a dead tie,
~13 points ahead of the best from-scratch result (Run 5, 65%/0.64). Neither architecture family
"won" — pretraining was the deciding factor, not CNN-vs-transformer (`results.md`, "Summary —
Phases A+B complete").

### Q4. Which command gets a prediction on an arbitrary MP3 file using the default best model?
- **A. `python -m src.predict --audio song.mp3`** ✅ *(correct)*
- B. `python -m src.evaluate --checkpoint checkpoints/run7_panns_frozen.pt`
- C. `python -m src.train --config configs/panns.yaml`
- D. `python -m src.ensemble_evaluate --checkpoints checkpoints/run7_panns_frozen.pt checkpoints/run8_ast_frozen.pt`

**Explanation:** `src/predict.py` is the only script built for arbitrary audio files (any format,
any length) — see `INFERENCE.md`. It defaults to `checkpoints/run7_panns_frozen.pt` (Run 7, PANNs)
when `--checkpoint` isn't passed. `evaluate.py` (B) and `ensemble_evaluate.py` (D) both only work
on IRMAS's own pre-split `test` set, not arbitrary files. `train.py` (C) trains a model from a
config — it doesn't predict on anything.

---

## Round 2 (2026-09-01) — 3/4 correct

### Q1. Why does `build_split()` group clips by a shared song ID instead of splitting individual clips randomly?
- **A. Prevent the same source recording's clips leaking across train/eval** ✅ *(correct)*
- B. Keep class balance exactly equal across splits
- C. Individual IRMAS files are too large to split
- D. Make the split reproducible across runs

**Explanation:** Checked directly against the real downloaded data: of 2,261 (class, song-id)
groups across the 6,705 Training clips, 2,250 have more than one clip sharing that id (up to 6) —
these are excerpts of the same source recording. A naive random per-clip split would put different
3s slices of the same recording into both train and eval, letting the model partly memorize that
recording's production/mixing fingerprint rather than learning the instrument's actual timbre
(`DECISIONS.md`, "Song-grouped IRMAS split" entry).

### Q2. Run 4 (regularization + SpecAugment combined, 30 epochs) underperformed both individual techniques. What did Run 5 prove was actually going on?
- **A. Combining techniques just needed more training time** ✅ *(correct)*
- B. The two techniques were fundamentally incompatible
- C. SpecAugment had a bug that needed fixing first
- D. The learning rate was wrong for the combined recipe

**Explanation:** Run 5 used the *exact same recipe* as Run 4 (weight decay + dropout +
SpecAugment), changing only `epochs: 30 → 60`. Val accuracy went from 52.7% to 60.8% — nearly
doubling — with a curve that was still climbing at epoch 60. This proved Run 4's "combining
regularization techniques doesn't help" conclusion was an artifact of an insufficient training
budget, not a real interaction effect (`results.md`, Run 5 verdict).

### Q3. What specifically went wrong with focal loss in Run 6?
- **A. Fixed clarinet's collapse but caused a new, worse "organ" collapse** ✅ *(correct)*
- B. It caused NaN losses from numerical instability
- C. It slowed training down with no accuracy change either way
- D. It only worked when combined with class-weighting, which wasn't tried

**Explanation:** Focal loss (built on Run 3's recipe) upweights low-confidence/hard predictions
during training. It substantially fixed the targeted problem — clarinet recall went from 0.16
(Run 3) to 0.63 — but manufactured a new, worse collapse: organ recall hit a literal 1.00 (100%),
meaning the model started defaulting to "organ" as an unconditional guess whenever unsure,
dragging its precision down to 0.32 and absorbing misclassifications from 6 other classes. Net
effect: worse overall than Run 3 (`results.md`, Run 6 verdict).

### Q4. Why does PANNs need its own separate Dataset class (`IRMASWaveformDataset`) instead of reusing `IRMASDataset`?
- **A. PANNs computes its own internal log-mel from raw 32kHz waveform** ✅ *(correct answer — student answered "don't know", asked for a full explanation of PANNs instead of guessing)*
- B. PANNs requires labels in a different format
- C. `IRMASDataset` doesn't support the song-grouped split
- D. PANNs needs stereo audio, not mono

**Explanation (full, given after the question):**

**What PANNs is:** "**P**retrained **A**udio **N**eural **N**etworks" — a 2019 model family
trained on **AudioSet**: ~2 million YouTube clips covering 527 general sound categories (not just
music — speech, environmental sounds, animals, everything). We used **CNN14**, ~80 million
parameters — about 185x bigger than our own `BaselineCNN` (~0.43M).

**Why "frozen backbone" matters — the actual mechanism:** CNN14 already learned, from those 2
million clips, how to turn raw audio into a rich 2048-number "embedding" capturing general
acoustic structure (timbre, pitch content, texture). We **froze** it entirely (`requires_grad=False`
in `panns_classifier.py`) and only trained a brand-new `Linear(2048, 11)` layer on top — about
22,000 parameters vs. CNN14's 80 million. So training wasn't teaching the model to hear — it
already knew how; we were only teaching it to map that existing hearing onto our 11 instrument
labels. That's why it hit ~70% val accuracy in its *first epoch* — something no from-scratch run
got near even after 30-60 epochs.

**Why the raw-waveform requirement specifically:** CNN14 doesn't take a spectrogram as input at
all — it takes raw audio samples directly `(batch, num_samples)`, and internally computes its
*own* spectrogram using its own pretrained settings (32kHz sample rate, 64 mel bins) baked in from
how it was pretrained. Feeding it our own log-mel output (16kHz, 128 bins) instead wouldn't
error — it would silently compute nonsense on data shaped nothing like what it expects. That's
exactly why `IRMASWaveformDataset` exists as a separate class: it hands PANNs (and AST) raw audio
at the right sample rate, skipping `audio_to_logmel.py` entirely.
