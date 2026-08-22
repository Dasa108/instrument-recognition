# Decision Log

**Purpose:** a running record of the non-obvious calls made on this project — what we chose, what
we considered instead, and *why*. This exists so the reasoning behind a choice survives even after
the choice itself feels obvious in hindsight, and so the student can build intuition for how these
calls get made (not just what was picked).

Each entry follows the same shape: **Decision → Context → Alternatives considered → Why this one →
Trade-offs / risks accepted → Status.**

New entries go at the top (most recent first).

---

## Bug fix: `mixed_precision: false` never actually disabled autocast

**What happened:** `run_epoch`'s autocast context used `enabled=(scaler is not None)` — but
`scaler` (a `torch.amp.GradScaler`) is *always* constructed regardless of the `mixed_precision`
config flag; only its own internal `enabled` state (correctly derived from the flag) controlled
whether it *scaled* gradients. The flag never actually gated whether `torch.autocast` itself ran.
Found via the Run 7 (PANNs) smoke test: `configs/panns.yaml` set `mixed_precision: false`
specifically because Cnn14's internal STFT frontend produces NaN logits under fp16 autocast
(verified directly — fp32 forward pass: clean logits; fp16 autocast forward pass: all-NaN), but
training still produced NaN loss from epoch 1 even with the flag set. Root-caused with a minimal
reproduction (manual forward pass, bypassing the training loop) before concluding it wasn't a
PANNs-specific numerical issue.

**Impact on prior results: none.** Every config through Run 6 (`base`/`reg`/`specaug`/`combined`/
`combined_extended`/`focal`/`class_weighted`) sets `mixed_precision: true` — the bug only matters
when a config wants it *off*, which `configs/panns.yaml` is the first to do. Runs 1-6 all got the
autocast behavior they intended either way; no results need revisiting.

**Fix:** `run_epoch` takes an explicit `use_amp: bool` parameter now, threaded from `main()`'s
already-correct `use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"` computation,
instead of inferring (incorrectly) from the scaler object's mere existence.

**Status:** Fixed and verified (2026-08-22) — PANNs smoke test went from all-NaN to a clean,
sane training curve after the fix.

---

## AST input-length mismatch: loop-pad, not embedding interpolation (MVP)

**Decision:** Loop-pad each 3s IRMAS clip to 11s before AST's own feature extraction, rather than
interpolating the pretrained positional-embedding grid down to ~300 frames.

**Context:** `MIT/ast-finetuned-audioset-10-10-0.4593`'s pretrained positional embeddings are
sized for 1024 time frames (~10.24s, its AudioSet pretraining clip length) — verified directly
against the loaded config (`num_mel_bins=128`, `max_length=1024`, matches this project's own
16kHz/128-mel choices on everything except clip length). This project's clips are ~301 frames
(3s). Also verified HF's `ASTFeatureExtractor`'s actual default behavior for short input
(reading `_extract_fbank_features`'s source, not assuming): it zero-pads the *computed fbank* to
1024 frames — i.e. ~70% of a naively-fed 3s clip's padded input would be silence, a real
distribution mismatch from AudioSet's real 10s clips.

**Alternatives considered:**
- **Interpolate the position-embedding grid** to ~300 frames (analogous to ViT/DeiT resolution
  interpolation) — more correct (no wasted computation on repeated audio) and more efficient, but
  more implementation risk (custom tensor surgery on the checkpoint's positional embeddings).
  Held as a documented follow-up, not attempted — only worth it if the MVP result is promising but
  clearly bottlenecked by the padding waste.
- **Concatenate multiple same-song clips** to reach ~10s — rejected: breaks the single-label-per-
  clip assumption underpinning `IRMASDataset`/the song-grouped split.
- **Leave it to the feature extractor's own zero-padding** (do nothing) — technically works
  (verified it doesn't crash), but feeds ~70% silence, a worse distribution match than repeated
  real signal.

**Why this one:** simplest, lowest implementation risk, keeps pretrained position embeddings
untouched, and replaces silence with real (if repeated) signal — matches this project's established
preference for the proven-default-first approach before custom tensor surgery.

**Trade-offs / risks accepted:** ~70% of the padded input is a repeated/artificial seam, not
real audio diversity — likely leaves some accuracy on the table vs. embedding interpolation.

**Outcome:** 78% test accuracy, macro F1 0.76 (best epoch 8 of 30, early-stopped — the fastest,
smoothest convergence of any run in the project) — effectively tied with PANNs (Run 7), both ~13
points ahead of the best from-scratch result. The loop-padding waste apparently wasn't a binding
constraint — embedding interpolation (the alternative considered above) was not pursued given this
result already matches the pretrained-CNN alternative. Full breakdown: `results.md`, Run 8.

**Status:** Done (2026-08-22).

---

## PANNs input pipeline: raw waveform, own download, not autocast-safe

**Decision:** `PANNsClassifier` uses `panns_inference`'s `Cnn14` directly (not its high-level
`AudioTagging` wrapper), fed raw 32kHz waveform via a new `IRMASWaveformDataset`, with checkpoint
download reusing this project's own `requests`-based `download_file()` (from `download_irmas.py`)
instead of the package's internal `os.system('wget ...')` call.

**Context:** Verified directly against the installed package (not assumed): `Cnn14.forward`
expects raw waveform `(batch, num_samples)` and computes its own internal log-mel (32kHz, 64 mel
bins, hop 320) — this project's existing `audio_to_logmel.py` output (16kHz, 128 bins) would be
the wrong representation entirely, not just a slightly different one. Also verified
`panns_inference.AudioTagging.__init__`'s checkpoint-download logic: a bare `os.system('wget -O
...')` call with no return-code check — a silent-failure risk (if `wget` is missing or the
download fails, training would proceed with an incomplete/missing checkpoint file and only fail
later, confusingly, at `torch.load`).

**Alternatives considered:**
- **Use `AudioTagging`'s wrapper directly** — simpler call site, but pulls in `DataParallel`
  wrapping and its own checkpoint-download fragility; also less composable into a normal
  training loop (its `forward` isn't designed to be called mid-`nn.Module`).
- **Vendor a minimal `Cnn14` definition**, loading only the raw state_dict — considered as a
  fallback if `panns-inference`/`torchlibrosa` had installation friction against this project's
  `numpy==2.5.2`. Not needed — verified `pip install panns-inference transformers` resolves and
  installs cleanly against the pinned versions, dry-run first before committing to the real
  install.

**Why this one:** `Cnn14` used directly composes cleanly as a backbone inside a normal
`nn.Module`/training loop (matches how `BaselineCNN`/`ASTClassifier` are used); reusing this
project's own download helper keeps checkpoint-fetching consistent (streamed, progress bar) and
avoids a silent-failure path the dependency itself has.

**Trade-offs / risks accepted:** Cnn14's internal STFT frontend is numerically unstable under fp16
autocast (produces NaN — see "Bug fix: mixed_precision: false" entry above) — `configs/panns.yaml`
runs in fp32 (`mixed_precision: false`), which is fine since the backbone is frozen (no backward
pass through it) so the fp32 compute cost is minor.

**Outcome:** 78% test accuracy, macro F1 0.76 (best epoch 14 of 30, early-stopped) — ~13 points
ahead of the best from-scratch result (Run 5, 65%). Effectively tied with AST (Run 8). Full
breakdown: `results.md`, Run 7.

**Status:** Done (2026-08-22).

---

## Loss: focal over class-weighted (Run 6)

**Decision:** Run 6 uses focal loss (`gamma=2.0`, no class weighting) as the primary fix for the
confirmed confusion patterns, not inverse-frequency class weighting. Both are implemented
(`src/losses.py`, config-selectable via a new `loss:` block) but class-weighting is held as an
optional control (`configs/class_weighted.yaml`), not the first thing tried.

**Context:** Confusion matrices across Runs 1-4 show the same family-structured mix-ups every run
regardless of regularization approach — clarinet↔sax/trumpet, cello↔violin, acoustic-guitar→piano
— persisting even in Run 2 (zero overfitting). This looks like a genuine feature-discriminability
limit, not something more regularization fixes. Checked actual training-split class counts via
`build_split()`: 306 (cel, rarest) to 622 (voi) — only ~2x spread, mild imbalance.

**Alternatives considered:**
- **Class weighting (inverse frequency)** — the standard fix for imbalanced classes. Checked
  whether it's actually well-motivated here: the worst confusions (gac↔pia: both classes
  *above*-median frequency; cla↔sax/tru: all mid-pack) involve classes that aren't rare relative
  to each other. Only cel↔vio has any plausible frequency angle (cel is the single rarest class),
  and even that confusion *grew* across Runs 1-4 regardless of which regularization was applied —
  arguing for a feature problem, not a sampling-frequency problem. Concluded: not well-motivated
  as the *primary* fix, but cheap enough to keep as an empirical control.

**Why this one:** focal loss upweights low-confidence/hard predictions during training, independent
of class frequency — directly targets "the model confidently confuses specific pairs," which is
what was actually observed, rather than "rare classes get too few gradient updates," which wasn't.
Built on Run 3's exact recipe (`configs/focal.yaml`) since that's the run with the clarinet-recall
collapse this is meant to fix, isolating focal loss's effect against it cleanly.

**Trade-offs / risks accepted:** `gamma=2.0` is the standard Lin et al. default, not tuned for this
task — same "test whether it helps at all before tuning further" philosophy as Runs 2-4's
hyperparameter choices.

**Outcome:** mixed, net negative vs. Run 3. Clarinet recall — the specific problem this targeted —
improved substantially (0.16→0.63), confirming focal loss *can* fix a targeted confusion. But it
introduced a new, more severe collapse: organ recall hit 1.00 (0.32 precision), becoming a
false-positive sink for 6 of the other 10 classes. Overall test accuracy 0.58/macro F1 0.56 —
worse than Run 3 (0.63/0.60) and well below Run 5 (0.65/0.64). Early stopping triggered for the
first time across all 6 runs (epoch 20, best was epoch 13) — this run's val curve was also the
most volatile of any run. Not adopted; the optional class-weighted control
(`configs/class_weighted.yaml`) was deprioritized rather than run, since Run 6's result doesn't
change the reasoning against class-weighting as a primary fix, and focus shifted to Phase B. A
lower `gamma` is a plausible untested follow-up if this is revisited. Full breakdown: `results.md`,
Run 6.

**Status:** Done (2026-08-22). Not adopted.

---

## Combined-recipe epoch budget: 60 epochs (Run 5)

**Decision:** Rerun Run 4's exact recipe (weight decay + conv dropout + SpecAugment) with
`epochs: 60` (was 30) and `early_stopping_patience: 12` (was 7), learning rate unchanged.

**Context:** Run 4 underperformed both Run 2 and Run 3 individually, but its training curve was
still slowly climbing at epoch 30, not clearly plateaued — `results.md`'s own Run 4 verdict flagged
this as likely under-training from stacking three regularization pressures in one epoch budget,
not evidence the combination is fundamentally counterproductive.

**Alternatives considered:**
- **80 or 100 epochs** — measured actual wall-clock across all 4 completed runs (TensorBoard
  event-file timestamps): consistently ~87s/epoch. 60 epochs ≈ 87min, same order of magnitude as
  prior runs; escalating straight to 80-100 without first checking whether 60 already plateaus
  would be guessing ahead of evidence the TensorBoard curve will make obvious anyway.
- **Also raising the learning rate** — rejected to keep this a single-variable test (more epochs,
  and only that), matching the project's existing ablation discipline (Runs 2-4 tested each
  technique in isolation before combining).

**Why this one:** directly tests Run 4's own stated hypothesis rather than assuming it. Patience
raised from 7→12 because a longer run may pass through a longer slow-improvement stretch before a
genuine late gain, and patience 7 never actually triggered in the original Run 4 even under slow
improvement — evidence patience wasn't the binding constraint there, the epoch cap was.

**Trade-offs / risks accepted:** ~87 extra minutes of compute for a result that might still just
confirm Run 4's original conclusion (combining doesn't help) rather than reverse it.

**Outcome:** decisive confirmation. Best val acc 0.5272→0.6082 (same recipe, only epochs changed),
test accuracy 55%→65%, macro F1 0.53→0.64 — the best single-model result of any run. Curve was
still climbing at epoch 60 (not clearly plateaued), so **Run 4's original "combining doesn't help"
conclusion was an artifact of an insufficient epoch budget, not a real interaction effect.** Full
breakdown: `results.md`, Run 5.

**Status:** Done (2026-08-22).

---

## Ensembling Run 2 + Run 3 (soft-vote)

**Decision:** Average Run 2's and Run 3's softmax probabilities on the test set
(`src/ensemble_evaluate.py`), no retraining.

**Context:** Run 2 (stable, balanced, zero overfitting) and Run 3 (highest individual accuracy,
but clarinet recall collapsed to 0.16) reach different failure modes on the *same* held-out test
split — a natural candidate for complementary rather than correlated errors.

**Alternatives considered:** none seriously — this is close to a free experiment (no new training,
minutes of compute), so there was no real cost/benefit trade-off to weigh; the only question was
whether to try it, and the answer was obviously yes given how cheap it is.

**Why this one:** directly tests the complementary-errors hypothesis at near-zero cost before
investing in anything more expensive.

**Outcome:** confirmed the hypothesis — ensemble reaches 65% test accuracy / 0.62 macro F1,
beating both individual runs (Run 3's previous-best: 63%/0.60). Clarinet recall partially recovers
(0.16→0.22, still below Run 2's own 0.47 alone). Full breakdown: `results.md`, "Ensemble" section.

**Status:** Done (2026-08-22).

---

## Runs 2-4: regularization / SpecAugment / combined, isolated then combined

**Decision:** Run three ablation-style experiments against Run 1's baseline, each in its own
config file: `configs/reg.yaml` (weight decay 1e-4 + conv-block dropout 0.2, isolated),
`configs/specaug.yaml` (SpecAugment — freq mask up to 16 of 128 mel bins, time mask up to 30 of
301 frames, 1 of each per sample, isolated), `configs/combined.yaml` (both together). Early
stopping (patience 7 epochs on val accuracy) added to all three as shared infrastructure, not a
fourth variant — see notes/improv_cnn.md, "Verdict: high priority" on regularization/early
stopping, which noted it's a free win with no real downside.

**Context:** Run 1 showed a clear overfitting signature (95%+ train acc vs. ~53% val).
`notes/improv_cnn.md` ranked regularization and SpecAugment as the two highest-priority fixes for
that specific diagnosis. Testing them in isolation before combining answers "which one actually
helped, and by how much" — combining them first would conflate the two effects.

**Hyperparameter choices (each a small default, not tuned):**
- Weight decay 1e-4 — a standard starting point for Adam, not aggressive.
- Conv-block dropout 0.2 — light; Run 1 already had 0.3 dropout in the dense head, so 0.2 in the
  conv blocks stacks with that rather than replacing it.
- SpecAugment mask sizes (16 mel bins / 30 frames) — roughly 12.5%/10% of the spectrogram's
  dimensions respectively, in line with the published SpecAugment paper's "LD" (light) policy
  scaled to this input size, since our clips are much shorter than the paper's.
- Early stopping patience 7 — generous enough to ride out a single bad epoch (Run 1's val acc
  swung by 20+ points between some adjacent epochs) without waiting the full 30 unnecessarily.

**Alternatives considered:** grid/random search over these values — rejected for now as premature;
these are reasonable defaults meant to test *whether the technique helps at all* before spending
compute tuning it further.

**Outcome:** Regularization alone (Run 2) fully eliminated the overfitting gap and improved test
accuracy 56%→60% (macro F1 0.53→0.57). SpecAugment alone (Run 3) reduced but didn't eliminate the
gap, and reached the highest accuracy of all four runs — 63% (macro F1 0.60) — though with a
volatile val curve and one collapsed class (clarinet recall 0.16). **Combining both (Run 4)
underperformed both individual techniques** — 55% accuracy (macro F1 0.53), roughly tied with the
Run 1 baseline. The combined run's curve was stable (no overfitting) but converged slower and
plateaued lower within the same 30-epoch budget — read as *under-training* from stacking three
regularization pressures at once, not evidence combining is fundamentally wrong. Full breakdown:
`results.md`. Best result overall: **Run 3, SpecAugment alone.**

**Status:** Done (2026-08-22).

---

## Bug fix: IRMAS Testing extraction marker was shared across all 3 parts

**What happened:** `download_irmas.py`'s `extract()` used a single `.extracted` marker file inside
the *destination* directory to avoid re-extracting. All three `IRMAS-TestingData-Part{1,2,3}.zip`
archives extract into the same destination (`data/raw/IRMAS-TestingData/`), so once Part1 created
that marker, Part2 and Part3 both downloaded and checksum-verified correctly but silently skipped
extraction. Not caught immediately — the download step reported success for all three, and it took
an explicit `find ... -iname "*.wav" | wc -l` (807 vs. the expected 2,874) to notice.

**Fix:** marker is now per-*archive* (`{archive}.zip.extracted` next to the archive itself in
`data/raw/_archives/`), not per-destination-directory. Re-ran extraction (no re-download needed,
archives were already verified) — confirmed 2,874/2,874 Testing files present afterward.

**Why this matters beyond the immediate fix:** a "did the command exit 0" check would have missed
this — the script's own success message was accurate but incomplete (it downloaded and verified
everything; it just didn't extract two of the four archives). Verifying against an independent
ground truth (the file count IRMAS's own documentation states) is what actually caught it.

**Status:** Fixed and verified (2026-08-22).

---

## Song-grouped IRMAS split (not per-clip)

**Decision:** Phase 1's val/test split (10%/10%, rest train) is carved from IRMAS Training data by
grouping clips on the numeric id embedded in the filename before `__N.wav`, and assigning whole
groups to a split — never splitting a group across train/val/test. Implemented in
`src/datasets/irmas_dataset.py`, `build_split()`.

**Context:** `spec.md` Section 3 already called for "avoid splitting the same song across
train/test to prevent leakage," but that was a stated intent, not yet verified against the real
data. Checked directly: of 2,261 (class, id) groups across the 6,705 downloaded Training clips,
2,250 have more than one clip sharing that id (up to 6) — these are excerpts of the same source
recording. A naive random per-clip split would put different 3s slices of the same recording into
both train and eval, letting the model partly "memorize" a recording's exact production/mixing
fingerprint rather than learning the instrument's timbre — inflating eval accuracy in a way that
wouldn't hold up on genuinely unseen audio.

**Alternatives considered:**
- **Per-clip random split** — simpler to implement, but confirmed unsafe given the id analysis
  above.
- **Use IRMAS's official Testing files as the held-out set** — ruled out already, see "IRMAS
  download scope" entry below: they're multi-labeled/variable-length, not compatible with a
  single-label eval.

**Why this one:** Directly closes the leakage gap the spec already flagged as a risk, verified
against the actual downloaded data rather than assumed. Split is done per-class (not globally) so
each class keeps its own ~80/10/10 proportion — checked: no class's val/test count starves, and a
verification pass confirmed zero group overlap between any two splits.

**Trade-offs / risks accepted:** The regex id-extraction is a heuristic on IRMAS's naming
convention, not a documented guarantee from the dataset itself — if a future IRMAS release changes
naming, this needs re-verification (the same check run here can be re-run to confirm).

**Status:** Confirmed (2026-08-21).

---

## IRMAS download scope: Training + Testing, both now

**Decision:** Download all four IRMAS Zenodo archives now (Training + Testing Parts 1–3, ~11GB
total), not just Training data.

**Context:** IRMAS ships two very different things under one DOI: Training data (6,705 clips,
single-label, fixed 3s — what Phase 1 actually trains/evals on) and Testing data (2,874 clips,
multi-labeled, variable-length 5–20s — built for a multi-label *detection* task). Phase 1's
val/test split is carved from Training data itself (grouped by song), **not** from the official
Testing files — see the correction to `spec.md` Section 3, which previously implied IRMAS's
official split was directly usable as a single-label train/test split.

**Alternatives considered:**
- **Training data only** — everything Phase 1 strictly needs right now (3.2GB vs 11GB). Testing
  data deferred until either a bonus single-label-compatible eval or Phase 2 multi-label work
  actually needs it.

**Why this one:** Explicit choice to grab everything in one pass rather than risk a second
multi-GB Zenodo download later when Phase 2 (or a bonus eval) needs the Testing files — disk space
is not a constraint (827GB free).

**Trade-offs / risks accepted:** ~7.8GB of Testing data sits unused in `data/raw/` until Phase 2 or
a bonus eval actually touches it.

**Status:** Confirmed (2026-08-21).

---

## Python version: 3.13 (superseded initial 3.11 default)

**Decision:** `Sound` env runs Python 3.13.15, not 3.11.

**Context:** The env was first created on Python 3.11 as a conservative default, reasoned from
general knowledge of the ecosystem rather than checked against the actual current package matrix.
Asked directly to re-derive it properly once `nvidia-smi` started working in-session (a reason to
double check everything compute-related, including this).

**What changed the answer:** Queried PyPI directly for the latest release of every package in the
stack and its Python support:
- `librosa` (latest, 1.0.0) and `scipy` (latest, 1.18.0) now *require* Python ≥3.12 — they dropped
  3.11 support entirely. 3.11 was not just conservative, it was already one version too old for the
  current libraries this project depends on.
- `torch` (2.13.0) and `numba` (0.67.0) — the two packages historically slowest to support a new
  Python release — both already ship wheels up to cp314.

**Alternatives considered:**
- **3.11** — the original choice; ruled out, see above (`librosa`/`scipy` don't support it anymore).
- **3.14** — newest release, full wheel coverage confirmed for every package in this stack too, but
  it's the most recently cut version with the least real-world mileage. No concrete package gap
  found, but no upside over 3.13 either.
- **3.12** — the floor requirement from `librosa`/`scipy`. Would work, but 3.13 has equally full
  wheel coverage and is simply newer.

**Why this one:** 3.13 is the newest version with full, verified wheel coverage (not just
classifier claims — checked actual `bdist_wheel` filenames) across every package this project uses,
one cycle more mature than the bleeding-edge 3.14.

**Trade-offs / risks accepted:** None identified — full stack reinstalled clean on 3.13.15, no
conflicts.

**Status:** Confirmed (2026-08-21). Supersedes the original 3.11 choice below.

---

## Environment: conda env `Sound`

**Decision:** Use a conda environment named `Sound` for all project work — created this session,
all deps (including `torch`/`torchaudio`, once CUDA was confirmed reachable) pinned in
`requirements.txt`.

**Context:** Needed a concrete, isolated Python environment before real code lands (download
script, preprocessing, training). Options were conda or a plain `venv`. Python version: see
"Python version: 3.13" entry above.

**Alternatives considered:**
- **`venv` + pip** — no extra dependency (ships with Python), but conda's package/environment
  management is generally the more common choice for ML/scientific-Python setups (handles native
  deps like `libsndfile` — used by `soundfile`/`librosa` — better than pip alone in some cases),
  and the student explicitly asked for conda by name.

**Why this one:** Explicit ask — named environment `Sound` requested directly.

**Trade-offs / risks accepted:** None significant.

**Status:** Confirmed (2026-08-21). `torch==2.13.0+cu130`/`torchaudio==2.11.0` installed and
verified (`torch.cuda.is_available() == True`, device: RTX 3050) once the GPU driver became
reachable from this session — see "Python version: 3.13" entry for exact package versions, all
pinned in `requirements.txt`.

---

## Repo layout: `src/datasets/` instead of `src/data/`

**Decision:** Name the code module `src/datasets/`, not `src/data/`.

**Context:** Draft repo structure had both a root `data/` (raw + processed dataset files,
gitignored — storage) and `src/data/` (download script + `Dataset`/`DataLoader` classes — code).
The split itself is standard (storage vs. the code that touches storage), but naming both `data`
made that distinction unclear at a glance.

**Alternatives considered:**
- `src/data_pipeline/` — also clear, emphasizes the download→load→batch flow rather than just the
  class definitions.
- Leave as `src/data/` — defensible (common enough convention), but doesn't resolve the ambiguity
  that prompted the question in the first place.

**Why this one:** `datasets/` is the more common PyTorch-ecosystem name for "this is where Dataset
class definitions and download logic live," and reads unambiguously distinct from the root `data/`
storage folder.

**Trade-offs / risks accepted:** None — pure naming clarity, no functional difference.

**Status:** Confirmed (2026-08-21).

---

## Experiment tracking: TensorBoard

**Decision:** Use TensorBoard (`torch.utils.tensorboard`) for logging training runs.

**Context:** Need a way to record what config produced what result once more than one training run
exists, without which comparing model variants means guessing from memory.

**Alternatives considered:**
- **Plain logs/CSV** — zero setup, but no built-in visualization; every comparison chart has to be
  hand-written, which becomes its own maintenance burden as soon as more than a couple of runs
  exist.
- **W&B (Weights & Biases)** — cloud-hosted dashboard, excels at comparing many runs side-by-side
  and sharing results, plus sweep tooling for hyperparameter search. Requires a free account and
  sends run metrics off-machine to their servers — an external dependency not justified yet for a
  single baseline CNN with a handful of runs.

**Why this one:**
- PyTorch-native, ships with the ecosystem already chosen — no extra dependency to install/learn.
- Fully local — no account, nothing leaves the machine.
- Gives live loss curves, histograms, and image logging (useful here: can watch log-mel spectrograms
  and confusion matrices update during training) — the real functionality that matters at this
  stage, without cloud overhead.
- Not a one-way door: switching to W&B later (e.g. if Phase 2 involves many tuning runs) is a small
  change, not a rearchitecture.

**Trade-offs / risks accepted:**
- No built-in multi-run comparison UI as polished as W&B's — acceptable while run count is low.

**Status:** Confirmed (2026-08-21).

---

## Compute: RTX 3050 (6GB VRAM, desktop)

**Decision:** Train locally on the student's RTX 3050 (desktop variant, confirmed 6GB VRAM via
`lspci` — `nvidia-smi` couldn't reach the driver from this shell, likely a sandboxing artifact, not
a hardware fact; irrelevant to the decision itself).

**Context:** Phase 1 model (Section 5 of `spec.md`) is a small CNN — 4–6 conv/pool blocks + dense
head — on 128×~130 log-mel spectrogram input. This is a lightweight architecture (low millions of
params), nothing like training a large transformer.

**Alternatives considered:**
- **Colab (free/paid GPU)** — no local setup, but session time limits and variable GPU allocation
  make it a worse fit given local hardware is already sufficient.
- **CPU-only** — not needed; would only be the fallback if no GPU were available.

**Why this one:**
- 6GB comfortably fits this model size at reasonable batch sizes (32–128) without needing
  aggressive memory tricks.
- Ampere architecture (RTX 30-series) has tensor cores — automatic mixed precision (`torch.cuda.amp`)
  gives a real speed-up for free-ish and is worth turning on from the start as a habit, even though
  VRAM isn't expected to be the bottleneck at this scale.
- No session limits, full control over environment — better for iterative debugging than Colab.

**Trade-offs / risks accepted:**
- If Phase 2 (multi-label, larger dataset/model, transfer-learning backbones like PANNs/VGGish) 
  turns out to need more VRAM than 6GB allows, may need to revisit — smaller batch size + gradient
  accumulation, or fall back to Colab for that phase specifically.

**Status:** Confirmed (2026-08-21).

---

## Framework: PyTorch

**Decision:** Use PyTorch (not TensorFlow).

**Context:** Spec draft assumed PyTorch; needed explicit confirmation since framework choice shapes
everything downstream (model code, training loop, ecosystem libraries).

**Alternatives considered:**
- **TensorFlow/Keras** — equally capable for this task; Keras's higher-level API is arguably
  gentler for a first model. Not chosen because the student wants to specifically learn PyTorch.

**Why this one:**
- Explicit learning goal: student wants PyTorch experience, and that's a legitimate, primary
  criterion for a learning project — not just raw technical merit.
- PyTorch's ecosystem (`torchaudio`, `torch.utils.tensorboard`, most published audio-ML research
  code) aligns well with an audio-focused project, reinforcing the choice.

**Trade-offs / risks accepted:** None significant — both frameworks are fully capable here.

**Status:** Confirmed (2026-08-21).

---

## Phase 1 dataset: IRMAS

**Decision:** Use IRMAS (Instrument Recognition in Musical Audio Signals) for Phase 1.

**Context:** Phase 1's objective is single-label predominant-instrument classification on short
clips — the task needs a dataset that's already labeled that way, at a size a from-scratch CNN
pipeline can be iterated on quickly.

**Alternatives considered:**
- **OpenMIC-2018** — 20k×10s clips, 20 classes, multi-label but *weakly* labeled (positive/
  negative/unknown per instrument, not fully annotated). Right task shape for Phase 2, but adds
  partial-label handling (masked loss) that's unnecessary complexity for a first working pipeline.
- **Slakh2100** — 2100 full songs, MIDI-rendered via sample libraries, perfect multi-label ground
  truth. Synthesized audio carries domain-gap risk: a model can learn to recognize soundfont/
  synth artifacts rather than how real instruments actually sound in a mix.
- **NSynth** — ~300k isolated single notes, monophonic. Wrong problem shape entirely — no
  polyphonic mixing, so it teaches timbre in isolation rather than "find the instrument in a mix,"
  which is the actual task.
- **MedleyDB** — ~196 real multitrack songs with instrument activation labels. High quality but
  too small to train a CNN from scratch on its own.
- **MusicNet** — annotated classical recordings. Skews almost entirely orchestral/classical,
  missing guitar, voice, organ etc. that are in our 11-class target list.

**Why this one:**
- Built for exactly this task (predominant-instrument labeling) — no relabeling/adapting needed.
- Real recordings, not synthesized — model has to handle real mixing/production mess, which is the
  actual skill we want, not a MIDI-render proxy for it.
- Small enough (~9k×3s clips) to download and iterate on in one sitting.
- Established benchmark — published baselines exist, giving an external sanity check on whether
  our results are "working as expected" or "something's broken."
- Sequencing: single-label first debugs the *pipeline* (preprocessing → CNN → eval loop) before
  Phase 2 adds multi-label complexity on top of it.

**Trade-offs / risks accepted:**
- 11 classes is a narrower instrument vocabulary than OpenMIC's 20 — acceptable since Phase 1 is
  about proving the pipeline works, not final coverage.
- Predominant-instrument framing doesn't reflect real-world audio (most clips have multiple
  instruments) — that's explicitly deferred to Phase 2, not solved here.

**Status:** Confirmed (2026-08-21).

---
