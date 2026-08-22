# Session Handoff

**Purpose of this file:** context for picking this project up on a new machine/session — read this
first, then `notes/THEORY_NOTES.md` and `spec.md` for the full detail, `DECISIONS.md` for the
reasoning behind non-obvious choices (dataset, framework, etc.) as they get made, `results.md` for
every training run's actual numbers, `notes/improv_cnn.md` for the menu of model-improvement
techniques under consideration, and `EXPERIMENTS.md` for where each run's config/checkpoint/
TensorBoard logs actually live on disk.

## Where things stand

This is a learning project: build an ML model to recognize musical instruments in audio. The
student (repo owner) has **no prior audio/DSP background**, so work so far has been a structured
theory walkthrough before writing any code.

- **`notes/THEORY_NOTES.md`** — complete 7-module theory primer, taught and written module-by-module:
  1. What sound physically is (pressure waves, frequency, amplitude)
  2. Digitizing sound (sampling rate, Nyquist, bit depth)
  3. Fourier Transform (time domain → frequency domain)
  4. Spectrograms (STFT, windowing, time/frequency trade-off)
  5. Mel scale & mel-spectrograms (psychoacoustics)
  6. Timbre (harmonics, formants, ADSR, transients — why instrument recognition is solvable at all)
  7. Bridging to ML (log-mel spectrogram as CNN input, single- vs multi-label framing, evaluation)
  - Plus one sidebar: handling variable-length audio input (windowing + silence-padding, not
    "EOS tokens" — there's no discrete audio vocabulary).
  - **Theory arc status: complete.** Student has confirmed understanding through Module 7.

- **`spec.md`** — **v1.0, finalized (2026-08-21).** All open questions resolved: IRMAS (Phase 1),
  PyTorch, local RTX 3050 (6GB VRAM), TensorBoard for tracking, fully self-paced. See
  `DECISIONS.md` for the reasoning behind each.

- **`DECISIONS.md`** — running decision log (ADR-style: decision / context / alternatives
  considered / why / trade-offs / status). Started this session to record the reasoning behind
  dataset, framework, compute, and tracking choices — student explicitly wants this pattern
  continued for future non-obvious calls, not just handed a decision without the alternatives.

## Next steps (as of handoff)

Repo is scaffolded (`src/`, `data/`, `configs/`, `notebooks/`, `requirements.txt`, `README.md`) —
matches `spec.md` Section 8. Conda env `Sound` is fully set up and verified: Python 3.13.15
(re-derived from the actual current PyPI package matrix, not assumed — see `DECISIONS.md`, "Python
version" entry), `torch==2.13.0+cu130` / `torchaudio==2.11.0` installed and confirmed working
(`torch.cuda.is_available() == True`, device: RTX 3050). All exact package versions pinned in
`requirements.txt`.

**Milestone 1 (download) — done for Training data, Testing still running.**
`src/datasets/download_irmas.py` is implemented (not a stub): downloads all 4 Zenodo archives,
verifies MD5, extracts. `data/raw/IRMAS-TrainingData/` (6,705 files, 11 class folders, confirmed
44.1kHz stereo ~3.0s clips) is downloaded, verified, extracted — usable now. Testing data (3 parts,
~7.8GB) was still downloading in the background as of this handoff (Zenodo throttled to
~500KB–2MB/s — check `data/raw/IRMAS-TestingData/` and re-run the script if it didn't finish; it
resumes/skips already-verified files). Note: IRMAS's Testing files are multi-labeled/
variable-length, not a drop-in single-label test set — see `spec.md` Section 3 and `DECISIONS.md`,
"IRMAS download scope" entry. Phase 1 val/test comes from splitting Training data by song instead.

**Milestone 2 (preprocessing) — done, verified visually.**
`src/preprocessing/audio_to_logmel.py` is implemented (not a stub): `load_audio` → `window` →
`to_logmel` → `process_clip`. Verified end-to-end on real IRMAS audio: output shape is
`(128, 301)` per 3s clip (spec.md's earlier `~130` placeholder was a rough guess — now the
verified real number, updated in `spec.md` Section 4). Plotted one spectrogram per class and
visually confirmed they show real structure (harmonic bands on pitched instruments, note-onset
transients on guitars, chord structure on piano) rather than noise — theory-to-practice check
passed.

Also generated one log-mel spectrogram plot per instrument class in
`notebooks/dataset_visualisation/` (gitignored, regenerable — regenerate via the snippet in
`src/preprocessing/audio_to_logmel.py`'s usage, or ask to redo it).

**Milestone 3 (baseline CNN + iteration) — done. Four runs completed, best result identified.**
- `src/datasets/irmas_dataset.py`: `IRMASDataset` + `build_split()` — song-grouped 80/10/10 split
  (verified zero leakage across splits — see `DECISIONS.md`, "Song-grouped IRMAS split" entry).
- `src/models/cnn.py`: `BaselineCNN` — 5 conv/pool blocks + global-avg-pool + dense head, ~0.43M
  params, optional conv-block dropout (added for Run 2/4).
- `src/train.py`: full training loop — AMP, TensorBoard logging, weight decay, SpecAugment
  (`src/preprocessing/audio_to_logmel.py`'s `spec_augment()`), early stopping.
- `src/evaluate.py`: accuracy/per-class P/R/F1/confusion matrix on the held-out `test` split (the
  song-grouped one from Training data, not IRMAS's official Testing files — see `spec.md` §3).

Also fixed a real bug found while verifying the download: `download_irmas.py`'s extraction-skip
marker was shared across all 3 IRMAS Testing archives, so Part2/Part3 downloaded/verified fine but
silently didn't extract — caught by checking the actual wav count (807 vs. expected 2,874), not by
the script's own "success" output. Fixed (per-archive marker) and re-extracted; all 2,874 files
now present. See `DECISIONS.md`, "Bug fix: IRMAS Testing extraction marker" entry.

**Six training runs + 1 ensemble, comparing overfitting/confusion fixes (`notes/improv_cnn.md`)
against Run 1's baseline — full breakdown, curves, confusion matrices in `results.md`; artifact
locations in `EXPERIMENTS.md`:**

| Run | Change | Test acc | Macro F1 |
|---|---|---:|---:|
| 1 baseline | none | 0.56 | 0.53 |
| 2 regularization | weight decay + conv dropout | 0.60 | 0.57 |
| 3 SpecAugment | time/freq masking | 0.63 | 0.60 |
| 4 combined | Run 2 + Run 3, 30 epochs | 0.55 | 0.53 |
| Ensemble (2+3) | soft-vote, no retraining | 0.65 | 0.62 |
| **5 combined, extended** | Run 4's recipe, 60 epochs | **0.65** | **0.64** |
| 6 focal + SpecAugment | Run 3's recipe + focal loss | 0.58 | 0.56 |

**Best result: Run 5, `checkpoints/run5_combined_extended.pt`.** Highest macro F1 of any
run/ensemble, ties the ensemble on accuracy, most balanced across classes (no F1 below 0.45).
**Key finding:** Run 4 and Run 5 are the *same recipe* — doubling the epoch budget alone closed a
10-point accuracy gap, meaning Run 4's original "combining regularization techniques doesn't help"
conclusion was a training-budget artifact, not a real interaction effect. Second finding: confusion
matrices across Runs 1-5 show the same family-structured mix-ups (clarinet↔sax/trumpet,
cello↔violin, guitar→piano) regardless of regularization approach, including in Run 2 (zero
overfitting) — a real feature-discriminability limit, not just overfitting. Ensembling and focal
loss both targeted this: ensembling worked cleanly (best accuracy); focal loss (Run 6) partially
fixed the clarinet confusion but manufactured a worse new one (organ recall hit 1.00) — net
negative, not adopted. See `DECISIONS.md` for all of the above with full reasoning.

**User asked directly: "should we try a transformer?"** Answer (see `notes/improv_cnn.md`, "Should
we use a transformer?" section): not from scratch — a transformer would very likely underperform
the current CNN given only ~5,300 training clips (well-documented in the ViT/AST literature). A
*pretrained* transformer (AST) could help, but that's the same lever as pretrained CNN embeddings
(PANNs) — pretraining is what matters, not the architecture family. Both are now planned (Phase B).

**Current phase: Phase B (pretrained embeddings) — planned, not yet implemented.**
Plan approved via plan-mode and saved at
`/home/sudarshanab/.claude/plans/federated-hopping-naur.md`. Both PANNs (CNN14, AudioSet-pretrained)
and AST (Audio Spectrogram Transformer, AudioSet-pretrained) will be tried and compared — user's
explicit choice ("both, compare"), not a single pick. Real integration risks already identified and
documented in the plan file / `notes/improv_cnn.md`: PANNs needs a raw-waveform data path (32kHz,
incompatible with the existing log-mel pipeline); AST's pretrained positional embeddings are sized
for 10s clips vs. this project's 3s clips (MVP: loop-pad; embedding interpolation held as a
follow-up). Both frozen-backbone first, fine-tuning only if that underperforms. This also requires
finally wiring up `cfg["model"]["name"]` (set in every config, never read) into a real dispatch —
see the plan file's "Comparison methodology" section.

Not yet started: Phase 2 (multi-label, OpenMIC-2018 / Slakh2100) — IRMAS Testing data (fully
downloaded, 2,874 multi-labeled clips) becomes relevant there; deprioritized behind Phase B per the
user's explicit sequencing choice.

Note (historical): `nvidia-smi` initially couldn't reach the GPU driver from within a Claude Code
session (likely a transient sandboxing state) — confirmed the card via `lspci` at the time. It
started working again in a later session (driver 595.84, CUDA 13.2), so training/dev commands can
run in-session normally now — no need to fall back to the regular terminal.

## Working style notes for whoever continues this

- Student prefers theory explained with intuition first, light math, real-world analogies, and a
  quick "does this make sense?" check before moving to the next concept.
- Notes are being kept as a living reference (`THEORY_NOTES.md`) — if teaching continues or new
  concepts come up, keep appending to it in the same style (one-line takeaway per section).
