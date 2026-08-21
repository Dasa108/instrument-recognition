# Session Handoff

**Purpose of this file:** context for picking this project up on a new machine/session — read this
first, then `notes/THEORY_NOTES.md` and `spec.md` for the full detail.

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

- **`spec.md`** — Draft v0.1 project spec, **not yet finalized**. Has open questions at the bottom
  (dataset confirmation, framework preference, compute availability, experiment tracking, timeline)
  that the student had not yet answered as of this handoff.

## Next steps (as of handoff)

Nothing has been implemented yet — no code, no data downloaded. The plan per `spec.md`:
1. Resolve the open questions in `spec.md` with the student, finalize the spec.
2. Scaffold the repo structure (`src/`, `data/`, `configs/`, etc.)
3. Write IRMAS download script.
4. Build preprocessing pipeline (audio → log-mel spectrograms), verify visually.
5. Train baseline CNN (Phase 1, single-label on IRMAS).
6. Iterate toward multi-label (Phase 2, OpenMIC-2018 / Slakh2100).

## Working style notes for whoever continues this

- Student prefers theory explained with intuition first, light math, real-world analogies, and a
  quick "does this make sense?" check before moving to the next concept.
- Notes are being kept as a living reference (`THEORY_NOTES.md`) — if teaching continues or new
  concepts come up, keep appending to it in the same style (one-line takeaway per section).
