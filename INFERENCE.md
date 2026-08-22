# Using the Model — Instrument Prediction on Your Own Audio

**What this file is:** instructions for actually *using* the trained model — pointing it at an
audio file and getting an instrument prediction back. For how the model was built/trained, see
`README.md` → `spec.md`/`results.md`/`DECISIONS.md`.

## Quick start

```bash
conda activate Sound
python -m src.predict --audio path/to/your/song.mp3
```

That's it — prints the predicted instrument(s) for the clip. Example:

```
model: panns_cnn14  (checkpoint epoch 14, val acc 0.7688)
audio: path/to/your/song.mp3  (5 x 3s windows, last one silence-padded if the clip didn't divide evenly)

Per-window prediction:
  [   0.0s -    3.0s]  organ (0.65), saxophone (0.12), cello (0.06)
  [   3.0s -    6.0s]  piano (0.82), organ (0.09), saxophone (0.03)
  ...

Overall prediction (mean across all 5 windows):
  piano            0.406
  organ            0.265
  saxophone        0.147
```

## What it does and doesn't do

- **Input:** any audio file `soundfile` can read — WAV, MP3, FLAC, OGG, AIFF, and more (verified
  directly on this setup: `libsndfile` 1.2.2, no `ffmpeg` needed). Any length — a 3-second clip or
  a full song both work.
- **Output:** the *predominant* instrument, per 3-second window of the clip, plus one aggregate
  prediction for the whole file (the average of all windows' predictions). It does **not** report
  every instrument simultaneously present in a window — this model only predicts one dominant
  instrument per window (see `spec.md` §2, "Phase 1" — multi-label detection is Phase 2, not built
  yet).
- **11 possible answers:** cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano,
  saxophone, trumpet, violin, voice. If the real dominant sound isn't one of these 11, the model
  will still confidently guess one of them — there's no "none of the above" option.
- **Longer clips are split into independent 3-second windows** (the same window length the model
  was trained on) and each is predicted separately — so a 15-second song shows 5 per-window
  predictions plus one overall summary. If the dominant instrument changes partway through a
  clip, the per-window breakdown will show that; the "overall" line averages across the whole
  clip and can wash out short passages that differ from the rest.

## Choosing which trained model to use

```bash
python -m src.predict --audio song.mp3 --checkpoint checkpoints/run7_panns_frozen.pt   # default
python -m src.predict --audio song.mp3 --checkpoint checkpoints/run8_ast_frozen.pt     # AST
python -m src.predict --audio song.mp3 --checkpoint checkpoints/run5_combined_extended.pt  # from-scratch CNN
```

Any checkpoint works — the script reads the checkpoint's own saved config to figure out which
model it is and preprocesses accordingly. **Run 7 (PANNs) and Run 8 (AST) are exactly tied** on
test accuracy and macro F1 (78% / 0.76 each — see `results.md`). The script defaults to PANNs for
a practical reason, not an accuracy reason: it's a plain CNN, so once its checkpoint is cached
locally there's no further network dependency, whereas AST calls out to the HuggingFace Hub for
its feature extractor config on every run. If you'd rather default to AST, pass `--checkpoint`
explicitly — nothing about the prediction quality changes either way.

`--top-k N` controls how many candidate instruments are shown per window (default 3).

## First-run notes

- **PANNs**: first use auto-downloads its pretrained checkpoint (~327MB) to `~/panns_data/`.
  After that it's cached and loads instantly.
- **AST**: first use downloads its feature extractor config from the HuggingFace Hub (small,
  seconds) — needs internet access each time unless you've configured local HF caching.
- Both checkpoints (`checkpoints/run7_panns_frozen.pt`, `checkpoints/run8_ast_frozen.pt`) must
  already exist locally — they're gitignored (not in the repo), produced by training
  (`python -m src.train --config configs/panns.yaml` / `configs/ast.yaml`, see `EXPERIMENTS.md`).

## Troubleshooting

- **"No such file"** on the checkpoint path — you haven't trained that model yet, or you're not
  in the repo root. Run the corresponding `python -m src.train --config ...` first (see
  `EXPERIMENTS.md`'s reproduce instructions), or point `--checkpoint` at one you do have.
- **A file format won't load** — extremely rare given the format list above, but if it happens,
  the underlying error will name the format; converting to WAV with any audio tool is a safe
  fallback.
- **Prediction looks wrong** — expected sometimes; no model here is 100% accurate (78% test
  accuracy for the default). See `results.md` for exactly which instruments this model confuses
  most often (e.g. clarinet/saxophone/trumpet, cello/violin) so a wrong guess isn't a mystery.
