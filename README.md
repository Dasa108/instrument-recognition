# Instrument Recognition

ML model to recognize musical instruments in audio. Learning project — see the docs below for the
full story, not just the code.

- **`HANDOFF.md`** — start here. Where the project stands and what's next.
- **`notes/THEORY_NOTES.md`** — the audio/DSP theory primer this project is built on.
- **`spec.md`** — the finalized project spec (scope, dataset, pipeline, model, evaluation).
- **`DECISIONS.md`** — why each non-obvious choice (dataset, framework, compute, tooling) was made,
  and what alternatives were considered.

## Layout

```
src/
├── datasets/       # download + Dataset/DataLoader classes (code, not data)
├── preprocessing/  # audio -> log-mel spectrogram
├── models/         # CNN architectures
├── train.py
└── evaluate.py
data/               # raw + processed datasets (gitignored — not in version control)
notebooks/          # exploration / visualization
configs/            # training configs (yaml)
```

## Setup

Uses a conda environment named `Sound` (Python 3.11) — already created; all project commands run
inside it.

```bash
conda activate Sound
pip install -r requirements.txt
# Then install torch/torchaudio matched to your CUDA version — see requirements.txt.
```

Status: spec finalized, scaffold in place, no training run yet. See `HANDOFF.md` for the current
milestone.
