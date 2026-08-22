"""Predict the predominant instrument in any audio file, using a trained checkpoint.

Handles arbitrary-length audio (not just IRMAS's fixed 3s clips): splits it into 3s windows,
silence-padding the last one (same convention as training — see
preprocessing.audio_to_logmel.window()), predicts each window separately, and reports both a
per-window breakdown and an aggregate whole-clip prediction (mean softmax across windows).

Works with any checkpoint (baseline_cnn / panns_cnn14 / ast) — the checkpoint's own stored config
determines which preprocessing path to use, matching how it was trained (src.models.registry).

Usage: conda activate Sound && python -m src.predict --audio path/to/song.mp3
       [--checkpoint checkpoints/run7_panns_frozen.pt] [--top-k 3]

Default checkpoint is Run 7 (PANNs). Run 8 (AST) is exactly tied with it on test accuracy/macro F1
(78% / 0.76 each, see results.md) — PANNs is the default only because it's the simpler dependency
at inference time (pure CNN, no HuggingFace Hub call needed once its checkpoint is cached). Pass
`--checkpoint checkpoints/run8_ast_frozen.pt` to use AST instead. See INFERENCE.md.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.irmas_dataset import IRMAS_CLASSES
from src.models.ast_classifier import AST_CHECKPOINT, AST_SAMPLE_RATE, loop_pad_waveform
from src.models.panns_classifier import PANNS_SAMPLE_RATE
from src.models.registry import load_checkpoint
from src.preprocessing.audio_to_logmel import load_audio, to_logmel, window

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "checkpoints/run7_panns_frozen.pt"
CLIP_SECONDS = 3.0

IRMAS_CLASS_NAMES = {
    "cel": "cello", "cla": "clarinet", "flu": "flute", "gac": "acoustic guitar",
    "gel": "electric guitar", "org": "organ", "pia": "piano", "sax": "saxophone",
    "tru": "trumpet", "vio": "violin", "voi": "voice",
}


def prepare_windows(audio_path: str, model_name: str) -> torch.Tensor:
    """Audio file -> a (num_windows, ...) batch tensor ready for the given model type's
    forward(), using exactly the preprocessing that model was trained with."""
    if model_name == "baseline_cnn":
        waveform = load_audio(audio_path, sample_rate=16000)
        windows = window(waveform, clip_seconds=CLIP_SECONDS, sample_rate=16000)
        logmels = np.stack([to_logmel(w) for w in windows])
        return torch.from_numpy(logmels).unsqueeze(1).float()  # (N, 1, 128, 301)

    elif model_name == "panns_cnn14":
        waveform = load_audio(audio_path, sample_rate=PANNS_SAMPLE_RATE)
        windows = window(waveform, clip_seconds=CLIP_SECONDS, sample_rate=PANNS_SAMPLE_RATE)
        return torch.from_numpy(windows).float()  # (N, num_samples)

    elif model_name == "ast":
        from transformers import ASTFeatureExtractor  # deferred: avoid an HF Hub call for
                                                        # callers that never use AST

        waveform = load_audio(audio_path, sample_rate=AST_SAMPLE_RATE)
        windows = window(waveform, clip_seconds=CLIP_SECONDS, sample_rate=AST_SAMPLE_RATE)
        padded = [loop_pad_waveform(w, sample_rate=AST_SAMPLE_RATE) for w in windows]
        feature_extractor = ASTFeatureExtractor.from_pretrained(AST_CHECKPOINT)
        features = feature_extractor(padded, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
        return features["input_values"]  # (N, 1024, 128)

    else:
        raise ValueError(f"unknown model.name: {model_name!r}")


def main(audio_path: str, checkpoint_path: str, top_k: int = 3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(REPO_ROOT / checkpoint_path, device)
    model_name = ckpt["config"]["model"]["name"]
    print(f"model: {model_name}  (checkpoint epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f})")

    x = prepare_windows(audio_path, model_name).to(device)
    num_windows = x.shape[0]
    print(f"audio: {audio_path}  ({num_windows} x {CLIP_SECONDS:.0f}s window"
          f"{'s' if num_windows != 1 else ''}, last one silence-padded if the clip didn't divide "
          f"evenly)")

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu()  # (N, 11)

    print("\nPer-window prediction:")
    for i, p in enumerate(probs):
        top = torch.topk(p, min(top_k, len(IRMAS_CLASSES)))
        t0, t1 = i * CLIP_SECONDS, (i + 1) * CLIP_SECONDS
        breakdown = ", ".join(
            f"{IRMAS_CLASS_NAMES[IRMAS_CLASSES[idx]]} ({prob:.2f})"
            for prob, idx in zip(top.values.tolist(), top.indices.tolist())
        )
        print(f"  [{t0:6.1f}s - {t1:6.1f}s]  {breakdown}")

    overall = probs.mean(dim=0)
    top = torch.topk(overall, min(top_k, len(IRMAS_CLASSES)))
    print(f"\nOverall prediction (mean across all {num_windows} window"
          f"{'s' if num_windows != 1 else ''}):")
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        name = IRMAS_CLASS_NAMES[IRMAS_CLASSES[idx]]
        print(f"  {name:<16s} {prob:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to any audio file (wav, mp3, flac, ogg, ...)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help=f"default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--top-k", type=int, default=3, help="how many candidate instruments to show")
    args = parser.parse_args()
    main(args.audio, args.checkpoint, args.top_k)
