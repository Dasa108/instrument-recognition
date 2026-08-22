"""AST (Audio Spectrogram Transformer) classifier — AudioSet-pretrained transformer + a fresh head.

Phase B, option 2 of 2 (see notes/improv_cnn.md "Phase B" section, DECISIONS.md "AST input-length
mismatch" entry) — this is the literal "pretrained transformer" answer to "should we try a
transformer": a from-scratch transformer would very likely underperform the current CNN on
~5,300 clips, but this one is pretrained on AudioSet (~2M clips).

Real integration point, verified against the actual checkpoint config (not assumed): this
checkpoint expects 16kHz / 128 mel bins (matches this project's existing choices exactly) but its
pretrained positional embeddings are sized for 1024 time frames (~10.24s — its AudioSet
pretraining clip length). This project's clips are ~3s (~301 frames). Two fixes considered:
  (a) loop-pad the raw waveform to ~10s before feature extraction — keeps pretrained position
      embeddings untouched, simplest. Chosen for the MVP.
  (b) interpolate the position-embedding grid down to ~300 frames — more correct/efficient (no
      wasted computation on repeated audio), more implementation risk. Not attempted yet.
Verified HF's ASTFeatureExtractor's *default* behavior for short input is zero-padding the fbank
after extraction (silence, ~70% of the padded region) — confirmed by reading its source
(_extract_fbank_features: torch.nn.ZeroPad2d when `max_length - n_frames > 0`). Loop-padding the
waveform ourselves first avoids that, feeding real (repeated) signal instead of a large silent gap.

`classifier.*` is the correct top-level module name for the head (verified against the loaded
model's named_parameters() — `ignore_mismatched_sizes=True` already reinitializes it to
`num_classes` automatically when it doesn't match the pretrained 527-class AudioSet head).
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTForAudioClassification

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000
AST_MAX_FRAMES = 1024  # pretrained positional-embedding grid size for this checkpoint
# Loop-pad target: comfortably exceeds the ~10.24s theoretical minimum (1024 frames at this
# extractor's 10ms hop) so the resulting fbank has >=1024 real (repeated) frames before the
# feature extractor's own truncate-to-1024 step — no zero-padding should ever trigger.
AST_LOOP_PAD_SECONDS = 11.0


def loop_pad_waveform(waveform: np.ndarray, sample_rate: int = AST_SAMPLE_RATE) -> np.ndarray:
    """Repeat a short clip until it reaches AST_LOOP_PAD_SECONDS, then trim to exactly that
    length. Deterministic (always the same output length), unlike leaving it to the feature
    extractor's own zero-pad/truncate step, which would otherwise see mostly silence for a 3s
    clip."""
    target_samples = int(AST_LOOP_PAD_SECONDS * sample_rate)
    if len(waveform) >= target_samples:
        return waveform[:target_samples]
    n_repeats = -(-target_samples // len(waveform))  # ceil division
    return np.tile(waveform, n_repeats)[:target_samples]


class ASTClassifier(nn.Module):
    def __init__(self, num_classes: int = 11, freeze_backbone: bool = True):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            AST_CHECKPOINT, num_labels=num_classes, ignore_mismatched_sizes=True,
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, p in self.ast.named_parameters():
                if not name.startswith("classifier"):
                    p.requires_grad = False

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # input_values: (batch, 1024, 128) — pre-extracted AST features (see
        # ast_collate_fn/prepare_ast_input), NOT raw waveform and NOT this project's log-mel.
        return self.ast(input_values=input_values).logits


def make_ast_collate_fn(feature_extractor: ASTFeatureExtractor):
    """Returns a DataLoader collate_fn: batch of (waveform, label) from IRMASWaveformDataset(
    sample_rate=16000) -> (input_values, labels) ready for ASTClassifier.forward().

    Loop-padding + feature extraction happens here (CPU, in DataLoader workers) rather than inside
    the model's forward(), matching the standard HF pattern of feature extraction as a
    preprocessing step, not a differentiable model op.
    """

    def collate_fn(batch: list[tuple[torch.Tensor, int]]):
        waveforms = [loop_pad_waveform(w.numpy()) for w, _ in batch]
        labels = torch.tensor([label for _, label in batch])
        features = feature_extractor(
            waveforms, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt"
        )
        return features["input_values"], labels

    return collate_fn
