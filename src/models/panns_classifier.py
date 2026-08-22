"""PANNs (CNN14) classifier — AudioSet-pretrained CNN backbone + a fresh classifier head.

Phase B, option 1 of 2 (see notes/improv_cnn.md "Phase B" section, DECISIONS.md "PANNs input
pipeline" entry). Backbone: `panns_inference`'s Cnn14 (Kong et al. 2020), pretrained on AudioSet
(~2M clips, 527 classes) — vastly more audio than IRMAS's ~5,300 training clips.

Real integration point verified against the installed package (not assumed): Cnn14 expects raw
waveform at 32kHz (batch, num_samples) and computes its own internal log-mel (64 mel bins, hop
320) — feeding it this project's own audio_to_logmel.py output (128 bins, 16kHz) would be the
wrong representation, not just a slightly different one. Use IRMASWaveformDataset (32kHz) for this
model, not IRMASDataset.

Checkpoint download deliberately does NOT use panns_inference's own AudioTagging wrapper (which
downloads via a bare `os.system('wget ...')` call with no return-code check — silent-failure risk,
and inconsistent with how this project downloads everything else). Reuses
download_irmas.download_file() (requests + progress bar) instead, to the same cache path
panns_inference itself uses (~/panns_data/) so a prior panns_inference download is still found.
"""

from pathlib import Path

import torch
import torch.nn as nn
from panns_inference.models import Cnn14

from src.datasets.download_irmas import download_file

PANNS_SAMPLE_RATE = 32000
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_CHECKPOINT_PATH = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
PANNS_AUDIOSET_CLASSES = 527  # pretrained checkpoint's own head size — must match to load weights
PANNS_EMBEDDING_DIM = 2048


def _ensure_checkpoint(checkpoint_path: Path) -> Path:
    # Real pretrained checkpoint is ~327MB; anything much smaller is a previous partial/failed
    # download (matches the size-sanity-check panns_inference itself uses, ported to our own
    # download path).
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 3 * 10**8:
        print(f"[download] PANNs Cnn14 checkpoint -> {checkpoint_path}")
        download_file(PANNS_CHECKPOINT_URL, checkpoint_path)
    return checkpoint_path


class PANNsClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 11,
        freeze_backbone: bool = True,
        checkpoint_path: str | Path | None = None,
    ):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=PANNS_SAMPLE_RATE, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=PANNS_AUDIOSET_CLASSES,
        )
        ckpt_path = Path(checkpoint_path) if checkpoint_path else PANNS_CHECKPOINT_PATH
        ckpt_path = _ensure_checkpoint(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(checkpoint["model"])

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # keep backbone's BatchNorm/SpecAugment in eval mode even
                                   # when the classifier head is in train mode (see train())

        self.head = nn.Linear(PANNS_EMBEDDING_DIM, num_classes)

    def train(self, mode: bool = True) -> "PANNsClassifier":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # override: never let .train() re-enable backbone BN/dropout
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, num_samples) raw 32kHz audio — NOT log-mel.
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(waveform)["embedding"]  # (batch, 2048)
        return self.head(embedding)
