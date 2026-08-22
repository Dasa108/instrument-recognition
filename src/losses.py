"""Loss functions beyond plain cross-entropy.

See DECISIONS.md "Loss: focal over class-weighted" entry for why focal loss (not inverse-frequency
class weighting) is the primary fix tried for the Runs 1-4 confusion patterns: training class
counts range 306-622 (~2x, mild imbalance), and the worst/most persistent confusions (gac<->pia,
cla<->sax/tru) involve classes that are NOT rare relative to each other — a frequency-based fix
isn't well-motivated. Focal loss targets low-confidence/hard predictions directly, regardless of
class frequency. See notes/improv_cnn.md, "Phase A" section.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from an IRMASDataset's labels, normalized to mean 1.0."""
    counts = torch.zeros(num_classes)
    for _, label in dataset.samples:  # IRMASDataset.samples: list[(Path, int)]
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    return weights * (num_classes / weights.sum())


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017).

    gamma=0 reduces to (optionally alpha-weighted) cross-entropy — useful as a sanity check.
    alpha (per-class weight) is independent of gamma so either knob can be tuned/disabled without
    the other (e.g. combine with compute_class_weights as a secondary experiment without changing
    the hard-example-reweighting behavior).

    AMP note: F.log_softmax is one of the ops torch.autocast promotes to fp32 automatically (same
    category as nn.CrossEntropyLoss's internals, which already run under this exact code path in
    train.py's run_epoch) — no special AMP handling needed here.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        log_p_t = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        loss = -((1 - p_t) ** self.gamma) * log_p_t
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()
