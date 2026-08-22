"""Ensemble evaluation — averages softmax probabilities from N checkpoints on the held-out test set.

Motivation: Run 2 (stable/balanced) and Run 3 (highest peak accuracy, collapsed clarinet recall)
reach different failure modes on the SAME song-grouped test split — see results.md summary table.
Soft-voting tests whether their errors are complementary (ensemble beats both) or correlated
(doesn't help), at zero training cost. See notes/improv_cnn.md, "Phase A" section.

Usage: conda activate Sound && python -m src.ensemble_evaluate \
    --checkpoints checkpoints/run2_regularization.pt checkpoints/run3_specaugment.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.datasets.irmas_dataset import IRMAS_CLASSES, IRMASDataset
from src.models.cnn import BaselineCNN

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensemble_predict(models: list[BaselineCNN], x: torch.Tensor) -> torch.Tensor:
    """Mean softmax probability across models for one batch. Returns (B, num_classes).

    Equal-weight averaging — the simplest thing that tests the hypothesis (complementary vs.
    correlated errors). Weighted averaging / TTA would be scope creep for this diagnostic step.
    """
    probs = torch.stack([F.softmax(m(x), dim=1) for m in models], dim=0)
    return probs.mean(dim=0)


def main(checkpoint_paths: list[str]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, ckpts = [], []
    for p in checkpoint_paths:
        model, ckpt = BaselineCNN.from_checkpoint(REPO_ROOT / p, device)
        models.append(model)
        ckpts.append(ckpt)
        print(f"loaded {p}: epoch {ckpt['epoch']} val acc {ckpt['val_acc']:.4f}")

    test_ds = IRMASDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"test: {len(test_ds)} clips, ensembling {len(models)} models")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = ensemble_predict(models, x).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print("\n" + classification_report(
        all_labels, all_preds, target_names=IRMAS_CLASSES, zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(IRMAS_CLASSES)))
    print("Confusion matrix (rows=true, cols=predicted):")
    print("     " + " ".join(f"{c:>5}" for c in IRMAS_CLASSES))
    for cls, row in zip(IRMAS_CLASSES, cm):
        print(f"{cls:>4} " + " ".join(f"{v:>5}" for v in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="e.g. checkpoints/run2_regularization.pt checkpoints/run3_specaugment.pt")
    args = parser.parse_args()
    main(args.checkpoints)
