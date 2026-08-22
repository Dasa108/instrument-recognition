"""Evaluation entrypoint — computes metrics on a trained checkpoint.

Spec: spec.md Section 6 (Evaluation):
    Phase 1 (single-label): accuracy, per-class precision/recall/F1, confusion matrix.

Evaluates on the held-out `test` split carved from IRMAS Training data (song-grouped, never seen
during training) — not IRMAS's official "Testing" files, which are multi-labeled and not a
drop-in single-label test set (see spec.md Section 3, DECISIONS.md "IRMAS download scope" entry).

Usage: conda activate Sound && python -m src.evaluate --checkpoint checkpoints/run2_regularization.pt
"""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.datasets.irmas_dataset import IRMAS_CLASSES
from src.models.registry import build_collate_fn, build_dataset, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(checkpoint_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(REPO_ROOT / checkpoint_path, device)
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")

    cfg = ckpt["config"]
    test_ds = build_dataset(cfg, "test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4,
                              collate_fn=build_collate_fn(cfg))
    print(f"test: {len(test_ds)} clips")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
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
    parser.add_argument("--checkpoint", required=True, help="e.g. checkpoints/run2_regularization.pt")
    args = parser.parse_args()
    main(args.checkpoint)
