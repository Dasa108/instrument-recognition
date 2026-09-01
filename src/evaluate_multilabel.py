"""Evaluation entrypoint for Phase 2 (multi-label) — computes metrics on a trained checkpoint.

Spec: spec.md Section 6 (Evaluation), Phase 2: macro-F1 and micro-F1 (primary metrics), per-class
precision/recall. Evaluates on the held-out `test` split of IRMAS's official Testing set
(song-grouped, never seen during training — src/datasets/irmas_multilabel_dataset.py).

Usage: conda activate Sound && python -m src.evaluate_multilabel --checkpoint checkpoints/phase2_panns.pt
"""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score, hamming_loss
from torch.utils.data import DataLoader

from src.datasets.irmas_dataset import IRMAS_CLASSES
from src.datasets.irmas_multilabel_dataset import IRMASMultilabelDataset
from src.models.registry import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.5


def main(checkpoint_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(REPO_ROOT / checkpoint_path, device)
    print(f"loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val micro-F1 {ckpt['val_micro_f1']:.4f}, macro-F1 {ckpt['val_macro_f1']:.4f})")

    test_ds = IRMASMultilabelDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"test: {len(test_ds)} windows")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = (torch.sigmoid(model(x)) > THRESHOLD).float().cpu()
            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    h_loss = hamming_loss(all_labels, all_preds)
    print(f"\nmicro-F1: {micro_f1:.4f}   macro-F1: {macro_f1:.4f}   "
          f"hamming loss: {h_loss:.4f} (fraction of individual instrument-present/absent "
          f"decisions that are wrong, lower is better)")

    print("\n" + classification_report(
        all_labels, all_preds, target_names=IRMAS_CLASSES, zero_division=0
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. checkpoints/phase2_panns.pt")
    args = parser.parse_args()
    main(args.checkpoint)
