"""Training entrypoint for Phase 2 (multi-label instrument detection), IRMAS Testing set.

Mirrors src/train.py's structure but for multi-label: BCEWithLogitsLoss instead of
CrossEntropyLoss/FocalLoss, IRMASMultilabelDataset instead of Phase 1's dataset, micro/macro F1
(spec.md Section 6) instead of accuracy for tracking/checkpointing — argmax "accuracy" doesn't mean
anything when several labels can be simultaneously correct.

A separate script rather than extending train.py: loss, label format (multi-hot float vector vs.
a single int class), and the tracked metric all differ meaningfully from Phase 1 — kept as its own
pipeline rather than branching the (working, closed) Phase 1 code path. Model construction is
reused as-is via src.models.registry.build_model — BaselineCNN/PANNsClassifier/ASTClassifier all
already output raw logits (no softmax baked in), so nothing about them needed to change for
multi-label; only the loss function and how predictions are read out differ.

Usage: conda activate Sound && python -m src.train_multilabel --config configs/phase2_panns.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.irmas_dataset import IRMAS_CLASSES
from src.datasets.irmas_multilabel_dataset import IRMASMultilabelDataset
from src.models.registry import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.5  # sigmoid decision threshold for "instrument present"


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, use_amp: bool):
    model.train(train)
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = (torch.sigmoid(logits) > THRESHOLD).float()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, micro_f1, macro_f1, all_labels, all_preds


def main(config_path: str = "configs/phase2_panns.yaml") -> None:
    with open(REPO_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {cfg['model']['name']}  task: multilabel (Phase 2)")

    train_ds = IRMASMultilabelDataset(split="train")
    val_ds = IRMASMultilabelDataset(split="val")
    print(f"train: {len(train_ds)} windows, val: {len(val_ds)} windows")

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_model(cfg, device)
    criterion = nn.BCEWithLogitsLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params, lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    patience = cfg["train"].get("early_stopping_patience")

    run_name = cfg["logging"]["run_name"]
    log_dir = REPO_ROOT / cfg["logging"]["log_dir"] / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"run: {run_name}  tensorboard logs: {log_dir}")

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    best_micro_f1 = 0.0
    epochs_since_improvement = 0

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss, train_micro_f1, train_macro_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True, use_amp=use_amp
        )
        val_loss, val_micro_f1, val_macro_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False, use_amp=use_amp
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("micro_f1/train", train_micro_f1, epoch)
        writer.add_scalar("micro_f1/val", val_micro_f1, epoch)
        writer.add_scalar("macro_f1/train", train_macro_f1, epoch)
        writer.add_scalar("macro_f1/val", val_macro_f1, epoch)

        print(f"epoch {epoch}/{epochs}  train loss {train_loss:.4f} micro_f1 {train_micro_f1:.4f} "
              f"macro_f1 {train_macro_f1:.4f}  val loss {val_loss:.4f} micro_f1 {val_micro_f1:.4f} "
              f"macro_f1 {val_macro_f1:.4f}")

        if val_micro_f1 > best_micro_f1:
            best_micro_f1 = val_micro_f1
            epochs_since_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch,
                 "val_micro_f1": val_micro_f1, "val_macro_f1": val_macro_f1, "config": cfg},
                ckpt_path,
            )
        else:
            epochs_since_improvement += 1
            if patience is not None and epochs_since_improvement >= patience:
                print(f"early stopping: no val micro-F1 improvement in {patience} epochs "
                      f"(best {best_micro_f1:.4f} at epoch {epoch - epochs_since_improvement})")
                break

    writer.close()
    print(f"done. best val micro-F1: {best_micro_f1:.4f}. checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_panns.yaml")
    args = parser.parse_args()
    main(args.config)
