"""Training entrypoint for the baseline CNN (Phase 1, single-label on IRMAS).

Spec: spec.md Section 5 (Model), Section 6 (Evaluation), Section 7 (Tech Stack).
Regularization/SpecAugment/early-stopping knobs: notes/improv_cnn.md, results.md Runs 2-4.

Usage: conda activate Sound && python -m src.train [--config configs/base.yaml]
"""

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.irmas_dataset import IRMAS_CLASSES, IRMASDataset
from src.models.cnn import BaselineCNN
from src.preprocessing.audio_to_logmel import spec_augment

REPO_ROOT = Path(__file__).resolve().parents[1]


def confusion_matrix_image(y_true, y_pred, class_names) -> torch.Tensor:
    """Render a confusion matrix as a (C, H, W) uint8 tensor for TensorBoard."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    return torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, aug_cfg=None):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train and aug_cfg and aug_cfg.get("specaugment"):
            x = spec_augment(
                x,
                freq_mask_param=aug_cfg.get("freq_mask_param", 16),
                time_mask_param=aug_cfg.get("time_mask_param", 30),
                num_freq_masks=aug_cfg.get("num_freq_masks", 1),
                num_time_masks=aug_cfg.get("num_time_masks", 1),
            )

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    return total_loss / total, correct / total, all_labels, all_preds


def main(config_path: str = "configs/base.yaml") -> None:
    with open(REPO_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = IRMASDataset(split="train")
    val_ds = IRMASDataset(split="val")
    print(f"train: {len(train_ds)} clips, val: {len(val_ds)} clips")

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = BaselineCNN(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    aug_cfg = cfg.get("augmentation", {})
    patience = cfg["train"].get("early_stopping_patience")

    run_name = cfg["logging"].get("run_name") or (
        f"baseline_cnn_{np.datetime_as_string(np.datetime64('now'), unit='s')}".replace(":", "-")
    )
    log_dir = REPO_ROOT / cfg["logging"]["log_dir"] / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"run: {run_name}  tensorboard logs: {log_dir}")

    # Log a batch of input spectrograms once, for a visual sanity check in TensorBoard.
    sample_x, _ = next(iter(train_loader))
    normed = (sample_x - sample_x.min()) / (sample_x.max() - sample_x.min() + 1e-8)
    writer.add_images("inputs/sample_batch", normed[:16], 0)

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    best_val_acc = 0.0
    epochs_since_improvement = 0

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True, aug_cfg=aug_cfg
        )
        val_loss, val_acc, val_labels, val_preds = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)

        print(f"epoch {epoch}/{epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if epoch % 5 == 0 or epoch == epochs:
            cm_img = confusion_matrix_image(val_labels, val_preds, IRMAS_CLASSES)
            writer.add_image("val/confusion_matrix", cm_img, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                 "config": cfg},
                ckpt_path,
            )
        else:
            epochs_since_improvement += 1
            if patience is not None and epochs_since_improvement >= patience:
                print(f"early stopping: no val-acc improvement in {patience} epochs "
                      f"(best {best_val_acc:.4f} at epoch {epoch - epochs_since_improvement})")
                break

    writer.close()
    print(f"done. best val acc: {best_val_acc:.4f}. checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
