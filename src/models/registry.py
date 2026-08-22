"""Model + dataset registry, dispatched from cfg["model"]["name"].

Added for Phase B (notes/improv_cnn.md) when a 3rd distinct architecture entered the picture.
`cfg["model"]["name"]` was set in every config from Run 1 onward but never actually read — train.py
and evaluate.py both hardcoded BaselineCNN construction directly. With PANNs/AST needing not just a
different model class but a different Dataset (raw waveform vs. log-mel) and, for AST, a custom
DataLoader collate_fn, that hardcoding became untenable. See DECISIONS.md.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.datasets.irmas_dataset import IRMASDataset
from src.datasets.irmas_waveform_dataset import IRMASWaveformDataset
from src.models.ast_classifier import AST_CHECKPOINT, AST_SAMPLE_RATE, ASTClassifier, make_ast_collate_fn
from src.models.cnn import BaselineCNN
from src.models.panns_classifier import PANNS_SAMPLE_RATE, PANNsClassifier

KNOWN_MODELS = ("baseline_cnn", "panns_cnn14", "ast")


def build_model(cfg, device):
    name = cfg["model"]["name"]
    if name == "baseline_cnn":
        return BaselineCNN(
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"].get("dropout", 0.0),
        ).to(device)
    elif name == "panns_cnn14":
        return PANNsClassifier(
            num_classes=cfg["model"]["num_classes"],
            freeze_backbone=cfg["model"].get("freeze_backbone", True),
            checkpoint_path=cfg["model"].get("checkpoint_path"),
        ).to(device)
    elif name == "ast":
        return ASTClassifier(
            num_classes=cfg["model"]["num_classes"],
            freeze_backbone=cfg["model"].get("freeze_backbone", True),
        ).to(device)
    else:
        raise ValueError(f"unknown model.name: {name!r} (expected one of {KNOWN_MODELS})")


def build_dataset(cfg, split: str) -> Dataset:
    name = cfg["model"]["name"]
    if name == "baseline_cnn":
        return IRMASDataset(split=split)
    elif name == "panns_cnn14":
        return IRMASWaveformDataset(split=split, sample_rate=PANNS_SAMPLE_RATE)
    elif name == "ast":
        return IRMASWaveformDataset(split=split, sample_rate=AST_SAMPLE_RATE)
    else:
        raise ValueError(f"unknown model.name: {name!r} (expected one of {KNOWN_MODELS})")


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    """Reconstruct + load weights from a checkpoint saved by train.py, dispatching model
    construction from the checkpoint's own stored config (works for any model.name, not just
    BaselineCNN — generalizes what used to be a BaselineCNN-only classmethod once evaluate.py and
    ensemble_evaluate.py both needed it for arbitrary model types too). Returns (model.eval(),
    ckpt) so callers can also read ckpt["epoch"]/ckpt["val_acc"]/ckpt["config"].
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def build_collate_fn(cfg):
    """Returns a DataLoader collate_fn, or None to use PyTorch's default."""
    if cfg["model"]["name"] == "ast":
        # Import here, not at module top, so importing this registry doesn't force a network
        # call to the HF Hub (from_pretrained) for callers that never touch an AST config.
        from transformers import ASTFeatureExtractor

        feature_extractor = ASTFeatureExtractor.from_pretrained(AST_CHECKPOINT)
        return make_ast_collate_fn(feature_extractor)
    return None
