"""Baseline CNN for instrument recognition.

Spec: spec.md Section 5 (Model):
    Phase 1: small CNN from scratch (4-6 conv/pool blocks + dense head) on log-mel input.
    Phase 2: same backbone, swap output head to per-class sigmoid + BCE loss (multi-label).

Input: (batch, 1, 128, 301) log-mel spectrograms (verified shape — spec.md Section 4).
5 conv/pool blocks + global average pool keeps the head agnostic to exact spatial size, so it
still works if window length/hop change later.

`dropout` (conv-block dropout rate) is a regularization knob added for results.md Run 2 —
see notes/improv_cnn.md, section 1. 0.0 (default) reproduces Run 1's architecture exactly.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 11, dropout: float = 0.0):
        super().__init__()
        channels = [1, 16, 32, 64, 128, 256]
        self.features = nn.Sequential(
            *[
                _ConvBlock(channels[i], channels[i + 1], dropout=dropout)
                for i in range(len(channels) - 1)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)  # logits — CrossEntropyLoss applies softmax internally (Phase 1)
