from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_writable(path: str | os.PathLike[str], overwrite: bool = False) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to reuse non-empty output directory: {output}. "
            "Choose a new run name or set overwrite=True explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def evaluat_metrics(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.shape != labels.shape:
        raise ValueError(f"Prediction shape {tuple(logits.shape)} != label shape {tuple(labels.shape)}")
    pred = torch.sigmoid(logits).ge(0.5)
    target = labels.ge(0.5)
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0
