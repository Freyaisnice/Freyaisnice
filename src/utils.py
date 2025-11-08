"""
Utility functions for training and evaluation.
"""

from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Path,
    metadata: Optional[Dict] = None
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if metadata:
        checkpoint.update(metadata)

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to map checkpoint to

    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0)
    }


def count_parameters(model: nn.Module) -> tuple:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def print_model_summary(model: nn.Module):
    """Print model summary."""
    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 50 + "\n")
