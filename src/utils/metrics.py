"""
Metrics calculation and tracking utilities.
"""
import torch
from typing import Tuple


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        """
        Initialize the meter.
        
        Args:
            name: Name of the metric being tracked.
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.
        
        Args:
            val: Value to add.
            n: Number of instances this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        """String representation of the meter."""
        return f"{self.name}: {self.avg:.4f}"


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Model outputs (logits or probabilities).
        targets: Ground truth labels.
    
    Returns:
        Accuracy as a percentage.
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def calculate_top_k_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        outputs: Model outputs (logits or probabilities).
        targets: Ground truth labels.
        k: Number of top predictions to consider.
    
    Returns:
        Top-k accuracy as a percentage.
    """
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0)
    return 100.0 * correct_k.item() / targets.size(0)


def get_predictions_and_labels(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract predictions and labels from model outputs.
    
    Args:
        outputs: Model outputs.
        targets: Ground truth labels.
    
    Returns:
        Tuple of (predictions, labels).
    """
    _, predictions = outputs.max(1)
    return predictions.cpu(), targets.cpu()
