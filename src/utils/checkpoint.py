"""
Checkpoint management utilities for saving and loading model states.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional, Any
import logging


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints during training."""
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = float('-inf')
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            is_best: Whether this is the best model so far.
            filename: Optional custom filename.
        
        Returns:
            Path to the saved checkpoint.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        self.checkpoints.append(checkpoint_path)
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoints = self.checkpoints[:-self.max_checkpoints]
            for checkpoint in old_checkpoints:
                if checkpoint.exists() and 'best_model' not in checkpoint.name:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint}")
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def load_best(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load the best model checkpoint.
        
        Args:
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.
        
        Returns:
            Dictionary containing checkpoint data.
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        return load_checkpoint(best_path, model, optimizer)


def save_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    **kwargs
) -> None:
    """
    Save a checkpoint to file.
    
    Args:
        filepath: Path to save the checkpoint.
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch.
        **kwargs: Additional data to save in checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint from file.
    
    Args:
        filepath: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
    
    Returns:
        Dictionary containing checkpoint data.
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint
