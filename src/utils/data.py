"""
Data loading and processing utilities.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional
import numpy as np


class CustomDataset(Dataset):
    """Custom dataset class for flexible data loading."""
    
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform=None):
        """
        Initialize dataset.
        
        Args:
            data: Input data tensor.
            labels: Labels tensor.
            transform: Optional transform to apply to data.
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index of item to retrieve.
        
        Returns:
            Tuple of (data, label).
        """
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


def create_data_loaders(
    dataset: Dataset,
    batch_size: int,
    train_split: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders from a dataset.
    
    Args:
        dataset: Full dataset to split.
        batch_size: Batch size for data loaders.
        train_split: Fraction of data to use for training.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for reproducible splits.
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def normalize_data(
    data: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize data using mean and standard deviation.
    
    Args:
        data: Data to normalize.
        mean: Optional pre-computed mean. If None, computed from data.
        std: Optional pre-computed std. If None, computed from data.
    
    Returns:
        Tuple of (normalized_data, mean, std).
    """
    if mean is None:
        mean = data.mean(dim=0)
    if std is None:
        std = data.std(dim=0)
    
    std = torch.where(std == 0, torch.ones_like(std), std)
    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std
