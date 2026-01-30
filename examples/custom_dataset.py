"""
Example: Using custom datasets with the PyTorch project.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import get_config
from models import SimpleModel
from logger import setup_logger
import torch.nn as nn
import torch.optim as optim
from utils import CheckpointManager, AverageMeter, calculate_accuracy


class CustomImageDataset(Dataset):
    """
    Example custom dataset for image classification.
    Replace this with your actual dataset logic.
    """
    
    def __init__(self, num_samples=1000, image_size=28, num_classes=10):
        """
        Initialize dataset.
        
        Args:
            num_samples: Number of samples to generate.
            image_size: Size of square images.
            num_classes: Number of classification classes.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.images = torch.randn(num_samples, 1, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        """Return dataset size."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get item by index.
        
        Args:
            idx: Index of item.
        
        Returns:
            Tuple of (image, label).
        """
        return self.images[idx], self.labels[idx]


def create_data_loaders(config):
    """
    Create train and validation data loaders.
    
    Args:
        config: Configuration object.
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = CustomerImageDataset(num_samples=5000)
    val_dataset = CustomImageDataset(num_samples=1000)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    return train_loader, val_loader


def train_with_custom_dataset():
    """Train model with custom dataset."""
    config = get_config()
    logger = setup_logger(log_dir=config.paths.logs_dir)
    
    logger.info("Loading custom dataset...")
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    model = SimpleModel(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes
    ).to(config.device.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    checkpoint_manager = CheckpointManager(config.paths.checkpoints_dir)
    
    logger.info("Starting training...")
    for epoch in range(1, 3):
        model.train()
        loss_meter = AverageMeter('Loss')
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device.device), targets.to(config.device.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), inputs.size(0))
        
        logger.info(f"Epoch {epoch}: Train Loss = {loss_meter.avg:.4f}")
        
        checkpoint_manager.save(
            model, optimizer, epoch,
            metrics={'train_loss': loss_meter.avg},
            is_best=(epoch == 1)
        )
    
    logger.info("Training complete!")


if __name__ == '__main__':
    train_with_custom_dataset()
