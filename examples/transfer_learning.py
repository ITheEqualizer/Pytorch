"""
Example: Transfer learning with pre-trained models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import get_config
from logger import setup_logger
from utils.model import freeze_model, unfreeze_model, count_parameters


class PretrainedFeatureExtractor(nn.Module):
    """
    Example transfer learning model using a pretrained feature extractor.
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize model.
        
        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output logits.
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


def demonstrate_transfer_learning():
    """Demonstrate transfer learning workflow."""
    config = get_config()
    logger = setup_logger()
    
    logger.info("Creating model for transfer learning...")
    model = PretrainedFeatureExtractor(num_classes=config.model.num_classes)
    model.to(config.device.device)
    
    logger.info(f"Total parameters: {count_parameters(model):,}")
    
    logger.info("\nPhase 1: Training classifier only (feature extractor frozen)")
    freeze_model(model.feature_extractor)
    
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    
    logger.info("\nPhase 2: Fine-tuning entire model")
    unfreeze_model(model.feature_extractor)
    
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
    )
    
    logger.info("\nTransfer learning setup complete!")
    logger.info("In practice, you would:")
    logger.info("1. Load pretrained weights for feature_extractor")
    logger.info("2. Train classifier with frozen features")
    logger.info("3. Unfreeze and fine-tune with lower learning rate")


if __name__ == '__main__':
    demonstrate_transfer_learning()
