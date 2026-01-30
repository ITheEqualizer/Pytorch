"""
Inference script for making predictions with trained PyTorch models.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Tuple
import numpy as np

from config import get_config
from logger import setup_logger
from models import SimpleModel
from utils.checkpoint import load_checkpoint


def load_model(
    model_path: Union[str, Path],
    device: torch.device,
    model_config: dict
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file.
        device: Device to load the model on.
        model_config: Configuration dictionary for model initialization.
    
    Returns:
        Loaded model in evaluation mode.
    
    Raises:
        FileNotFoundError: If model checkpoint doesn't exist.
    """
    model = SimpleModel(
        input_size=model_config.input_size,
        hidden_size=model_config.hidden_size,
        num_classes=model_config.num_classes,
        dropout=0.0
    )
    
    load_checkpoint(model_path, model)
    model.to(device)
    model.eval()
    
    return model


def predict(
    model: nn.Module,
    input_data: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions on input data.
    
    Args:
        model: Trained model to use for inference.
        input_data: Input tensor.
        device: Device to run inference on.
    
    Returns:
        Tuple of (predictions, probabilities).
    """
    with torch.no_grad():
        input_data = input_data.to(device)
        outputs = model(input_data)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = outputs.max(1)
    
    return predictions, probabilities


def predict_batch(
    model: nn.Module,
    data_loader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a batch of data.
    
    Args:
        model: Trained model to use for inference.
        data_loader: DataLoader containing input data.
        device: Device to run inference on.
    
    Returns:
        Tuple of (all_predictions, all_probabilities) as numpy arrays.
    """
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    return np.concatenate(all_predictions), np.concatenate(all_probabilities)


def main() -> None:
    """Main inference function."""
    config = get_config()
    logger = setup_logger()
    
    logger.info(f"Using device: {config.device.device}")
    
    model_path = config.paths.checkpoints_dir / 'best_model.pth'
    
    if not model_path.exists():
        model_path = config.paths.models_dir / 'best_model.pth'
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Please train a model first.")
        return
    
    try:
        model = load_model(model_path, config.device.device, config.model)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    dummy_input = torch.randn(1, 1, 28, 28)
    
    predictions, probabilities = predict(model, dummy_input, config.device.device)
    
    logger.info(f"Prediction: {predictions.item()}")
    logger.info(f"Confidence: {probabilities.max().item():.4f}")
    
    probs_np = probabilities.cpu().numpy()[0]
    logger.info("Class probabilities:")
    for i, prob in enumerate(probs_np):
        logger.info(f"  Class {i}: {prob:.4f}")


if __name__ == '__main__':
    main()
