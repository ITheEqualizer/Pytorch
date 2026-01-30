"""
Model utilities for initialization, parameter counting, and model manipulation.
"""
import torch
import torch.nn as nn
from typing import Dict, List


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.
    
    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model to summarize.
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 70)


def initialize_weights(model: nn.Module, init_type: str = 'xavier') -> None:
    """
    Initialize model weights using specified initialization strategy.
    
    Args:
        model: Model to initialize.
        init_type: Type of initialization ('xavier', 'kaiming', 'normal').
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.01)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def freeze_model(model: nn.Module, layer_names: List[str] = None) -> None:
    """
    Freeze model parameters to prevent training.
    
    Args:
        model: Model to freeze.
        layer_names: Optional list of specific layer names to freeze. If None, freezes all.
    """
    if layer_names is None:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False


def unfreeze_model(model: nn.Module, layer_names: List[str] = None) -> None:
    """
    Unfreeze model parameters to allow training.
    
    Args:
        model: Model to unfreeze.
        layer_names: Optional list of specific layer names to unfreeze. If None, unfreezes all.
    """
    if layer_names is None:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True


def get_layer_lr_groups(
    model: nn.Module,
    base_lr: float,
    layer_decay: float = 0.95
) -> List[Dict]:
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Args:
        model: Model to create groups for.
        base_lr: Base learning rate.
        layer_decay: Decay factor for learning rate across layers.
    
    Returns:
        List of parameter group dictionaries.
    """
    param_groups = []
    
    for i, (name, params) in enumerate(model.named_parameters()):
        if params.requires_grad:
            lr = base_lr * (layer_decay ** i)
            param_groups.append({
                'params': params,
                'lr': lr,
                'name': name
            })
    
    return param_groups
