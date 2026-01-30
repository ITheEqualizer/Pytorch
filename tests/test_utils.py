"""
Unit tests for utility modules.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.metrics import AverageMeter, calculate_accuracy
from utils.model import count_parameters, initialize_weights, freeze_model, unfreeze_model
from utils.checkpoint import CheckpointManager


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


def test_average_meter():
    meter = AverageMeter("test")
    assert meter.avg == 0.0
    assert meter.count == 0
    
    meter.update(10.0, 2)
    assert meter.avg == 10.0
    assert meter.count == 2
    
    meter.update(20.0, 2)
    assert meter.avg == 15.0
    assert meter.count == 4


def test_calculate_accuracy():
    outputs = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0], [1.0, 0.5, 2.0]])
    targets = torch.tensor([0, 1, 2])
    
    acc = calculate_accuracy(outputs, targets)
    assert acc == 100.0


def test_count_parameters():
    model = SimpleTestModel()
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    assert total_params == trainable_params
    assert total_params == (10 * 5 + 5) + (5 * 2 + 2)


def test_freeze_unfreeze_model():
    model = SimpleTestModel()
    
    freeze_model(model)
    for param in model.parameters():
        assert not param.requires_grad
    
    unfreeze_model(model)
    for param in model.parameters():
        assert param.requires_grad


def test_checkpoint_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manager = CheckpointManager(checkpoint_dir, max_checkpoints=2)
        
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={'loss': 0.5, 'acc': 90.0},
            is_best=True
        )
        
        assert checkpoint_path.exists()
        assert (checkpoint_dir / 'best_model.pth').exists()


def test_initialize_weights():
    model = SimpleTestModel()
    
    initialize_weights(model, init_type='xavier')
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            assert param is not None
