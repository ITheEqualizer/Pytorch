"""
Unit tests for model architectures.
"""
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import SimpleModel


def test_simple_model_initialization():
    model = SimpleModel(input_size=784, hidden_size=128, num_classes=10)
    assert model.input_size == 784
    assert model.hidden_size == 128
    assert model.num_classes == 10


def test_simple_model_forward():
    model = SimpleModel()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 10)


def test_simple_model_with_dropout():
    model = SimpleModel(dropout=0.5)
    assert model.dropout is not None


def test_simple_model_without_dropout():
    model = SimpleModel(dropout=0.0)
    assert model.dropout is None


def test_simple_model_parameter_count():
    model = SimpleModel(input_size=784, hidden_size=128, num_classes=10)
    param_count = model.get_num_parameters()
    
    expected_count = (784 * 128 + 128) + (128 * 10 + 10)
    assert param_count == expected_count


def test_simple_model_custom_sizes():
    model = SimpleModel(input_size=1024, hidden_size=256, num_classes=20)
    input_tensor = torch.randn(2, 1024)
    
    output = model(input_tensor)
    
    assert output.shape == (2, 20)


def test_simple_model_eval_mode():
    model = SimpleModel()
    model.eval()
    
    input_tensor = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output1 = model(input_tensor)
        output2 = model(input_tensor)
    
    assert torch.allclose(output1, output2)
