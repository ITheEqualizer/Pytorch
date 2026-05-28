"""
Shared pytest fixtures and test configuration.
"""

# flake8: noqa: E402
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from matplotlib import pyplot as plt


@pytest.fixture
def small_model():
    """A tiny SimpleModel suitable for fast unit tests."""
    from models import SimpleModel

    return SimpleModel(input_size=10, hidden_size=5, num_classes=2)


@pytest.fixture
def classification_batch():
    """Logits and targets where the top-1 prediction is correct for every row."""
    outputs = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0], [1.0, 0.5, 2.0]])
    targets = torch.tensor([0, 1, 2])
    return outputs, targets


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close any figures created during a test to avoid resource warnings."""
    yield
    plt.close("all")
