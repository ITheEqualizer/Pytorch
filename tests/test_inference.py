"""
Component tests for the inference helpers and the dummy-data pipeline.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config, TrainingConfig
from inference import predict, predict_batch
from models import SimpleModel
from train import create_dummy_data


def test_create_dummy_data_shapes():
    config = Config(training=TrainingConfig(batch_size=8))
    train_loader, val_loader = create_dummy_data(config)

    inputs, targets = next(iter(train_loader))
    assert inputs.shape[1:] == (1, 28, 28)
    assert inputs.shape[0] <= 8
    assert targets.max().item() < config.model.num_classes
    assert targets.min().item() >= 0
    assert len(val_loader.dataset) == 200


def test_predict_shapes():
    device = torch.device("cpu")
    model = SimpleModel(input_size=784, hidden_size=16, num_classes=10)
    inputs = torch.randn(3, 1, 28, 28)

    predictions, probabilities = predict(model, inputs, device)

    assert predictions.shape == (3,)
    assert probabilities.shape == (3, 10)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-5)


def test_predict_batch_concatenates():
    device = torch.device("cpu")
    model = SimpleModel(input_size=784, hidden_size=16, num_classes=10)
    dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))
    loader = DataLoader(dataset, batch_size=4)

    predictions, probabilities = predict_batch(model, loader, device)

    assert predictions.shape == (10,)
    assert probabilities.shape == (10, 10)
