"""
Tests for the training-loop building blocks and efficiency flags.
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config, TrainingConfig
from models import SimpleModel
from train import create_dummy_data, create_grad_scaler, train_epoch


def _build_model(config):
    return SimpleModel(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
    )


def test_create_grad_scaler_disabled_on_cpu():
    scaler = create_grad_scaler(torch.device("cpu"), enabled=False)
    assert scaler.is_enabled() is False


def test_train_epoch_runs_with_grad_clip():
    config = Config(training=TrainingConfig(batch_size=64, num_epochs=1))
    train_loader, _ = create_dummy_data(config)
    model = _build_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = create_grad_scaler(config.device.device, enabled=False)

    loss, acc = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        config.device.device,
        1,
        scaler=scaler,
        gradient_clip=config.training.gradient_clip,
    )

    assert math.isfinite(loss)
    assert 0.0 <= acc <= 100.0


def test_deterministic_config_reproducible():
    def first_batch_loss():
        config = Config(training=TrainingConfig(batch_size=64), deterministic=True)
        train_loader, _ = create_dummy_data(config)
        model = _build_model(config)
        criterion = nn.CrossEntropyLoss()
        inputs, targets = next(iter(train_loader))
        return criterion(model(inputs), targets).item()

    assert first_batch_loss() == first_batch_loss()
