"""
Unit tests for configuration module.
"""
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config, PathConfig, ModelConfig, TrainingConfig, DeviceConfig, get_config


def test_path_config_initialization():
    path_config = PathConfig()
    assert path_config.project_root.exists()
    assert path_config.data_dir.exists()
    assert path_config.models_dir.exists()
    assert path_config.outputs_dir.exists()


def test_model_config_defaults():
    model_config = ModelConfig()
    assert model_config.input_size == 784
    assert model_config.hidden_size == 128
    assert model_config.num_classes == 10
    assert model_config.dropout == 0.2


def test_training_config_defaults():
    training_config = TrainingConfig()
    assert training_config.batch_size == 64
    assert training_config.num_epochs == 10
    assert training_config.learning_rate == 0.001


def test_device_config():
    device_config = DeviceConfig()
    assert isinstance(device_config.device, torch.device)


def test_config_initialization():
    config = Config()
    assert isinstance(config.paths, PathConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.device, DeviceConfig)
    assert config.seed == 42


def test_get_config():
    config = get_config()
    assert isinstance(config, Config)


def test_custom_model_config():
    custom_model_config = ModelConfig(
        input_size=1024,
        hidden_size=256,
        num_classes=20
    )
    assert custom_model_config.input_size == 1024
    assert custom_model_config.hidden_size == 256
    assert custom_model_config.num_classes == 20


def test_custom_training_config():
    custom_training_config = TrainingConfig(
        batch_size=128,
        num_epochs=50,
        learning_rate=0.0001
    )
    assert custom_training_config.batch_size == 128
    assert custom_training_config.num_epochs == 50
    assert custom_training_config.learning_rate == 0.0001
