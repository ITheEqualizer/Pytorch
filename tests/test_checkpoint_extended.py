"""
Unit tests for checkpoint loading, discovery, and cleanup.
"""

from unittest.mock import patch

import pytest
import torch

from models import SimpleModel
from utils.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint


def _make_model():
    return SimpleModel(input_size=10, hidden_size=5, num_classes=2)


def test_load_checkpoint_round_trip(tmp_path):
    model_a = _make_model()
    optimizer = torch.optim.Adam(model_a.parameters())
    filepath = tmp_path / "ckpt.pth"
    save_checkpoint(filepath, model_a, optimizer, epoch=3)

    model_b = _make_model()
    with patch("utils.checkpoint.torch.load", wraps=torch.load) as torch_load:
        checkpoint = load_checkpoint(filepath, model_b)

    assert checkpoint["epoch"] == 3
    assert torch_load.call_args.kwargs["map_location"] == model_b.fc1.weight.device
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(param_a, param_b)


def test_load_checkpoint_restores_optimizer(tmp_path):
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model(torch.randn(4, 10)).sum().backward()
    optimizer.step()

    filepath = tmp_path / "ckpt.pth"
    save_checkpoint(filepath, model, optimizer, epoch=1)

    model_b = _make_model()
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.001)
    load_checkpoint(filepath, model_b, optimizer_b)

    assert optimizer_b.param_groups[0]["lr"] == 0.005


def test_load_checkpoint_missing_file_raises(tmp_path):
    model = _make_model()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pth", model)


def test_checkpoint_manager_load_best(tmp_path):
    manager = CheckpointManager(tmp_path)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    manager.save(model, optimizer, epoch=1, metrics={"val_acc": 90.0}, is_best=True)

    fresh = _make_model()
    checkpoint = manager.load_best(fresh)

    assert checkpoint["epoch"] == 1
    for param, restored in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(param, restored)


def test_checkpoint_manager_cleanup(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=2)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())

    manager.save(model, optimizer, epoch=1, metrics={"val_acc": 10.0}, is_best=True)
    for epoch in (2, 3, 4):
        manager.save(model, optimizer, epoch=epoch, metrics={"val_acc": float(epoch)})

    assert not (tmp_path / "checkpoint_epoch_1.pth").exists()
    assert not (tmp_path / "checkpoint_epoch_2.pth").exists()
    assert (tmp_path / "checkpoint_epoch_3.pth").exists()
    assert (tmp_path / "checkpoint_epoch_4.pth").exists()
    assert (tmp_path / "best_model.pth").exists()


def test_checkpoint_manager_discovers_existing(tmp_path):
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    first = CheckpointManager(tmp_path, max_checkpoints=5)
    for epoch in (1, 2, 3):
        first.save(model, optimizer, epoch=epoch, metrics={"val_acc": float(epoch)})

    second = CheckpointManager(tmp_path, max_checkpoints=5)
    names = [path.name for path in second.checkpoints]

    assert names == [
        "checkpoint_epoch_1.pth",
        "checkpoint_epoch_2.pth",
        "checkpoint_epoch_3.pth",
    ]


def test_checkpoint_manager_keeps_overwritten_checkpoint_after_restart(tmp_path):
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    first = CheckpointManager(tmp_path, max_checkpoints=2)
    first.save(model, optimizer, epoch=1, metrics={"val_acc": 1.0})
    first.save(model, optimizer, epoch=2, metrics={"val_acc": 2.0})

    restarted = CheckpointManager(tmp_path, max_checkpoints=2)
    saved_path = restarted.save(model, optimizer, epoch=1, metrics={"val_acc": 3.0})

    assert saved_path.exists()
    assert [path.name for path in restarted.checkpoints] == [
        "checkpoint_epoch_2.pth",
        "checkpoint_epoch_1.pth",
    ]


def test_checkpoint_manager_restores_best_metric(tmp_path):
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    first = CheckpointManager(tmp_path)
    first.save(model, optimizer, epoch=1, metrics={"val_acc": 88.5}, is_best=True)

    with patch("utils.checkpoint.torch.load", wraps=torch.load) as torch_load:
        second = CheckpointManager(tmp_path)

    assert second.best_metric == 88.5
    assert torch_load.call_args.kwargs["map_location"] == "cpu"
