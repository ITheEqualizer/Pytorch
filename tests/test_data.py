"""
Unit tests for data utilities.
"""

import torch

from utils.data import CustomDataset, create_data_loaders, normalize_data


def test_custom_dataset_len():
    data = torch.randn(20, 3)
    labels = torch.randint(0, 2, (20,))
    dataset = CustomDataset(data, labels)
    assert len(dataset) == 20


def test_custom_dataset_getitem():
    data = torch.randn(5, 3)
    labels = torch.arange(5)
    dataset = CustomDataset(data, labels)
    sample, label = dataset[2]
    assert torch.equal(sample, data[2])
    assert label.item() == 2


def test_custom_dataset_transform():
    data = torch.ones(4, 3)
    labels = torch.zeros(4, dtype=torch.long)
    dataset = CustomDataset(data, labels, transform=lambda x: x * 2)
    sample, _ = dataset[0]
    assert torch.equal(sample, torch.full((3,), 2.0))


def test_create_data_loaders_split():
    data = torch.randn(100, 4)
    labels = torch.randint(0, 3, (100,))
    dataset = CustomDataset(data, labels)

    train_loader, val_loader = create_data_loaders(
        dataset, batch_size=10, train_split=0.8, num_workers=0
    )

    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 20


def test_create_data_loaders_reproducible():
    data = torch.randn(50, 4)
    labels = torch.randint(0, 3, (50,))
    dataset = CustomDataset(data, labels)

    first, _ = create_data_loaders(dataset, batch_size=10, num_workers=0, seed=7)
    second, _ = create_data_loaders(dataset, batch_size=10, num_workers=0, seed=7)

    assert first.dataset.indices == second.dataset.indices


def test_normalize_data_computes_stats():
    data = torch.randn(1000, 5) * 3 + 7
    normalized, mean, std = normalize_data(data)

    assert torch.allclose(normalized.mean(dim=0), torch.zeros(5), atol=1e-5)
    assert torch.allclose(normalized.std(dim=0), torch.ones(5), atol=1e-1)
    assert mean.shape == (5,)
    assert std.shape == (5,)


def test_normalize_data_provided_stats():
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    mean = torch.zeros(3)
    std = torch.ones(3)
    normalized, out_mean, out_std = normalize_data(data, mean=mean, std=std)

    assert torch.equal(normalized, data)
    assert torch.equal(out_mean, mean)
    assert torch.equal(out_std, std)


def test_normalize_data_zero_std():
    data = torch.ones(6, 2)
    normalized, _, std = normalize_data(data)

    assert torch.isfinite(normalized).all()
    assert (std != 0).all()
