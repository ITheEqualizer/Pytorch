"""
Unit tests for the metrics helpers not covered by test_utils.
"""

import torch

from utils.metrics import (
    calculate_accuracy,
    calculate_top_k_accuracy,
    get_predictions_and_labels,
)


def test_top_k_accuracy_k1_matches_top1(classification_batch):
    outputs, targets = classification_batch
    top1 = calculate_accuracy(outputs, targets)
    topk = calculate_top_k_accuracy(outputs, targets, k=1)
    assert topk == top1


def test_top_k_accuracy_includes_runner_up():
    outputs = torch.tensor([[1.0, 2.0, 0.0]])
    targets = torch.tensor([0])

    assert calculate_accuracy(outputs, targets) == 0.0
    assert calculate_top_k_accuracy(outputs, targets, k=1) == 0.0
    assert calculate_top_k_accuracy(outputs, targets, k=2) == 100.0


def test_get_predictions_and_labels(classification_batch):
    outputs, targets = classification_batch
    predictions, labels = get_predictions_and_labels(outputs, targets)

    assert torch.equal(predictions, outputs.argmax(dim=1))
    assert torch.equal(labels, targets)
    assert predictions.device.type == "cpu"
    assert labels.device.type == "cpu"
