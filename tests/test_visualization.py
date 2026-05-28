"""
Unit tests for visualization utilities.

These rely on the Agg matplotlib backend configured in conftest.py so that
plt.show() never blocks in a headless environment.
"""

import numpy as np

from utils.visualization import (
    plot_confusion_matrix,
    plot_learning_rate,
    plot_training_curves,
)


def test_plot_training_curves_saves(tmp_path):
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "train_acc": [40.0, 60.0, 75.0],
        "val_acc": [38.0, 58.0, 72.0],
    }
    save_path = tmp_path / "curves.png"
    plot_training_curves(history, save_path=save_path)
    assert save_path.exists()


def test_plot_training_curves_partial():
    history = {"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}
    plot_training_curves(history)


def test_plot_confusion_matrix_saves(tmp_path):
    cm = np.array([[5, 1], [2, 7]])
    save_path = tmp_path / "cm.png"
    plot_confusion_matrix(cm, class_names=["a", "b"], save_path=save_path)
    assert save_path.exists()


def test_plot_confusion_matrix_no_names(tmp_path):
    cm = np.array([[3, 0], [1, 4]])
    save_path = tmp_path / "cm_no_names.png"
    plot_confusion_matrix(cm, save_path=save_path)
    assert save_path.exists()


def test_plot_learning_rate_saves(tmp_path):
    save_path = tmp_path / "lr.png"
    plot_learning_rate([0.1, 0.05, 0.025, 0.0125], save_path=save_path)
    assert save_path.exists()
