"""
Visualization utilities for training metrics and results.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        metrics_history: Dictionary mapping metric names to lists of values.
        save_path: Optional path to save the plot.
        figsize: Figure size tuple.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        axes[0].plot(metrics_history['train_loss'], label='Train Loss')
        axes[0].plot(metrics_history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    if 'train_acc' in metrics_history and 'val_acc' in metrics_history:
        axes[1].plot(metrics_history['train_acc'], label='Train Acc')
        axes[1].plot(metrics_history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array.
        class_names: Optional list of class names.
        save_path: Optional path to save the plot.
        figsize: Figure size tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_rate(
    lr_history: List[float],
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 5)
) -> None:
    """
    Plot learning rate schedule.
    
    Args:
        lr_history: List of learning rates per epoch.
        save_path: Optional path to save the plot.
        figsize: Figure size tuple.
    """
    plt.figure(figsize=figsize)
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
