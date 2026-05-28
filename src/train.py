"""
Training script for PyTorch models with comprehensive logging and checkpointing.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, get_config
from logger import MetricsLogger, setup_logger
from models import SimpleModel
from utils import AverageMeter, CheckpointManager, calculate_accuracy
from utils.model import print_model_summary

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore[assignment]


def create_grad_scaler(device: torch.device, enabled: bool) -> GradScaler:
    """
    Create a gradient scaler compatible across PyTorch versions.

    Args:
        device: Device the model trains on.
        enabled: Whether mixed-precision gradient scaling is active.

    Returns:
        A GradScaler. A disabled scaler behaves like full-precision training.
    """
    try:
        return GradScaler(device=device.type, enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    gradient_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: Neural network model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer for parameter updates.
        device: Device to train on (CPU or CUDA).
        epoch: Current epoch number.
        scaler: Gradient scaler for mixed-precision training. A disabled
            scaler makes the step behave like full-precision training.
        gradient_clip: Maximum gradient norm for clipping. No clipping when None.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Accuracy")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(outputs, targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        pbar.set_postfix(
            {"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2f}%"}
        )

    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.

    Args:
        model: Neural network model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on (CPU or CUDA).

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Accuracy")

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc = calculate_accuracy(outputs, targets)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


def create_dummy_data(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy datasets for demonstration.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_data = torch.randn(1000, 1, 28, 28)
    train_labels = torch.randint(0, config.model.num_classes, (1000,))
    val_data = torch.randn(200, 1, 28, 28)
    val_labels = torch.randint(0, config.model.num_classes, (200,))

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    pin_memory = config.training.pin_memory and config.device.device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=config.training.drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def main() -> None:
    """Main training function."""
    config = get_config()

    logger = setup_logger(log_dir=config.paths.logs_dir)
    metrics_logger = MetricsLogger(logger)

    logger.info(f"Using device: {config.device.device}")
    logger.info(f"Configuration: {config}")

    train_loader, val_loader = create_dummy_data(config)

    model = SimpleModel(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
    ).to(config.device.device)

    if config.training.compile_model:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed, using eager model: {e}")

    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    use_amp = config.training.use_amp and config.device.device.type == "cuda"
    scaler = create_grad_scaler(config.device.device, enabled=use_amp)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience,
    )

    writer = SummaryWriter(log_dir=str(config.paths.logs_dir))
    checkpoint_manager = CheckpointManager(config.paths.checkpoints_dir)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config.training.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.training.num_epochs}")

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device.device,
            epoch,
            scaler=scaler,
            gradient_clip=config.training.gradient_clip,
        )

        val_loss, val_acc = validate(model, val_loader, criterion, config.device.device)

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        metrics_logger.log_epoch(epoch, metrics)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(val_acc)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
            metrics_logger.log_best_model(epoch, "val_acc", val_acc)
        else:
            patience_counter += 1

        checkpoint_manager.save(model, optimizer, epoch, metrics, is_best=is_best)

        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    writer.close()
    logger.info(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
