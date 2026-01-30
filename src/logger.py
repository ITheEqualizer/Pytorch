"""
Centralized logging configuration for the project.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "pytorch_project",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger.
        log_dir: Directory to save log files. If None, only console logging is used.
        level: Logging level.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"training_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics logger.
        
        Args:
            logger: Base logger instance to use.
        """
        self.logger = logger
        self.metrics_history = {}
    
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names and values.
        """
        metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metric_str}")
        
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def log_best_model(self, epoch: int, metric: str, value: float) -> None:
        """
        Log when a new best model is saved.
        
        Args:
            epoch: Epoch number.
            metric: Metric name that improved.
            value: New best value.
        """
        self.logger.info(f"New best model at epoch {epoch} - {metric}: {value:.4f}")
    
    def get_history(self, metric: str) -> list:
        """
        Get history of a specific metric.
        
        Args:
            metric: Name of the metric.
        
        Returns:
            List of metric values across epochs.
        """
        return self.metrics_history.get(metric, [])
