"""
PyTorch project package.
"""

__version__ = "1.0.0"
__author__ = "PyTorch Project"

from .config import Config, get_config
from .logger import MetricsLogger, setup_logger

__all__ = ["get_config", "Config", "setup_logger", "MetricsLogger", "__version__"]
