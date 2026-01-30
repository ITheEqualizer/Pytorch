"""
PyTorch project package.
"""
__version__ = '1.0.0'
__author__ = 'PyTorch Project'

from .config import get_config, Config
from .logger import setup_logger, MetricsLogger

__all__ = [
    'get_config',
    'Config',
    'setup_logger',
    'MetricsLogger',
    '__version__'
]
