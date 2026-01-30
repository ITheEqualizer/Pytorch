"""
Configuration management for PyTorch project.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class PathConfig:
    """Configuration for project paths."""
    
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.outputs_dir = self.project_root / "outputs"
        self.logs_dir = self.outputs_dir / "logs"
        self.checkpoints_dir = self.outputs_dir / "checkpoints"
        
        self._create_directories()
    
    def _create_directories(self) -> None:
        for directory in [self.data_dir, self.models_dir, self.outputs_dir, 
                         self.logs_dir, self.checkpoints_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    input_size: int = 784
    hidden_size: int = 128
    num_classes: int = 10
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_workers: int = 4
    pin_memory: bool = True
    gradient_clip: Optional[float] = 1.0
    early_stopping_patience: int = 5
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5


@dataclass
class DeviceConfig:
    """Configuration for compute device."""
    
    use_cuda: bool = True
    device: torch.device = field(init=False)
    
    def __post_init__(self):
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    seed: int = 42
    
    def __post_init__(self):
        self._set_seed()
    
    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


def get_config() -> Config:
    """
    Get the default configuration.
    
    Returns:
        Config instance with default settings.
    """
    return Config()
