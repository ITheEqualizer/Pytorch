# Usage Guide

## Quick Start

### Training a Model

```python
from config import get_config
from models import SimpleModel
from logger import setup_logger
import torch.nn as nn
import torch.optim as optim

config = get_config()
logger = setup_logger(log_dir=config.paths.logs_dir)

model = SimpleModel(
    input_size=config.model.input_size,
    hidden_size=config.model.hidden_size,
    num_classes=config.model.num_classes
).to(config.device.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
```

### Running Training

```bash
cd src
python train.py
```

### Running Inference

```bash
cd src
python inference.py
```

## Custom Dataset Integration

### Creating a Custom Dataset

```python
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = self.load_data(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, label = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
    def load_data(self, path):
        pass
```

### Using Custom Dataset in Training

```python
from torch.utils.data import DataLoader

dataset = MyDataset('path/to/data')
train_loader = DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.training.num_workers
)
```

## Creating Custom Models

### Define Your Model

```python
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyCustomModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
```

### Register in Models Module

Add to `src/models/__init__.py`:

```python
from .my_custom_model import MyCustomModel

__all__ = ['SimpleModel', 'MyCustomModel']
```

## Configuration Customization

### Modify Default Configuration

```python
from config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(
        input_size=1024,
        hidden_size=256,
        num_classes=20
    ),
    training=TrainingConfig(
        batch_size=128,
        num_epochs=50,
        learning_rate=0.0001
    )
)
```

### Using Environment-Specific Configs

```python
from config import get_config

config = get_config()

config.training.batch_size = 32
config.training.num_epochs = 100
```

## Checkpoint Management

### Save Checkpoints

```python
from utils import CheckpointManager

checkpoint_manager = CheckpointManager(config.paths.checkpoints_dir)

checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={'val_acc': 95.5},
    is_best=True
)
```

### Load Checkpoints

```python
from utils.checkpoint import load_checkpoint

checkpoint = load_checkpoint(
    filepath='outputs/checkpoints/best_model.pth',
    model=model,
    optimizer=optimizer
)

start_epoch = checkpoint['epoch']
metrics = checkpoint['metrics']
```

## Logging and Monitoring

### Setup Logging

```python
from logger import setup_logger, MetricsLogger

logger = setup_logger(name='my_experiment', log_dir=config.paths.logs_dir)
metrics_logger = MetricsLogger(logger)
```

### Log Training Metrics

```python
metrics = {
    'train_loss': 0.234,
    'train_acc': 94.2,
    'val_loss': 0.312,
    'val_acc': 92.1
}

metrics_logger.log_epoch(epoch=5, metrics=metrics)
```

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=str(config.paths.logs_dir))

writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.add_histogram('weights/fc1', model.fc1.weight, epoch)

writer.close()
```

View TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

## Advanced Techniques

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    scheduler.step(val_acc)
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Model Freezing/Unfreezing

```python
from utils.model import freeze_model, unfreeze_model

freeze_model(model, layer_names=['encoder'])

unfreeze_model(model)
```

## Best Practices

1. **Always use configuration objects** instead of hard-coding parameters
2. **Log extensively** to track experiments
3. **Save checkpoints regularly** to avoid losing progress
4. **Use TensorBoard** for visualization
5. **Type hint** all function signatures
6. **Document with docstrings**, not comments
7. **Test incrementally** as you build
