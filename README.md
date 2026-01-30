# PyTorch Professional Project Template

A production-ready, professionally structured PyTorch project template with comprehensive utilities, logging, checkpointing, and best practices baked in.

## ✨ Features

- 🏗️ **Professional Architecture** - Modular design with clear separation of concerns
- ⚙️ **Configuration Management** - Centralized dataclass-based configuration
- 📊 **Advanced Logging** - Structured logging with TensorBoard integration
- 💾 **Checkpoint Management** - Automated model versioning and best model tracking
- 📈 **Metrics Tracking** - Built-in accuracy, loss, and custom metrics
- 🎯 **Type Hints** - Fully type-hinted codebase for better IDE support
- 📚 **Comprehensive Documentation** - Docstrings throughout, zero code comments
- 🧪 **Unit Tests** - Test suite for all major components
- 🐳 **Docker Support** - Both GPU (CUDA) and CPU containers
- 🎓 **Examples Included** - Custom datasets, transfer learning, and more

## 🆕 What's New

This project has been significantly enhanced with professional features:

- **Configuration System**: Centralized dataclass-based config management
- **Advanced Utilities**: Checkpoint management, metrics tracking, visualization
- **Enhanced Training**: Early stopping, LR scheduling, comprehensive logging
- **Code Quality**: Zero comments, full docstrings, complete type hints
- **Testing**: Unit tests for all major components
- **Documentation**: Architecture guide, usage examples, API docs
- **Examples**: Custom datasets, transfer learning demonstrations

## 📋 Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- For GPU support: [NVIDIA Docker Runtime](https://github.com/NVIDIA/nvidia-docker)
- Python 3.8+ (for local development)

## 🚀 Quick Start

### Using PowerShell (Windows)

```powershell
.\docker-commands.ps1 build
.\docker-commands.ps1 run
.\docker-commands.ps1 shell
```

Once in the container:

```bash
cd /workspace/src
python train.py
```

### Using Docker Compose

```bash
docker-compose up -d pytorch-gpu
docker exec -it pytorch-gpu bash
```

### Local Installation (Without Docker)

```bash
pip install -e .
cd src
python train.py
```

## 📁 Project Structure

```
Pytorch/
├── src/                         # Main source code
│   ├── config.py               # Configuration management
│   ├── logger.py               # Logging utilities
│   ├── train.py                # Training script with early stopping
│   ├── inference.py            # Inference script
│   ├── models/                 # Model definitions
│   │   ├── simple_nn.py       # Simple neural network
│   │   └── __init__.py
│   └── utils/                  # Utility modules
│       ├── checkpoint.py      # Checkpoint management
│       ├── metrics.py         # Metrics calculation
│       ├── model.py           # Model utilities
│       ├── data.py            # Data utilities
│       ├── visualization.py   # Visualization tools
│       └── __init__.py
├── examples/                    # Usage examples
│   ├── custom_dataset.py      # Custom dataset integration
│   └── transfer_learning.py   # Transfer learning example
├── tests/                       # Unit tests
│   ├── test_config.py
│   ├── test_utils.py
│   └── test_models.py
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md        # System architecture
│   └── USAGE.md               # Detailed usage guide
├── data/                        # Dataset directory
├── models/                      # Saved models
├── outputs/                     # Training outputs
│   ├── logs/                  # TensorBoard logs
│   └── checkpoints/           # Model checkpoints
├── notebooks/                   # Jupyter notebooks
├── setup.py                    # Package configuration
├── requirements.txt            # Python dependencies
├── docker-compose.yml         # Docker Compose config
└── README.md                  # This file
```

## 💻 Usage

### Training

```python
from config import get_config
from models import SimpleModel
from logger import setup_logger
import torch.nn as nn

config = get_config()
logger = setup_logger(log_dir=config.paths.logs_dir)

model = SimpleModel(
    input_size=config.model.input_size,
    hidden_size=config.model.hidden_size,
    num_classes=config.model.num_classes
).to(config.device.device)
```

Run training:

```bash
cd src
python train.py
```

### Inference

```bash
cd src
python inference.py
```

### Custom Configuration

```python
from config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(hidden_size=256, num_classes=20),
    training=TrainingConfig(batch_size=128, num_epochs=50)
)
```

## 🎯 Key Features

### Configuration Management

Centralized configuration using dataclasses:

```python
from config import get_config

config = get_config()
config.training.batch_size = 128
config.model.hidden_size = 256
```

### Checkpoint Management

Automatic model versioning:

```python
from utils import CheckpointManager

checkpoint_manager = CheckpointManager(config.paths.checkpoints_dir)
checkpoint_manager.save(model, optimizer, epoch, metrics, is_best=True)
```

### Metrics Tracking

Built-in metrics calculation:

```python
from utils import AverageMeter, calculate_accuracy

loss_meter = AverageMeter('Loss')
acc = calculate_accuracy(outputs, targets)
```

### Logging

Structured logging with TensorBoard:

```python
from logger import setup_logger, MetricsLogger

logger = setup_logger(log_dir=config.paths.logs_dir)
metrics_logger = MetricsLogger(logger)
metrics_logger.log_epoch(epoch, metrics)
```

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and module descriptions
- **[Usage Guide](docs/USAGE.md)** - Detailed usage examples and best practices

## 🧪 Running Tests

```bash
cd tests
pytest -v
```

## 🎓 Examples

### Custom Dataset

```bash
python examples/custom_dataset.py
```

### Transfer Learning

```bash
python examples/transfer_learning.py
```

## 🐳 Docker Commands

### PowerShell (Windows)

```powershell
.\docker-commands.ps1 help         # Show all commands
.\docker-commands.ps1 build        # Build GPU image
.\docker-commands.ps1 build-cpu    # Build CPU image
.\docker-commands.ps1 run          # Run GPU container
.\docker-commands.ps1 run-cpu      # Run CPU container
.\docker-commands.ps1 shell        # Open bash shell
.\docker-commands.ps1 jupyter      # Start Jupyter notebook
.\docker-commands.ps1 tensorboard  # Start TensorBoard
.\docker-commands.ps1 stop         # Stop containers
.\docker-commands.ps1 clean        # Remove containers/images
```

### Makefile (Linux/Mac)

```bash
make help                          # Show all commands
make build                         # Build GPU image
make run                           # Run GPU container
make shell                         # Open bash shell
make jupyter                       # Start Jupyter
make tensorboard                   # Start TensorBoard
make stop                          # Stop containers
make clean                         # Cleanup
```

## 📊 TensorBoard

Start TensorBoard to visualize training:

```powershell
.\docker-commands.ps1 tensorboard
```

Then open: <http://localhost:6006>

## 🔧 Advanced Features

### Early Stopping

Automatic training termination when validation performance plateaus.

### Learning Rate Scheduling

Dynamic learning rate adjustment based on validation metrics.

### Model Utilities

- Parameter counting
- Weight initialization strategies
- Model freezing/unfreezing for transfer learning
- Layer-wise learning rate decay

### Data Utilities

- Custom dataset classes
- Train/val split utilities
- Data normalization helpers

### Visualization

- Training curve plotting
- Confusion matrix visualization
- Learning rate schedule plots

## 🎨 Code Quality

- **Zero Comments**: Self-documenting code with clear naming
- **Type Hints**: Full type annotations for IDE support
- **Docstrings**: Google-style docstrings for all functions
- **PEP 8**: Follows Python style guidelines
- **Modular**: Clear separation of concerns
- **Tested**: Unit tests for critical components

## 🔍 Customization

### Adding Python Packages

Edit `requirements.txt` and rebuild:

```powershell
.\docker-commands.ps1 build
```

### Environment Variables

Copy and edit `.env`:

```powershell
Copy-Item .env.example .env
```

### GPU Configuration

Modify `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
```

## 🐛 Troubleshooting

### GPU Not Detected

1. Ensure NVIDIA Docker runtime is installed
2. Check: `nvidia-smi`
3. Verify: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Port Already in Use

Change port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8889:8888"  # Use different port
```

## ❓ FAQ

### How do I add my own model?

Create a new file in `src/models/` and add it to `src/models/__init__.py`. See `simple_nn.py` for reference.

### How do I use my own dataset?

Check out `examples/custom_dataset.py` for a complete example of integrating custom datasets.

### Where are my trained models saved?

Models are saved in:

- `outputs/checkpoints/` - All checkpoints
- `outputs/checkpoints/best_model.pth` - Best performing model

### How do I resume training?

Use the checkpoint utilities to load a previous checkpoint:

```python
from utils.checkpoint import load_checkpoint
checkpoint = load_checkpoint('outputs/checkpoints/best_model.pth', model, optimizer)
start_epoch = checkpoint['epoch'] + 1
```

### Can I run this without Docker?

Yes! Install with `pip install -e .` and run scripts directly.

### How do I monitor training progress?

Use TensorBoard: `.\docker-commands.ps1 tensorboard` then open <http://localhost:6006>

## 📖 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [Project Architecture](docs/ARCHITECTURE.md)
- [Usage Guide](docs/USAGE.md)

## 📄 License

See LICENSE file for details.

---

**Built with best practices for production ML projects**
