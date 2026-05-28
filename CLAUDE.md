# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyTorch project **template** (not an application with real data). `src/train.py` and `src/inference.py` run end-to-end on randomly generated dummy data (`create_dummy_data`, `torch.randn` dummy input) so the pipeline works out of the box. Wiring in a real dataset/model is the intended first customization.

## Import convention (read before editing or running anything)

Modules inside `src/` import each other as **flat top-level modules**, not as a package:

```python
from config import get_config
from models import SimpleModel
from utils import CheckpointManager, AverageMeter, calculate_accuracy
from utils.model import print_model_summary
```

There are no `src.` prefixes and no relative imports. Consequences:

- Scripts only run with `src/` as the working directory or on `PYTHONPATH`. Always `cd src && python train.py` (same for `inference.py`). Running `python src/train.py` from the repo root fails with `ModuleNotFoundError`.
- There are **no `console_scripts` entry points** — the flat-import layout makes them unresolvable (importing `src.train` triggers `from config import ...` which has no top-level `config` module). Don't add them; run via `cd src && python train.py`.
- Keep the flat style when adding modules under `src/`. New public symbols should be re-exported via `src/models/__init__.py` or `src/utils/__init__.py` (`__all__`) — that curated surface is the intended public API.

## Common commands

Local (no Docker):
```bash
pip install -e .              # installs torch + core deps from pyproject.toml
cd src && python train.py     # train (dummy data); writes to ../outputs/
cd src && python inference.py # loads outputs/checkpoints/best_model.pth
```

Tests (each test file inserts `../src` onto `sys.path`, so any CWD works):
```bash
pytest -v                                          # from repo root
pytest tests/test_config.py -v                     # one file
pytest tests/test_config.py::test_get_config -v    # one test
```

Lint/format/type-check (dev extras; config in `pyproject.toml` + `.flake8`):
```bash
pip install -e ".[dev]"
black src tests        # line-length 88
isort src tests        # black profile
flake8 src tests
mypy src
```

Docker (GPU via `make`, or `docker-commands.ps1` on Windows):
```bash
make build && make run && make shell   # then inside: cd /workspace/src && python train.py
make tensorboard                       # http://localhost:6006
make jupyter                           # http://localhost:8888
make build-cpu / make run-cpu / make shell-cpu   # CPU variants (Dockerfile.cpu)
```

## Dependency gotcha

Two dependency lists that disagree, on purpose:

- **`pyproject.toml`** `dependencies` include `torch` + a minimal core set — use this for local dev (`pip install -e .` / `pip install -e .[dev]`).
- **`requirements.txt`** does **not** list `torch` (it assumes the `pytorch/pytorch:2.1.0-cuda12.1` Docker base image already provides it) and adds heavier extras (`transformers`, `timm`, `opencv-python`, `albumentations`, `pandas`, etc.). `pip install -r requirements.txt` alone will **not** give you torch.

Overlapping packages (`torchvision`, `tensorboard`, `tqdm`) are kept in sync between the two. When changing dependencies, update whichever list matches the install path you care about (often both).

## Architecture

Everything is configuration-driven through a single nested dataclass tree in [src/config.py](src/config.py). `get_config()` returns a `Config` composed of `PathConfig`, `ModelConfig`, `TrainingConfig`, and `DeviceConfig`. Override by constructing sub-configs explicitly (e.g. `Config(model=ModelConfig(hidden_size=256))`).

**`get_config()` / `Config()` has side effects** — construction runs `PathConfig._create_directories()` (creates `data/`, `models/`, `outputs/`, `outputs/logs/`, `outputs/checkpoints/`) and `Config._set_seed()` (seeds `random`, `numpy`, `torch`, and CUDA with `seed=42`). `DeviceConfig` auto-selects CUDA when available (and enables TF32 + `cudnn.benchmark`), else CPU.

Efficiency flags (all CPU-safe; default to current behavior): `TrainingConfig.use_amp` (mixed precision, CUDA-only, default off), `TrainingConfig.compile_model` (`torch.compile`, default off, falls back to eager on failure), `TrainingConfig.gradient_clip` (now applied in `train_epoch`, default `1.0`), `TrainingConfig.drop_last`, and `Config.deterministic` (disables TF32/benchmark and enables deterministic algorithms for reproducibility).

`src/train.py` is the orchestrator and the place to understand the data flow:
1. `get_config()` → logger ([src/logger.py](src/logger.py): file+console logging plus `MetricsLogger` for epoch metric history) → `SummaryWriter` (TensorBoard) → `CheckpointManager`.
2. Build data (`create_dummy_data` — replace this), `SimpleModel`, `CrossEntropyLoss`, `Adam`, `ReduceLROnPlateau`.
3. Epoch loop: `train_epoch` / `validate` return `(loss, acc)` tracked via `AverageMeter`; metrics logged to MetricsLogger + TensorBoard; scheduler steps on `val_acc`; checkpoint saved every epoch with `is_best` flag; **early stopping** on `early_stopping_patience` epochs without `val_acc` improvement.

Checkpoints land in `outputs/checkpoints/`, best as `best_model.pth`. `inference.py` loads `outputs/checkpoints/best_model.pth` (fallback `models/best_model.pth`).

Module map: `src/models/` (architectures, start from `simple_nn.py`), `src/utils/` (`checkpoint.py` versioning, `metrics.py` `AverageMeter`/`calculate_accuracy`, `model.py` param-count/freeze/init helpers, `data.py`, `visualization.py`). `examples/` shows custom-dataset and transfer-learning patterns.

## Docker volume caveat

`docker-compose.yml` mounts only `src/`, `data/`, `models/`, `outputs/`, `notebooks/` into the container. Edits to `tests/`, `examples/`, `requirements.txt`, or the Dockerfiles on the host are **not** reflected in a running container — rebuild (`make build`) to pick those up.

## Code style

Match the existing style: **no inline comments** (self-documenting names), Google-style docstrings on every public function/class, full type hints, dataclasses for configuration. `black` (line-length 88), `isort` (black profile), `flake8`, and `mypy` are the dev tools — configured in `pyproject.toml` and `.flake8`. CI (`.github/workflows/ci.yml`) runs all four plus `pytest` on Python 3.9–3.12.
