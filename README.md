# PyTorch Docker Project

A complete, production-ready Docker setup for PyTorch-based machine learning projects.

## Features

- 🐳 Docker and Docker Compose setup
- 🚀 Both GPU (CUDA) and CPU support
- 📊 Jupyter Notebook integration
- 📈 TensorBoard support
- 🔧 Pre-configured with popular ML libraries
- 📦 Easy-to-use PowerShell scripts for Windows
- 🎯 Example training and inference scripts

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- For GPU support: [NVIDIA Docker Runtime](https://github.com/NVIDIA/nvidia-docker)

## Quick Start

### Using PowerShell (Windows)

1. **Build the Docker image:**
   ```powershell
   # For GPU support
   .\docker-commands.ps1 build
   
   # For CPU only
   .\docker-commands.ps1 build-cpu
   ```

2. **Run the container:**
   ```powershell
   # For GPU
   .\docker-commands.ps1 run
   
   # For CPU
   .\docker-commands.ps1 run-cpu
   ```

3. **Access the container:**
   ```powershell
   # Open a bash shell
   .\docker-commands.ps1 shell
   
   # Or for CPU container
   .\docker-commands.ps1 shell-cpu
   ```

### Using Docker Compose Directly

```bash
# Build and run GPU container
docker-compose up -d pytorch-gpu

# Build and run CPU container
docker-compose up -d pytorch-cpu

# Stop containers
docker-compose down
```

## Project Structure

```
Pytorch/
├── Dockerfile              # GPU-enabled Docker image
├── Dockerfile.cpu          # CPU-only Docker image
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── docker-commands.ps1     # PowerShell helper scripts
├── Makefile               # Make commands (Linux/Mac)
├── .dockerignore          # Files to exclude from Docker build
├── .env.example           # Environment variables template
├── src/
│   ├── train.py           # Training script example
│   └── inference.py       # Inference script example
├── notebooks/
│   └── example_notebook.ipynb  # Jupyter notebook example
├── data/                  # Data directory (mounted volume)
├── models/                # Saved models (mounted volume)
└── outputs/               # Training outputs (mounted volume)
```

## Available Commands

### PowerShell Commands (Windows)

```powershell
.\docker-commands.ps1 help          # Show all available commands
.\docker-commands.ps1 build         # Build GPU Docker image
.\docker-commands.ps1 build-cpu     # Build CPU Docker image
.\docker-commands.ps1 run           # Run GPU container
.\docker-commands.ps1 run-cpu       # Run CPU container
.\docker-commands.ps1 stop          # Stop all containers
.\docker-commands.ps1 clean         # Remove containers and images
.\docker-commands.ps1 jupyter       # Start Jupyter notebook (GPU)
.\docker-commands.ps1 jupyter-cpu   # Start Jupyter notebook (CPU)
.\docker-commands.ps1 tensorboard   # Start TensorBoard
.\docker-commands.ps1 shell         # Open bash shell in GPU container
.\docker-commands.ps1 shell-cpu     # Open bash shell in CPU container
```

### Makefile Commands (Linux/Mac)

```bash
make help          # Show all available commands
make build         # Build GPU Docker image
make build-cpu     # Build CPU Docker image
make run           # Run GPU container
make run-cpu       # Run CPU container
make stop          # Stop all containers
make clean         # Remove containers and images
make jupyter       # Start Jupyter notebook (GPU)
make tensorboard   # Start TensorBoard
make shell         # Open bash shell in container
```

## Using Jupyter Notebooks

1. Start Jupyter server:
   ```powershell
   .\docker-commands.ps1 jupyter
   ```

2. Open your browser and navigate to `http://localhost:8888`

3. Use the token displayed in the terminal to log in

## Using TensorBoard

1. Start TensorBoard:
   ```powershell
   .\docker-commands.ps1 tensorboard
   ```

2. Open your browser and navigate to `http://localhost:6006`

## Running Training Scripts

Inside the container:

```bash
cd /workspace/src
python train.py
```

Or from outside the container:

```powershell
docker exec -it pytorch-gpu python /workspace/src/train.py
```

## Running Inference

```bash
cd /workspace/src
python inference.py
```

## Customization

### Adding Python Packages

Add packages to `requirements.txt` and rebuild the image:

```powershell
.\docker-commands.ps1 build
```

### Environment Variables

1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` with your settings

3. Update `docker-compose.yml` to use the env file:
   ```yaml
   env_file:
     - .env
   ```

### GPU Configuration

To use specific GPUs, modify the `CUDA_VISIBLE_DEVICES` environment variable in `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

## Volumes

The following directories are mounted as volumes:

- `./src` → `/workspace/src` - Source code
- `./data` → `/workspace/data` - Datasets
- `./models` → `/workspace/models` - Saved models
- `./outputs` → `/workspace/outputs` - Training outputs, logs
- `./notebooks` → `/workspace/notebooks` - Jupyter notebooks

## Troubleshooting

### GPU Not Detected

1. Ensure NVIDIA Docker runtime is installed
2. Check GPU availability: `nvidia-smi`
3. Verify Docker can access GPU: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Port Already in Use

Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8889:8888"  # Use port 8889 instead of 8888
```

### Permission Issues

On Windows, ensure Docker has permission to access your project directory.

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

## License

See LICENSE file for details.
