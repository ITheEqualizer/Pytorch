# PowerShell script for Docker commands on Windows

function Show-Help {
    Write-Host "Available commands:" -ForegroundColor Green
    Write-Host "  .\docker-commands.ps1 build         - Build GPU Docker image"
    Write-Host "  .\docker-commands.ps1 build-cpu     - Build CPU Docker image"
    Write-Host "  .\docker-commands.ps1 run           - Run GPU container"
    Write-Host "  .\docker-commands.ps1 run-cpu       - Run CPU container"
    Write-Host "  .\docker-commands.ps1 stop          - Stop all containers"
    Write-Host "  .\docker-commands.ps1 clean         - Remove containers and images"
    Write-Host "  .\docker-commands.ps1 jupyter       - Start Jupyter notebook in GPU container"
    Write-Host "  .\docker-commands.ps1 jupyter-cpu   - Start Jupyter notebook in CPU container"
    Write-Host "  .\docker-commands.ps1 tensorboard   - Start TensorBoard in GPU container"
    Write-Host "  .\docker-commands.ps1 shell         - Open bash shell in GPU container"
    Write-Host "  .\docker-commands.ps1 shell-cpu     - Open bash shell in CPU container"
}

$command = $args[0]

switch ($command) {
    "build" {
        docker-compose build pytorch-gpu
    }
    "build-cpu" {
        docker-compose build pytorch-cpu
    }
    "run" {
        docker-compose up -d pytorch-gpu
    }
    "run-cpu" {
        docker-compose up -d pytorch-cpu
    }
    "stop" {
        docker-compose down
    }
    "clean" {
        docker-compose down -v --rmi all
    }
    "jupyter" {
        docker exec -it pytorch-gpu jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    }
    "jupyter-cpu" {
        docker exec -it pytorch-cpu jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    }
    "tensorboard" {
        docker exec -it pytorch-gpu tensorboard --logdir=/workspace/outputs --host=0.0.0.0 --port=6006
    }
    "shell" {
        docker exec -it pytorch-gpu bash
    }
    "shell-cpu" {
        docker exec -it pytorch-cpu bash
    }
    default {
        Show-Help
    }
}
