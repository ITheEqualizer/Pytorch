.PHONY: help build build-cpu run run-cpu stop clean jupyter tensorboard shell

help:
	@echo "Available commands:"
	@echo "  make build         - Build GPU Docker image"
	@echo "  make build-cpu     - Build CPU Docker image"
	@echo "  make run           - Run GPU container"
	@echo "  make run-cpu       - Run CPU container"
	@echo "  make stop          - Stop all containers"
	@echo "  make clean         - Remove containers and images"
	@echo "  make jupyter       - Start Jupyter notebook in GPU container"
	@echo "  make jupyter-cpu   - Start Jupyter notebook in CPU container"
	@echo "  make tensorboard   - Start TensorBoard in GPU container"
	@echo "  make shell         - Open bash shell in GPU container"
	@echo "  make shell-cpu     - Open bash shell in CPU container"

build:
	docker-compose build pytorch-gpu

build-cpu:
	docker-compose build pytorch-cpu

run:
	docker-compose up -d pytorch-gpu

run-cpu:
	docker-compose up -d pytorch-cpu

stop:
	docker-compose down

clean:
	docker-compose down -v --rmi all

jupyter:
	docker exec -it pytorch-gpu jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

jupyter-cpu:
	docker exec -it pytorch-cpu jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard:
	docker exec -it pytorch-gpu tensorboard --logdir=/workspace/outputs --host=0.0.0.0 --port=6006

shell:
	docker exec -it pytorch-gpu bash

shell-cpu:
	docker exec -it pytorch-cpu bash
