.PHONY: help build build-gpu dev train-cpu train-gpu test clean shell jupyter tensorboard

# Default target
help:
	@echo "IARA Deep Learning Project - Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make build         Build CPU Docker image"
	@echo "  make build-gpu     Build GPU Docker image"
	@echo "  make build-all     Build all images"
	@echo ""
	@echo "Development:"
	@echo "  make dev           Start Jupyter Lab (http://localhost:8888)"
	@echo "  make shell         Start interactive bash shell (CPU)"
	@echo "  make shell-gpu     Start interactive bash shell (GPU)"
	@echo ""
	@echo "Training:"
	@echo "  make train-cpu     Start CPU training environment"
	@echo "  make train-gpu     Start GPU training environment"
	@echo "  make train-prod    Run production training (UV, with caching)"
	@echo "  make train-prod-all Run production training with all DCs (UV)"
	@echo "  make train-prod-docker Run production training (Docker CPU)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run unit tests"
	@echo "  make test-demo     Run demo mode (requires data)"
	@echo ""
	@echo "Utilities:"
	@echo "  make tensorboard   Start TensorBoard (http://localhost:6009)"
	@echo "  make mlflow        Start MLflow UI (http://localhost:5000)"
	@echo "  make clean         Stop and remove all containers"
	@echo "  make clean-all     Clean containers, images, and volumes"
	@echo ""
	@echo "UV Commands (local):"
	@echo "  make uv-venv       Create virtual environment"
	@echo "  make uv-install    Install dependencies with uv"
	@echo "  make uv-install-dev Install dev dependencies"
	@echo "  make uv-clean      Remove virtual environment"
	@echo "  make uv-sync       Sync dependencies"
	@echo "  make uv-lock       Update lockfile"

# Build targets
build:
	docker-compose build dev train-cpu

build-gpu:
	docker-compose build train-gpu

build-all:
	docker-compose build

# Development targets
dev:
	@echo "Starting Jupyter Lab..."
	@echo "Access at: http://localhost:8888"
	docker-compose up -d dev
	@echo "Logs: docker-compose logs -f dev"

shell:
	docker-compose run --rm train-cpu bash

shell-gpu:
	docker-compose run --rm train-gpu bash

# Training targets
train-cpu:
	docker-compose run --rm train-cpu bash

train-gpu:
	docker-compose run --rm train-gpu bash

# Production training targets (with caching)
train-prod:
	@echo "Starting production training (UV)..."
	@echo "Using H DC only, MEL features, with smart caching"
	.venv/bin/python scripts/train_production.py

train-prod-all:
	@echo "Starting production training with ALL DCs (UV)..."
	@echo "This will take longer but trains on full dataset"
	.venv/bin/python scripts/train_production.py --all-dcs

train-prod-docker:
	@echo "Starting production training (Docker)..."
	docker-compose run --rm train-cpu python3 scripts/train_production.py

train-prod-docker-all:
	@echo "Starting production training with ALL DCs (Docker)..."
	docker-compose run --rm train-cpu python3 scripts/train_production.py --all-dcs

# Hyperparameter tuning targets
tune:
	@echo "Starting hyperparameter tuning (UV)..."
	@echo "Using A,B,C,D DCs for multi-class training"
	.venv/bin/python scripts/tune_hyperparameters.py

tune-all:
	@echo "Starting hyperparameter tuning with ALL DCs (UV)..."
	.venv/bin/python scripts/tune_hyperparameters.py --all-dcs

tune-docker:
	@echo "Starting hyperparameter tuning (Docker)..."
	docker-compose run --rm train-cpu python3 scripts/tune_hyperparameters.py

tune-docker-all:
	@echo "Starting hyperparameter tuning with ALL DCs (Docker)..."
	docker-compose run --rm train-cpu python3 scripts/tune_hyperparameters.py --all-dcs

# Testing targets
test:
	docker-compose run --rm test

test-demo:
	docker-compose run --rm train-cpu python test_dataset.py --mode demo \
		--data-dir data/iara \
		--csv-path IARA/src/iara/dataset_info/iara.csv \
		--num-samples 3

# Utility targets
jupyter:
	@echo "Jupyter Lab running at: http://localhost:8888"
	docker-compose logs -f dev

tensorboard:
	@echo "Starting TensorBoard..."
	@echo "Access at: http://localhost:6009"
	docker-compose up -d tensorboard

mlflow:
	@echo "Starting MLflow UI..."
	@echo "Access at: http://localhost:5000"
	docker-compose up -d mlflow

# Cleanup targets
stop:
	docker-compose stop

clean:
	docker-compose down

clean-all:
	docker-compose down -v --rmi all

# UV commands (for local development without Docker)
uv-venv:
	@echo "Creating virtual environment with uv..."
	uv venv

uv-install:
	@echo "Installing dependencies with uv..."
	uv pip install -r requirements.txt
	@echo "✓ Dependencies installed"

uv-install-dev:
	@echo "Installing dev dependencies with uv..."
	uv pip install -r requirements.txt
	uv pip install -e ".[dev,experiment]"
	@echo "✓ Dev dependencies installed"

uv-sync:
	uv pip sync

uv-lock:
	uv pip compile pyproject.toml -o requirements.txt

uv-clean:
	@echo "Removing virtual environment..."
	rm -rf .venv
	@echo "✓ Virtual environment removed"

# Show running containers
ps:
	docker-compose ps

# View logs
logs:
	docker-compose logs -f

logs-dev:
	docker-compose logs -f dev

logs-train:
	docker-compose logs -f train-cpu

# Exec into running container
exec-dev:
	docker-compose exec dev bash

exec-train:
	docker-compose exec train-cpu bash
