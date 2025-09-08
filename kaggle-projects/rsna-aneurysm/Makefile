# RSNA Aneurysm Detection - Development Makefile
# 
# Common development tasks and commands for the medical imaging ML project

.PHONY: help test test-fast test-coverage lint format security install clean docs

# Default target
help:
	@echo "RSNA Aneurysm Detection - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests with verbose output"
	@echo "  test-fast     Run only unit tests (skip slow/integration)"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  test-security Run security-focused tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Run linting (ruff) and type checking"
	@echo "  format        Format code with black and ruff"
	@echo "  security      Run security analysis with bandit"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install dependencies and pre-commit hooks"
	@echo "  install-dev   Install dev dependencies including test tools"
	@echo "  clean         Clean temporary files and caches"
	@echo ""
	@echo "Data:"
	@echo "  data-pipeline Run DVC data processing pipeline"
	@echo "  data-status   Check data pipeline status"

# Testing
test:
	@echo "🧪 Running all tests..."
	pytest tests/ -v --tb=short

test-fast:
	@echo "⚡ Running fast tests only..."
	pytest tests/ -v -m "not slow and not integration" --tb=short

test-coverage:
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=scripts --cov-report=html --cov-report=term-missing

test-security:
	@echo "🛡️  Running security tests..."
	pytest tests/ -v -m security --tb=short

test-watch:
	@echo "👀 Running tests in watch mode..."
	pytest-watch tests/ --runner="pytest -v --tb=short"

# Code Quality
lint:
	@echo "🔍 Running linting..."
	ruff check scripts/ tests/
	@echo "✅ Linting completed"

format:
	@echo "🎨 Formatting code..."
	black scripts/ tests/
	ruff check --fix scripts/ tests/
	@echo "✅ Code formatted"

security:
	@echo "🛡️  Running security analysis..."
	bandit -r scripts/ -f json -o security-report.json || true
	bandit -r scripts/ --severity-level medium
	@echo "✅ Security analysis completed"

type-check:
	@echo "🔍 Running type checking..."
	mypy scripts/ --ignore-missing-imports --no-strict-optional
	@echo "✅ Type checking completed"

# Setup and Installation
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "🪝 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Installation completed"

install-dev:
	@echo "📦 Installing dev dependencies..."
	pip install -r requirements.txt
	pip install pytest-watch mypy bandit
	pre-commit install
	@echo "✅ Dev installation completed"

install-gpu:
	@echo "🚀 Installing GPU dependencies..."
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt
	@echo "✅ GPU installation completed"

# Cleaning
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	find . -type f -name ".coverage" -delete || true
	find . -type d -name "htmlcov" -exec rm -rf {} + || true
	@echo "✅ Cleanup completed"

clean-data:
	@echo "🧹 Cleaning data cache..."
	rm -rf data/processed/*.png || true
	rm -rf data/processed/*.npy || true
	dvc cache dir --unset || true
	@echo "✅ Data cleanup completed"

# Data Pipeline
data-pipeline:
	@echo "📊 Running DVC data pipeline..."
	dvc repro
	@echo "✅ Data pipeline completed"

data-status:
	@echo "📊 Checking data pipeline status..."
	dvc status
	dvc metrics show

data-pull:
	@echo "📥 Pulling data from remote..."
	dvc pull

data-push:
	@echo "📤 Pushing data to remote..."
	dvc push

# Model Training (example)
train-baseline:
	@echo "🚂 Training baseline model..."
	cd experiments/exp0001 && python training.py

# Pre-commit
pre-commit:
	@echo "🪝 Running pre-commit hooks..."
	pre-commit run --all-files

# Documentation
docs-security:
	@echo "📖 Generating security documentation..."
	@echo "Security report available in security-report.json"
	@echo "Run 'make security' to update the report"

# Environment info
info:
	@echo "🔧 Environment Information"
	@echo "========================="
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not installed"
	@python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA: Unknown"
	@echo "Git Branch: $$(git branch --show-current 2>/dev/null || echo 'Unknown')"
	@echo "Git SHA: $$(git rev-parse --short HEAD 2>/dev/null || echo 'Unknown')"

# Quality checks (CI/CD)
ci-test:
	@echo "🤖 Running CI test suite..."
	pytest tests/ -v --tb=short --cov=scripts --cov-report=xml
	ruff check scripts/ tests/
	bandit -r scripts/ --severity-level medium

# Development workflow
dev-setup: clean install-dev
	@echo "🚀 Development environment ready!"
	@echo "Next steps:"
	@echo "  1. Run 'make test' to verify setup"
	@echo "  2. Run 'make data-pipeline' to process data"
	@echo "  3. Start developing with 'make test-watch'"

# Quick development cycle  
dev-cycle: format lint test-fast
	@echo "✅ Development cycle completed!"

# Production readiness check
prod-check: format lint security test
	@echo "🚀 Production readiness check completed!"