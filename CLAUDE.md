# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is a Kaggle competition template following grandmaster-level experiment management practices, specifically designed for the Titanic competition. The repository uses a hierarchical configuration system and structured experiment tracking.

### Key Architectural Components

- **Experiment Management**: Each experiment gets its own directory (`experiments/exp0001/`) with complete isolation and reproducibility
- **Configuration Hierarchy**: Base configs in `configs/` directory, experiment-specific snapshots in each experiment folder
- **Data Pipeline**: DVC-managed data processing pipeline with versioned artifacts
- **Model Training**: LightGBM with categorical feature support and deterministic training
- **Cross-Validation**: Fixed CV splits stored as parquet files for consistent evaluation
- **Experiment Tracking**: W&B integration with artifact management

### File Synchronization (Important)

The training notebooks exist in two formats:
- `experiments/exp0001/training.py` - Python script format
- `experiments/exp0001/training.ipynb` - Jupyter notebook format

These files are managed with Jupytext but **do not auto-sync**. To synchronize changes:
```bash
jupytext --sync training.ipynb
```

## Common Commands

### Code Quality and Formatting
```bash
# Format and lint code
black .
ruff . --fix

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Data Pipeline (DVC)
```bash
# Run complete data pipeline
dvc repro

# Download/restore data
dvc pull

# Add new data to DVC tracking
dvc add data/external/new_data.csv
dvc push
```

### Kaggle Data Management
```bash
# Download competition data
cd kaggle-projects/titanic
kaggle competitions download -c titanic -p data/raw --unzip

# Or use DVC pipeline
python -m scripts.download_data --competition titanic --output data/raw
```

### Experiment Execution

#### Google Colab Setup
```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/LIGHTBGM-TEM.git
%cd LIGHTBGM-TEM/kaggle-projects/titanic/experiments/exp0001

# 2. Install dependencies
!pip install -r env/requirements.lock

# 3. Set API keys via Colab Secrets
from google.colab import userdata
import os
os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

#### Local Development
```bash
# Navigate to experiment directory
cd kaggle-projects/titanic/experiments/exp0001

# Run training pipeline
python training.py
# or use Jupyter notebook: training.ipynb
```

### Data Processing
```bash
# Preprocess raw data
python -m scripts.preprocess --config configs/data.yaml --input data/raw --output data/processed

# Create CV folds
python -m scripts.make_folds --config configs/cv.yaml --data data/processed/train_processed.parquet --output cv_folds.parquet
```

## Configuration System

### Base Configurations (`configs/`)
- `data.yaml` - Data schema and preprocessing settings
- `cv.yaml` - Cross-validation strategies
- `lgbm.yaml` - LightGBM model parameters
- `features.yaml` - Feature engineering specifications

### Experiment Snapshots
Each experiment directory contains `config.yaml` with frozen settings for complete reproducibility.

## Model Training Specifics

### LightGBM Configuration
- Uses deterministic training with `deterministic: true` and `force_row_wise: true`
- Categorical features handled natively (Sex, Embarked, Title, AgeBand, FareBand)
- Early stopping with 200 rounds patience
- Device type configurable (cpu/gpu/cuda)

### Cross-Validation Strategy
- Stratified K-Fold (default: 5 folds, seed=42)
- CV splits stored as `cv_folds.parquet` with split_id for identification
- Out-of-fold predictions saved for analysis

## Experiment Artifacts

Each experiment produces standardized outputs:
- `oof.parquet` - Out-of-fold predictions (index, fold, y_true, y_pred)
- `metrics.json` - CV metrics (mean, std, per-fold scores)
- `model/fold*.lgb` - Fold-specific LightGBM models
- `submissions/submission.csv` - Kaggle submission file
- `cv_folds.parquet` - Fixed CV splits
- `notes.md` - Experiment documentation
- `wandb_run.txt` - W&B run URL and ID
- `git_sha.txt` - Git commit hash for version tracking

## Feature Engineering

### Implemented Features
- Title extraction from names (Mr/Mrs/Miss/Master/Rare)
- Family size and isolation indicators
- Age and fare binning
- Missing value imputation strategies

### Categorical Feature Handling
All categorical features are converted to pandas category dtype and passed to LightGBM's native categorical support.

## Development Workflow

1. **Experiment Setup**: Copy/modify base experiment directory
2. **Configuration**: Update `config.yaml` with experiment parameters
3. **Training**: Run training notebook/script with W&B tracking
4. **Evaluation**: Analyze OOF predictions for CV quality
5. **Inference**: Generate submission and update experiment log
6. **Documentation**: Update `notes.md` with findings

## Key Dependencies

Core requirements managed in `requirements.txt`:
- LightGBM 4.0+ with categorical support
- pandas 2.0+ for modern data handling
- DVC 3.0+ for data versioning
- W&B 0.15+ for experiment tracking
- Jupytext 1.15+ for notebook-script sync

## Quality Assurance

Pre-commit hooks enforce:
- Black code formatting
- Ruff linting with auto-fix
- Notebook output stripping (nbstripout)
- YAML/JSON validation
- Import sorting with isort