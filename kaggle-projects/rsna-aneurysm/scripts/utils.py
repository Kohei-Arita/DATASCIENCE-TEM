"""
RSNA Aneurysm Detection - Utility Functions

General utility functions for experiment management, data handling,
and common operations in the RSNA aneurysm detection project.
"""

import os
import json
import yaml
import pickle
import random
import logging
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """

    log_level = getattr(logging, level.upper())

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup handlers
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=handlers)

    logger = logging.getLogger(__name__)
    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.safe_dump(config, f, indent=2, sort_keys=False)


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file"""

    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def save_json(data: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save data to JSON file"""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pickle(data: Any, save_path: Union[str, Path]) -> None:
    """Save data to pickle file"""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(pickle_path: Union[str, Path]) -> Any:
    """Load data from pickle file"""

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    return data


def get_git_hash() -> str:
    """
    Get current git commit hash

    Returns:
        Git commit hash (short form)
    """

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch() -> str:
    """
    Get current git branch name

    Returns:
        Git branch name
    """

    try:
        result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_experiment_directory(
    base_dir: Union[str, Path], experiment_id: str, template_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create new experiment directory with standard structure

    Args:
        base_dir: Base experiments directory
        experiment_id: Experiment ID (e.g., 'exp0001')
        template_dir: Optional template directory to copy from

    Returns:
        Path to created experiment directory
    """

    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_id

    # Create main directory
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    subdirs = ["model", "submissions", "env", "logs"]
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(exist_ok=True)

    # Copy template files if provided
    if template_dir:
        template_dir = Path(template_dir)
        if template_dir.exists():
            import shutil

            template_files = ["training.ipynb", "evaluation.ipynb", "inference.ipynb", "env/requirements.lock"]

            for file_path in template_files:
                src = template_dir / file_path
                dst = exp_dir / file_path

                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Union[str, Path],
    is_best: bool = False,
) -> None:
    """
    Save model checkpoint with optimizer and scheduler states

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Performance metrics
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best model
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load checkpoint to

    Returns:
        Checkpoint information
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "git_hash": checkpoint.get("git_hash", "unknown"),
        "timestamp": checkpoint.get("timestamp", "unknown"),
    }


def calculate_model_size(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Calculate model size and parameter counts

    Args:
        model: PyTorch model

    Returns:
        Model size information
    """

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024**2)

    return {
        "total_params": param_count,
        "trainable_params": trainable_params,
        "non_trainable_params": param_count - trainable_params,
        "model_size_mb": model_size_mb,
    }


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def create_submission_file(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    submission_path: Union[str, Path],
    id_column: str = "image_id",
    target_column: str = "aneurysm",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create Kaggle submission file

    Args:
        predictions: Model predictions (probabilities)
        test_df: Test dataframe with image IDs
        submission_path: Path to save submission file
        id_column: Column name for image IDs
        target_column: Column name for predictions
        threshold: Optional threshold for binary predictions

    Returns:
        Submission dataframe
    """

    submission_df = pd.DataFrame(
        {
            id_column: test_df[id_column],
            target_column: predictions if threshold is None else (predictions >= threshold).astype(int),
        }
    )

    # Save submission file
    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved: {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

    return submission_df


def update_experiments_log(log_path: Union[str, Path], experiment_data: Dict[str, Any]) -> None:
    """
    Update experiments log CSV file

    Args:
        log_path: Path to experiments log CSV
        experiment_data: Experiment results to append
    """

    log_path = Path(log_path)

    # Load existing log or create new one
    if log_path.exists():
        experiments_df = pd.read_csv(log_path)
    else:
        experiments_df = pd.DataFrame()

    # Add new experiment data
    new_row = pd.DataFrame([experiment_data])
    experiments_df = pd.concat([experiments_df, new_row], ignore_index=True)

    # Save updated log
    experiments_df.to_csv(log_path, index=False)
    print(f"Experiments log updated: {log_path}")


class EarlyStopping:
    """
    Early stopping utility class

    Monitors validation metric and stops training when no improvement
    is observed for a specified number of epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max", restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            restore_best_weights: Whether to restore best weights when stopped
        """

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.epochs_without_improvement = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should be stopped

        Args:
            score: Current validation score
            model: Model to save best weights from

        Returns:
            True if training should be stopped
        """

        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_weights = model.state_dict().copy()
            return False

        # Check for improvement
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.epochs_without_improvement = 0
            if model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.epochs_without_improvement += 1

        # Check if should stop
        if self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = self.epochs_without_improvement
            return True

        return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore best weights to model"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress of training"""

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_system_info():
    """Print system and environment information"""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    # Python version
    import sys

    print(f"Python version: {sys.version}")

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # CUDA information
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available")

    # Memory information
    try:
        import psutil

        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("psutil not available for memory info")

    print("=" * 50)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test seeding
    set_seed(42)

    # Test git functions
    git_hash = get_git_hash()
    git_branch = get_git_branch()
    print(f"Git hash: {git_hash}")
    print(f"Git branch: {git_branch}")

    # Test time formatting
    test_time = 3725.5  # 1 hour, 2 minutes, 5.5 seconds
    formatted_time = format_time(test_time)
    print(f"Formatted time: {formatted_time}")

    # Print system info
    print_system_info()

    print("All tests passed!")
