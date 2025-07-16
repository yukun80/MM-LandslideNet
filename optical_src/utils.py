"""
Utility functions for MM-LandslideNet Optical Baseline

This module contains common utility functions used across the optical baseline implementation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
import torch
import numpy as np
import random


def setup_logging(
    log_level: str = "INFO", log_file: Optional[Union[str, Path]] = None, format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the optical baseline package.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging is used.
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create logger
    logger = logging.getLogger("optical_baseline")
    logger.setLevel(numeric_level)

    # Clear any existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logging setup complete. Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Returns:
        torch.device: Best available device (CUDA if available, else CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger("optical_baseline")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger = logging.getLogger("optical_baseline")
        logger.info("CUDA not available, using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count the total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculate the size of a model in MB.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
