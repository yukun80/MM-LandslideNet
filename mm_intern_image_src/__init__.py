"""
MM-InternImage-TNF: Multi-Modal InternImage with TNF Fusion for Landslide Detection

This package implements a state-of-the-art multi-modal deep learning model for landslide
detection using Sentinel-1 SAR and Sentinel-2 optical satellite imagery. The model combines
three InternImage-T backbones with a TNF-style fusion mechanism.

Key Components:
- config: Global configuration management
- dataset: Multi-modal data loading and preprocessing
- models: MM-InternImage-TNF architecture implementation
- utils: Loss functions, metrics, and utilities
- train: Training and evaluation orchestration
- predict: Standalone inference functionality

Example Usage:
    from mm_intern_image_src import create_model, run_training
    from mm_intern_image_src.config import config

    # Create model
    model = create_model()

    # Train model
    run_training()
"""

__version__ = "1.0.0"
__author__ = "MM-LandslideNet Team"
__description__ = "Multi-Modal InternImage with TNF Fusion for Landslide Detection"

# Import key components for easy access
from .config import config
from .models import create_optical_dominated_model as create_model
from .dataset import MultiModalLandslideDataset, create_datasets
from .train import run_training
from .utils import (
    CombinedLoss,
    FocalLoss,
    DiceLoss,
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
)

# Define what gets imported with "from mm_intern_image_src import *"
__all__ = [
    # Configuration
    "config",
    # Models
    "create_model",
    "MMInternImageTNF",
    "TNFFusionBlock",
    # Dataset
    "MultiModalLandslideDataset",
    "create_datasets",
    # Training
    "run_training",
    # Utils - Loss Functions
    "CombinedLoss",
    "FocalLoss",
    "DiceLoss",
    # Utils - Metrics and Checkpoints
    "calculate_metrics",
    "save_checkpoint",
    "load_checkpoint",
    "seed_everything",
]
