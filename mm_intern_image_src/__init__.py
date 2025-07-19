"""
MM-TNF: Multi-Modal TNF Model for Landslide Detection

This package implements a state-of-the-art dual-branch model with TNF fusion
for landslide detection using Sentinel-1 SAR and Sentinel-2 optical satellite imagery.

Key Components:
- config: Global configuration management
- dataset: Multi-modal data loading and preprocessing
- models: MM-TNF architecture implementation (dual-branch + TNF fusion)
- utils: Loss functions, metrics, and utilities
- train: Training orchestration for TNF model
- predict: Inference and prediction pipeline

Architecture Overview:
    OpticalBranch (InternImage-T) + SARBranch (EfficientNet-B0) → TNF Fusion → Ensemble Output

Key Changes from Previous Version:
- Simplified from 3-branch to 2-branch architecture
- TNF-inspired gate-based fusion mechanism
- Ensemble prediction from optical + sar + fusion branches
- Optimized data flow for 64×64 images

Example Usage:
    from mm_intern_image_src import create_model, run_training
    from mm_intern_image_src.config import config

    # Create TNF model
    model = create_model()

    # Train model
    run_training()

    # Run prediction
    from mm_intern_image_src.predict import run_prediction
    run_prediction("path/to/checkpoint.pth", "outputs/predictions")
"""

__version__ = "2.0.0"
__author__ = "MM-LandslideNet Team"
__description__ = "Multi-Modal TNF Model for Landslide Detection"

# Import key components for easy access
from .config import config

# Import model components
try:
    from .models import create_tnf_model, MMTNFModel, OpticalBranch, SARBranch, TNFFusionModule

    # Alias for backward compatibility
    create_model = create_tnf_model
    create_optical_dominated_model = create_tnf_model
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    create_model = None
    create_tnf_model = None

# Import dataset components
try:
    from .dataset import (
        MultiModalLandslideDataset,
        create_datasets,
        get_augmentations,
        get_tta_augmentations,
        load_exclude_ids,
    )
except ImportError as e:
    print(f"Warning: Could not import dataset: {e}")

# Import training components
try:
    from .train import run_training, setup_training
except ImportError as e:
    print(f"Warning: Could not import training: {e}")
    run_training = None

# Import prediction components
try:
    from .predict import TNFPredictor, run_prediction
except ImportError as e:
    print(f"Warning: Could not import prediction: {e}")

# Import utility components
try:
    from .utils import (
        # Loss functions
        CombinedLoss,
        FocalLoss,
        DiceLoss,
        # Metrics and evaluation
        calculate_metrics,
        format_metrics,
        get_class_weights,
        # Model management
        save_checkpoint,
        load_checkpoint,
        # Training utilities
        EarlyStopping,
        seed_everything,
    )
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")

# Define what gets imported with "from mm_intern_image_src import *"
__all__ = [
    # Configuration
    "config",
    # Model components
    "create_model",
    "create_tnf_model",
    "MMTNFModel",
    "OpticalBranch",
    "SARBranch",
    "TNFFusionModule",
    # Dataset components
    "MultiModalLandslideDataset",
    "create_datasets",
    "get_augmentations",
    "get_tta_augmentations",
    "load_exclude_ids",
    # Training
    "run_training",
    "setup_training",
    # Prediction
    "TNFPredictor",
    "run_prediction",
    # Loss functions
    "CombinedLoss",
    "FocalLoss",
    "DiceLoss",
    # Metrics and utilities
    "calculate_metrics",
    "format_metrics",
    "get_class_weights",
    "save_checkpoint",
    "load_checkpoint",
    "EarlyStopping",
    "seed_everything",
]


# Version info
def get_version_info():
    """Get detailed version information."""
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "Not installed"

    try:
        import timm

        timm_version = timm.__version__
    except ImportError:
        timm_version = "Not installed"

    return {
        "mm_tnf_version": __version__,
        "torch_version": torch_version,
        "timm_version": timm_version,
        "description": __description__,
    }


# Configuration validation
def validate_installation():
    """Validate that all required components are properly installed."""
    issues = []

    # Check required imports
    required_components = [
        ("models", create_tnf_model),
        ("training", run_training),
        ("dataset", MultiModalLandslideDataset),
    ]

    for name, component in required_components:
        if component is None:
            issues.append(f"Failed to import {name} components")

    # Check configuration
    try:
        config.validate_config()
    except Exception as e:
        issues.append(f"Configuration validation failed: {e}")

    if issues:
        print("⚠️ Installation validation found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Installation validation passed")
        return True


# Print version info when imported
def print_version():
    """Print version information."""
    version_info = get_version_info()
    print(f"MM-TNF {version_info['mm_tnf_version']} - {version_info['description']}")
    print(f"PyTorch: {version_info['torch_version']}, TIMM: {version_info['timm_version']}")


# Backward compatibility aliases
# These ensure existing code continues to work
MMInternImageTNF = MMTNFModel if "MMTNFModel" in globals() else None
TNFFusionBlock = TNFFusionModule if "TNFFusionModule" in globals() else None

# Add backward compatibility to __all__
__all__.extend(["MMInternImageTNF", "TNFFusionBlock", "create_optical_dominated_model"])  # Legacy alias

# Optional: Auto-validate on import (can be disabled)
import os

if os.getenv("MM_TNF_SKIP_VALIDATION", "0") != "1":
    try:
        validate_installation()
    except Exception as e:
        print(f"⚠️ Auto-validation failed: {e}")
