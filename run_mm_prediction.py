#!/usr/bin/env python3
"""
Convenience script to run Asymmetric Dual-Backbone Fusion Model inference.

Usage:
python run_mm_prediction.py \
    --model_path outputs/mm_intern_image_tnf/AsymmetricFusionModel_best.pth \
    --test_data_dir dataset/test_data \
    --output_csv outputs/submissions/submission_InternImage.csv \
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run inference
from mm_intern_image_src.predict import main

if __name__ == "__main__":
    results = main()
