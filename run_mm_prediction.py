#!/usr/bin/env python3
"""
Convenience script to run MM-InternImage-TNF inference.

Usage:
    python run_mm_prediction.py --model_path /path/to/model.pth --image_path /path/to/image.npy
    python run_mm_prediction.py --model_path /path/to/model.pth --image_path /path/to/image.npy --verbose
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
    print(f"\nPrediction completed: {results['prediction']} (confidence: {results['confidence']:.4f})")
