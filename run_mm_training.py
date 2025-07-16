#!/usr/bin/env python3
"""
Convenience script to run MM-InternImage-TNF training.

Usage:
    python run_mm_training.py
    python run_mm_training.py --resume /path/to/checkpoint.pth
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run training
from mm_intern_image_src.train import main

if __name__ == "__main__":
    main()
