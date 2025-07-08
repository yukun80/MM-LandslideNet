"""
Launch script for MM-LandslideNet Optical Baseline Training

This script can be run directly from the project root directory.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the optical baseline training
if __name__ == "__main__":
    print("=" * 60)
    print("MM-LandslideNet Optical Baseline Training Launcher")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print("Starting training...")
    print("=" * 60)

    try:
        from optical_src.train import main

        main()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user.")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Training failed with error: {e}")
        print("=" * 60)
        raise
