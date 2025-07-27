#!/usr/bin/env python3
"""
Data Visualization Demo Script
Quick test of multi-modal data visualization functionality
"""

import sys
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from scripts.step1_data_visualization_comprehensive import MultiModalVisualizer


def demo_visualization():
    """Demo data visualization functionality"""
    print("🎨 Multi-modal Landslide Detection Data Visualization Demo")
    print("=" * 50)

    # Initialize configuration
    config = Config()

    # Create visualizer
    visualizer = MultiModalVisualizer(config)

    print("\n📋 Available demo options:")
    print("1. Quick visualization of first 7147 training samples")
    print("2. Quick visualization of first 5397 test samples")
    print("3. Create dataset summary statistics")
    print("4. Exit")

    while True:
        choice = input("\nPlease select demo option (1-4): ").strip()

        if choice == "1":
            print("\n🚂 Visualizing first 7147 training samples...")
            visualizer.visualize_dataset("train", max_samples=7147)

        elif choice == "2":
            print("\n🧪 Visualizing first 5397 test samples...")
            visualizer.visualize_dataset("test", max_samples=5397)

        elif choice == "3":
            print("\n📊 Creating dataset summary statistics...")
            visualizer.create_dataset_summary()

        elif choice == "4":
            print("👋 Exiting demo")
            break

        else:
            print("❌ Invalid selection, please enter 1-4")

    print(f"\n📁 All visualization results saved in: {visualizer.output_dir}")


if __name__ == "__main__":
    demo_visualization()
