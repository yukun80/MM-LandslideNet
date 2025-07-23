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
    print("4. Visualize specific sample")
    print("5. Complete 12-band visualization (first 5 training samples)")
    print("6. Exit")

    while True:
        choice = input("\nPlease select demo option (1-6): ").strip()

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
            sample_id = input("Please enter sample ID (e.g., ID_HUD1ST): ").strip()
            if sample_id:
                # Try to find in training set first
                train_path = config.TRAIN_DATA_DIR / f"{sample_id}.npy"
                if train_path.exists():
                    import numpy as np

                    data = np.load(train_path)
                    visualizer.visualize_sample(sample_id, data, is_test=False)
                    print(f"✅ Processed training sample: {sample_id}")
                else:
                    # Find in test set
                    test_path = config.TEST_DATA_DIR / f"{sample_id}.npy"
                    if test_path.exists():
                        import numpy as np

                        data = np.load(test_path)
                        visualizer.visualize_sample(sample_id, data, is_test=True)
                        print(f"✅ Processed test sample: {sample_id}")
                    else:
                        print(f"❌ Sample not found: {sample_id}")

        elif choice == "5":
            print("\n🌈 Creating complete 12-band visualization for first 5 training samples...")
            print("This will show all individual bands plus composite images with correct English labels.")

            # Get first 5 training samples
            import pandas as pd
            import numpy as np

            train_df = pd.read_csv(config.TRAIN_CSV)
            sample_ids = train_df["ID"].head(5).tolist()

            for i, sample_id in enumerate(sample_ids, 1):
                try:
                    print(f"Processing sample {i}/5: {sample_id}")
                    train_path = config.TRAIN_DATA_DIR / f"{sample_id}.npy"
                    if train_path.exists():
                        data = np.load(train_path)
                        save_path = visualizer.visualize_sample(sample_id, data, is_test=False)
                        print(f"✅ Saved: {save_path}")
                    else:
                        print(f"❌ File not found: {train_path}")
                except Exception as e:
                    print(f"❌ Error processing {sample_id}: {str(e)}")

            print("🎉 Complete 12-band visualization finished!")

        elif choice == "6":
            print("👋 Exiting demo")
            break

        else:
            print("❌ Invalid selection, please enter 1-6")

    print(f"\n📁 All visualization results saved in: {visualizer.output_dir}")


if __name__ == "__main__":
    demo_visualization()
