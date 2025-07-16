#!/usr/bin/env python3
"""
Data Quality Assessment Script (RGB-based)
Calculate quality metrics using only RGB channels (0-2) from Sentinel-2 optical data
to identify low-information content samples, avoiding SAR noise interference.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config


class RGBQualityAssessment:
    """
    RGB-based data quality assessment using only optical channels (0-2)
    Avoids SAR noise interference in quality evaluation
    """

    def __init__(self, config):
        """Initialize with project configuration"""
        self.config = config

        # Create output directory
        self.config.create_dirs()

        # Load training labels
        self.train_df = pd.read_csv(self.config.TRAIN_CSV)
        print(f"Loaded training metadata: {len(self.train_df)} samples")

        # RGB channel indices (Sentinel-2 optical data)
        self.rgb_channels = [0, 1, 2]  # Red, Green, Blue
        print("📸 Using RGB channels for quality assessment:")
        for idx in self.rgb_channels:
            print(f"   Channel {idx}: {self.config.CHANNEL_DESCRIPTIONS[idx]}")

    def calculate_rgb_quality_score(self, image_data):
        """
        Calculate quality score using only RGB channels
        Args:
            image_data: numpy array of shape (H, W, C)
        Returns:
            dict: RGB quality metrics
        """
        if len(image_data.shape) != 3:
            print(f"Warning: Unexpected image shape {image_data.shape}")
            return {
                "rgb_std_red": 0.0,
                "rgb_std_green": 0.0,
                "rgb_std_blue": 0.0,
                "rgb_std_mean": 0.0,
                "rgb_contrast": 0.0,
                "rgb_brightness": 0.0,
            }

        # Extract RGB channels
        rgb_data = image_data[:, :, self.rgb_channels]  # Shape: (H, W, 3)

        # Calculate individual channel statistics
        red_channel = rgb_data[:, :, 0]
        green_channel = rgb_data[:, :, 1]
        blue_channel = rgb_data[:, :, 2]

        # Standard deviation for each RGB channel
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        # Mean of RGB standard deviations
        rgb_std_mean = np.mean([red_std, green_std, blue_std])

        # Additional quality metrics
        # Contrast: standard deviation of the grayscale image
        grayscale = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
        rgb_contrast = np.std(grayscale)

        # Brightness: mean of the grayscale image
        rgb_brightness = np.mean(grayscale)

        return {
            "rgb_std_red": red_std,
            "rgb_std_green": green_std,
            "rgb_std_blue": blue_std,
            "rgb_std_mean": rgb_std_mean,
            "rgb_contrast": rgb_contrast,
            "rgb_brightness": rgb_brightness,
        }

    def assess_all_training_images(self):
        """
        Process all training images and calculate RGB quality scores
        """
        print("🔍 Starting RGB-based data quality assessment...")
        print(f"Processing {len(self.train_df)} training samples...")

        quality_scores = []
        failed_loads = []

        for idx, row in tqdm(self.train_df.iterrows(), total=len(self.train_df), desc="Assessing RGB quality"):

            image_id = row["ID"]
            image_path = self.config.TRAIN_DATA_DIR / f"{image_id}.npy"

            try:
                # Load image data
                if not image_path.exists():
                    print(f"Warning: Image file not found: {image_path}")
                    failed_loads.append(image_id)
                    continue

                image_data = np.load(image_path)

                # Calculate RGB quality score
                quality_metrics = self.calculate_rgb_quality_score(image_data)

                # Store result
                result = {
                    "image_id": image_id,
                    "label": row["label"],
                    "shape": f"{image_data.shape[0]}x{image_data.shape[1]}x{image_data.shape[2]}",
                }
                result.update(quality_metrics)
                quality_scores.append(result)

            except Exception as e:
                print(f"Error processing {image_id}: {str(e)}")
                failed_loads.append(image_id)
                continue

        print(f"✅ Successfully processed {len(quality_scores)} images")
        if failed_loads:
            print(f"❌ Failed to load {len(failed_loads)} images")
            print("Failed images:", failed_loads[:10], "..." if len(failed_loads) > 10 else "")

        return quality_scores

    def save_quality_scores(self, quality_scores):
        """
        Save RGB quality scores to CSV file
        """
        output_file = self.config.DATASET_ROOT / "data_check" / "image_quality_scores.csv"

        # Convert to DataFrame
        df = pd.DataFrame(quality_scores)

        # Sort by RGB mean standard deviation (descending - highest quality first)
        df = df.sort_values("rgb_std_mean", ascending=False)

        # Save to CSV
        df.to_csv(output_file, index=False)

        print(f"💾 RGB quality scores saved to: {output_file}")
        print(f"RGB Quality Score Statistics:")
        print(f"  RGB Mean Std - Mean: {df['rgb_std_mean'].mean():.4f}")
        print(f"  RGB Mean Std - Median: {df['rgb_std_mean'].median():.4f}")
        print(f"  RGB Mean Std - Min: {df['rgb_std_mean'].min():.4f}")
        print(f"  RGB Mean Std - Max: {df['rgb_std_mean'].max():.4f}")
        print(f"  RGB Mean Std - Std: {df['rgb_std_mean'].std():.4f}")

        print(f"\n  RGB Contrast - Mean: {df['rgb_contrast'].mean():.4f}")
        print(f"  RGB Contrast - Median: {df['rgb_contrast'].median():.4f}")

        print(f"\n  Individual Channel Statistics:")
        print(f"  Red Std   - Mean: {df['rgb_std_red'].mean():.4f}, Median: {df['rgb_std_red'].median():.4f}")
        print(f"  Green Std - Mean: {df['rgb_std_green'].mean():.4f}, Median: {df['rgb_std_green'].median():.4f}")
        print(f"  Blue Std  - Mean: {df['rgb_std_blue'].mean():.4f}, Median: {df['rgb_std_blue'].median():.4f}")

        return output_file, df


def main():
    """Main execution function"""
    print("🚀 Phase 1 - Step 1: RGB-Based Data Quality Assessment")
    print("=" * 60)
    print("📸 Using only RGB channels (0-2) from Sentinel-2 optical data")
    print("🚫 Excluding SAR channels to avoid noise interference")
    print("=" * 60)

    # Initialize configuration
    config = Config()

    # Check if train data directory exists
    if not config.TRAIN_DATA_DIR.exists():
        print(f"❌ Training data directory not found: {config.TRAIN_DATA_DIR}")
        print("Please ensure the training data (.npy files) are available.")
        return False

    # Initialize assessment tool
    assessor = RGBQualityAssessment(config)

    # Process all training images
    quality_scores = assessor.assess_all_training_images()

    if not quality_scores:
        print("❌ No images were successfully processed!")
        return False

    # Save results
    output_file, df = assessor.save_quality_scores(quality_scores)

    print("\n🎉 Step 1 completed successfully!")
    print(f"📊 RGB quality assessment results saved to: {output_file}")
    print(f"📈 Processed {len(df)} images with RGB quality scores")
    print("🎯 Ready for Step 2: RGB quality analysis and threshold determination")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
