#!/usr/bin/env python3
"""
Clean Dataset Statistics Calculator (RGB-filtered)
Calculate channel statistics (mean/std) and class balance on cleaned dataset
after excluding low-quality images based on RGB optical channel assessment.
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


class RGBFilteredDatasetAnalyzer:
    """
    Calculate statistics for dataset cleaned using RGB-based quality filtering
    """

    def __init__(self, config):
        """Initialize with project configuration"""
        self.config = config

        # Create output directory
        self.config.create_dirs()

        # Load training labels
        self.train_df = pd.read_csv(self.config.TRAIN_CSV)
        print(f"Loaded training metadata: {len(self.train_df)} samples")

        # Define channel groups for organized statistics
        self.channel_groups = {
            "optical": {
                "channels": [0, 1, 2, 3],  # Red, Green, Blue, NIR
                "name": "Sentinel-2 Optical",
                "description": "Red, Green, Blue, Near-Infrared",
            },
            "sar_descending": {
                "channels": [4, 5],  # VV, VH descending
                "name": "SAR Descending",
                "description": "VV, VH descending pass",
            },
            "sar_desc_diff": {
                "channels": [6, 7],  # Diff VV, Diff VH descending
                "name": "SAR Descending Diff",
                "description": "Differential VV, VH descending",
            },
            "sar_ascending": {
                "channels": [8, 9],  # VV, VH ascending
                "name": "SAR Ascending",
                "description": "VV, VH ascending pass",
            },
            "sar_asc_diff": {
                "channels": [10, 11],  # Diff VV, Diff VH ascending
                "name": "SAR Ascending Diff",
                "description": "Differential VV, VH ascending",
            },
        }

        print(f"📊 Channel groups defined:")
        for group_name, group_info in self.channel_groups.items():
            print(f"   {group_info['name']}: channels {group_info['channels']}")

    def load_rgb_exclusion_list(self):
        """
        Load RGB-based exclusion list from JSON file
        """
        exclusion_file = self.config.DATASET_ROOT / "data_check" / "exclude_ids.json"

        if not exclusion_file.exists():
            raise FileNotFoundError(f"RGB exclusion list not found: {exclusion_file}")

        with open(exclusion_file, "r") as f:
            exclusion_data = json.load(f)

        excluded_ids = set(exclusion_data["excluded_image_ids"])
        print(f"📋 Loaded RGB-based exclusion list: {len(excluded_ids)} images to exclude")
        print(f"   RGB threshold: {exclusion_data['threshold']:.4f}")
        print(f"   Threshold metric: {exclusion_data.get('threshold_metric', 'rgb_std_mean')}")
        print(f"   Exclusion method: {exclusion_data.get('threshold_method', 'RGB-based')}")
        print(f"   Exclusion percentage: {exclusion_data['excluded_percentage']:.1f}%")

        return excluded_ids, exclusion_data

    def get_clean_dataset_info(self, excluded_ids):
        """
        Get information about the RGB-filtered clean dataset
        """
        # Filter out excluded images
        clean_mask = ~self.train_df["ID"].isin(excluded_ids)
        clean_df = self.train_df[clean_mask].copy()

        print(f"\n🧹 RGB-filtered clean dataset information:")
        print(f"   Original dataset: {len(self.train_df)} images")
        print(f"   RGB-excluded images: {len(excluded_ids)} images")
        print(f"   Clean dataset: {len(clean_df)} images")
        print(f"   Retention rate: {len(clean_df)/len(self.train_df)*100:.1f}%")

        # Class distribution in clean dataset
        clean_class_counts = pd.Series(clean_df["label"]).value_counts().sort_index()
        print(f"\n   Clean dataset class distribution:")
        print(f"   Class 0 (Non-landslide): {clean_class_counts[0]} ({clean_class_counts[0]/len(clean_df)*100:.1f}%)")
        print(f"   Class 1 (Landslide): {clean_class_counts[1]} ({clean_class_counts[1]/len(clean_df)*100:.1f}%)")

        # Compare with original distribution
        original_class_counts = pd.Series(self.train_df["label"]).value_counts().sort_index()
        print(f"\n   Original vs RGB-Clean class retention:")
        for class_label in [0, 1]:
            original_count = original_class_counts[class_label]
            clean_count = clean_class_counts[class_label]
            retention_rate = clean_count / original_count * 100
            print(f"   Class {class_label}: {clean_count}/{original_count} ({retention_rate:.1f}% retained)")

        return clean_df

    def calculate_channel_statistics_by_group(self, clean_df, excluded_ids):
        """
        Calculate statistics for each channel group separately
        """
        print(f"\n📊 Calculating channel statistics by group on RGB-filtered dataset...")
        print(f"Processing {len(clean_df)} clean images...")

        # Initialize data collection by channel group
        group_data = {}
        for group_name, group_info in self.channel_groups.items():
            group_data[group_name] = {channel: [] for channel in group_info["channels"]}

        processed_count = 0
        failed_loads = []

        for idx, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Processing RGB-clean images"):

            image_id = row["ID"]

            # Skip if in exclusion list (double check)
            if image_id in excluded_ids:
                continue

            image_path = self.config.TRAIN_DATA_DIR / f"{image_id}.npy"

            try:
                # Load image data
                if not image_path.exists():
                    print(f"Warning: Image file not found: {image_path}")
                    failed_loads.append(image_id)
                    continue

                image_data = np.load(image_path)

                # Validate shape
                if image_data.shape != (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS):
                    print(f"Warning: Unexpected shape for {image_id}: {image_data.shape}")
                    failed_loads.append(image_id)
                    continue

                # Collect data for each channel group
                for group_name, group_info in self.channel_groups.items():
                    for channel in group_info["channels"]:
                        channel_data = image_data[:, :, channel].flatten()
                        group_data[group_name][channel].extend(channel_data)

                processed_count += 1

            except Exception as e:
                print(f"Error processing {image_id}: {str(e)}")
                failed_loads.append(image_id)
                continue

        print(f"✅ Successfully processed {processed_count} RGB-clean images")
        if failed_loads:
            print(f"❌ Failed to load {len(failed_loads)} images")

        return group_data, processed_count

    def compute_group_statistics(self, group_data):
        """
        Compute statistics for each channel group
        """
        print("📈 Computing channel group statistics...")

        group_stats = {}

        for group_name, group_info in self.channel_groups.items():
            print(f"\n🔍 Processing {group_info['name']} channels...")

            group_stats[group_name] = {
                "name": group_info["name"],
                "description": group_info["description"],
                "channels": {},
            }

            for channel in group_info["channels"]:
                if len(group_data[group_name][channel]) > 0:
                    channel_values = np.array(group_data[group_name][channel])

                    # Calculate comprehensive statistics
                    mean_val = float(np.mean(channel_values))
                    std_val = float(np.std(channel_values))
                    min_val = float(np.min(channel_values))
                    max_val = float(np.max(channel_values))
                    median_val = float(np.median(channel_values))
                    q25_val = float(np.percentile(channel_values, 25))
                    q75_val = float(np.percentile(channel_values, 75))

                    group_stats[group_name]["channels"][f"channel_{channel}"] = {
                        "channel_index": channel,
                        "name": self.config.CHANNEL_DESCRIPTIONS[channel],
                        "mean": mean_val,
                        "std": std_val,
                        "min": min_val,
                        "max": max_val,
                        "median": median_val,
                        "q25": q25_val,
                        "q75": q75_val,
                        "pixel_count": len(channel_values),
                    }

                    print(
                        f"   Channel {channel:2d} ({self.config.CHANNEL_DESCRIPTIONS[channel]:25s}): "
                        f"mean={mean_val:8.3f}, std={std_val:8.3f}"
                    )
                else:
                    print(f"Warning: No data for channel {channel}")

        return group_stats

    def save_final_statistics(self, group_stats, clean_df, excluded_ids, exclusion_data, processed_count):
        """
        Save comprehensive statistics for RGB-filtered dataset
        """
        print(f"\n💾 Saving final RGB-filtered statistics...")

        # Prepare comprehensive statistics
        final_stats = {
            "dataset_info": {
                "original_image_count": len(self.train_df),
                "excluded_image_count": len(excluded_ids),
                "clean_image_count": len(clean_df),
                "processed_image_count": processed_count,
                "retention_rate": len(clean_df) / len(self.train_df),
                "exclusion_threshold": exclusion_data["threshold"],
                "exclusion_method": exclusion_data.get("threshold_method", "rgb_optical_channels_only_5th_percentile"),
                "exclusion_metric": exclusion_data.get("threshold_metric", "rgb_std_mean"),
                "filtering_approach": "RGB_optical_channels_only",
            },
            "class_distribution": {
                "original": {
                    "class_0": int(self.train_df[self.train_df["label"] == 0].shape[0]),
                    "class_1": int(self.train_df[self.train_df["label"] == 1].shape[0]),
                },
                "clean": {
                    "class_0": int(clean_df[clean_df["label"] == 0].shape[0]),
                    "class_1": int(clean_df[clean_df["label"] == 1].shape[0]),
                },
            },
            "channel_statistics_by_group": group_stats,
            "data_specifications": {
                "image_height": self.config.IMG_HEIGHT,
                "image_width": self.config.IMG_WIDTH,
                "num_channels": self.config.IMG_CHANNELS,
                "channel_descriptions": self.config.CHANNEL_DESCRIPTIONS,
                "channel_groups": {
                    name: {"name": info["name"], "description": info["description"], "channels": info["channels"]}
                    for name, info in self.channel_groups.items()
                },
            },
            "quality_assessment_info": {
                "rgb_quality_filtering": True,
                "rgb_channels_used": [0, 1, 2],  # Red, Green, Blue
                "sar_channels_excluded_from_quality": [4, 5, 6, 7, 8, 9, 10, 11],
                "quality_metrics": ["rgb_std_mean", "rgb_contrast", "rgb_brightness"],
                "threshold_percentile": exclusion_data.get("statistics", {}).get("threshold_percentile", 5.0),
            },
            "processing_info": {
                "script_version": "2.0_RGB_filtered",
                "processing_date": pd.Timestamp.now().isoformat(),
                "config_file": "configs/config.py",
            },
        }

        # Calculate class balance metrics
        clean_class_counts = pd.Series(clean_df["label"]).value_counts().sort_index()
        class_balance = {
            "class_0_count": int(clean_class_counts[0]),
            "class_1_count": int(clean_class_counts[1]),
            "class_0_percentage": float(clean_class_counts[0] / len(clean_df)),
            "class_1_percentage": float(clean_class_counts[1] / len(clean_df)),
            "imbalance_ratio": float(clean_class_counts[0] / clean_class_counts[1]),
            "minority_class": 1 if clean_class_counts[1] < clean_class_counts[0] else 0,
        }

        final_stats["class_balance"] = class_balance

        # Save to JSON
        output_file = self.config.DATASET_ROOT / "data_check" / "channel_stats.json"
        with open(output_file, "w") as f:
            json.dump(final_stats, f, indent=2)

        print(f"✅ Final RGB-filtered statistics saved to: {output_file}")

        # Print comprehensive summary
        print(f"\n📋 Final RGB-Filtered Dataset Summary:")
        print(f"   📸 Filtering method: RGB optical channels only (channels 0-2)")
        print(f"   🧹 Clean images: {len(clean_df)} (retention: {len(clean_df)/len(self.train_df)*100:.1f}%)")
        print(
            f"   ⚖️  Class balance: {class_balance['class_0_count']} : {class_balance['class_1_count']} "
            f"(ratio: {class_balance['imbalance_ratio']:.2f})"
        )
        print(f"   📊 Channel groups processed: {len(group_stats)}")
        print(
            f"   🎯 Total pixels processed: {processed_count * self.config.IMG_HEIGHT * self.config.IMG_WIDTH * self.config.IMG_CHANNELS:,}"
        )

        print(f"\n📈 Channel Group Statistics Summary:")
        for group_name, group_info in group_stats.items():
            channel_count = len(group_info["channels"])
            print(f"   {group_info['name']:20s}: {channel_count} channels processed")

        return output_file


def main():
    """Main execution function"""
    print("🚀 Phase 1 - Step 3: RGB-Filtered Clean Dataset Statistics")
    print("=" * 70)
    print("📸 Using RGB-based quality assessment for data filtering")
    print("🚫 SAR channels excluded from quality assessment (but included in statistics)")
    print("=" * 70)

    # Initialize configuration
    config = Config()

    # Check if train data directory exists
    if not config.TRAIN_DATA_DIR.exists():
        print(f"❌ Training data directory not found: {config.TRAIN_DATA_DIR}")
        print("Please ensure the training data (.npy files) are available.")
        return False

    # Initialize analyzer
    analyzer = RGBFilteredDatasetAnalyzer(config)

    try:
        # Load RGB-based exclusion list
        excluded_ids, exclusion_data = analyzer.load_rgb_exclusion_list()

        # Get clean dataset information
        clean_df = analyzer.get_clean_dataset_info(excluded_ids)

        # Calculate channel statistics by group
        group_data, processed_count = analyzer.calculate_channel_statistics_by_group(clean_df, excluded_ids)

        # Compute group statistics
        group_stats = analyzer.compute_group_statistics(group_data)

        if not group_stats:
            print("❌ No channel statistics were calculated!")
            return False

        # Save final statistics
        output_file = analyzer.save_final_statistics(
            group_stats, clean_df, excluded_ids, exclusion_data, processed_count
        )

        print("\n🎉 Step 3 completed successfully!")
        print(f"📊 Definitive RGB-filtered channel statistics: {output_file}")
        print("🎯 Phase 1 RGB-based data cleaning completed!")
        print("\n✨ Key improvements over previous approach:")
        print("   • RGB-only quality assessment (more reliable)")
        print("   • Separate statistics by channel type")
        print("   • Preserved SAR data while filtering on optical quality")

        return True

    except Exception as e:
        print(f"❌ Error in RGB-filtered statistics calculation: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
