#!/usr/bin/env python3
"""
RGB Quality Score Analysis Script
Analyze RGB-based quality score distribution, determine threshold for low-quality data,
and generate exclusion list using only optical channel information.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config


class RGBQualityAnalyzer:
    """
    Analyzer for RGB-based image quality scores and threshold determination
    """

    def __init__(self, config):
        """Initialize with project configuration"""
        self.config = config

        # Create output directory
        self.config.create_dirs()

    def load_quality_scores(self):
        """
        Load RGB quality scores from CSV file
        """
        quality_file = self.config.DATASET_ROOT / "data_check" / "image_quality_scores.csv"

        if not quality_file.exists():
            raise FileNotFoundError(f"Quality scores file not found: {quality_file}")

        df = pd.read_csv(quality_file)
        print(f"📊 Loaded RGB quality scores for {len(df)} images")

        # Verify expected columns exist
        expected_cols = ["rgb_std_mean", "rgb_contrast", "rgb_std_red", "rgb_std_green", "rgb_std_blue"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")

        return df

    def analyze_distribution(self, df):
        """
        Analyze RGB quality score distribution
        """
        print("\n🔍 Analyzing RGB quality score distribution...")

        # RGB Mean Standard Deviation statistics
        print("RGB Mean Standard Deviation Statistics:")
        stats = df["rgb_std_mean"].describe()
        print(stats)

        # Calculate percentiles for threshold analysis
        percentiles = [1, 2, 5, 10, 15, 20, 25]
        print(f"\nRGB Mean Std - Low-end percentiles:")
        for p in percentiles:
            value = np.percentile(df["rgb_std_mean"], p)
            count = len(df[df["rgb_std_mean"] <= value])
            print(f"  {p}th percentile: {value:.4f} ({count} images, {count/len(df)*100:.1f}%)")

        # RGB Contrast statistics
        print(f"\nRGB Contrast Statistics:")
        contrast_stats = df["rgb_contrast"].describe()
        print(f"  Mean: {contrast_stats['mean']:.4f}")
        print(f"  Median: {contrast_stats['50%']:.4f}")
        print(f"  Min: {contrast_stats['min']:.4f}")
        print(f"  Max: {contrast_stats['max']:.4f}")

        # Individual RGB channel statistics
        print(f"\nIndividual RGB Channel Statistics:")
        for channel, col in zip(["Red", "Green", "Blue"], ["rgb_std_red", "rgb_std_green", "rgb_std_blue"]):
            channel_stats = df[col].describe()
            print(f"  {channel:5s} - Mean: {channel_stats['mean']:8.4f}, Median: {channel_stats['50%']:8.4f}")

        # Class distribution analysis
        print(f"\nClass distribution:")
        class_counts = df["label"].value_counts()
        print(f"  Class 0 (Non-landslide): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
        print(f"  Class 1 (Landslide): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")

        return stats

    def visualize_distribution(self, df):
        """
        Create comprehensive visualization of RGB quality score distribution
        """
        print("\n📈 Creating RGB quality score distribution visualizations...")

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create comprehensive figure with 3x3 layout
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle("RGB Quality Score Distribution Analysis", fontsize=16, fontweight="bold")

        # 1. RGB Mean Std overall histogram
        axes[0, 0].hist(df["rgb_std_mean"], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].axvline(
            df["rgb_std_mean"].mean(), color="red", linestyle="--", label=f'Mean: {df["rgb_std_mean"].mean():.4f}'
        )
        axes[0, 0].axvline(
            df["rgb_std_mean"].median(),
            color="orange",
            linestyle="--",
            label=f'Median: {df["rgb_std_mean"].median():.4f}',
        )
        axes[0, 0].set_xlabel("RGB Mean Standard Deviation")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("RGB Mean Std Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. RGB Mean Std by class
        df_plot = df.copy()
        df_plot["Class"] = df_plot["label"].map({0: "Non-landslide", 1: "Landslide"})
        sns.boxplot(data=df_plot, x="Class", y="rgb_std_mean", ax=axes[0, 1])
        axes[0, 1].set_title("RGB Mean Std by Class")
        axes[0, 1].set_ylabel("RGB Mean Standard Deviation")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. RGB Contrast distribution
        axes[0, 2].hist(df["rgb_contrast"], bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
        axes[0, 2].axvline(
            df["rgb_contrast"].mean(), color="red", linestyle="--", label=f'Mean: {df["rgb_contrast"].mean():.4f}'
        )
        axes[0, 2].set_xlabel("RGB Contrast")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].set_title("RGB Contrast Distribution")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Individual RGB channel distributions
        colors = ["red", "green", "blue"]
        channels = ["rgb_std_red", "rgb_std_green", "rgb_std_blue"]
        channel_names = ["Red Channel Std", "Green Channel Std", "Blue Channel Std"]

        for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
            row = 1
            col = i
            axes[row, col].hist(df[channel], bins=40, alpha=0.7, color=color, edgecolor="black")
            axes[row, col].axvline(
                df[channel].mean(), color="darkred", linestyle="--", label=f"Mean: {df[channel].mean():.4f}"
            )
            axes[row, col].set_xlabel(f"{name}")
            axes[row, col].set_ylabel("Frequency")
            axes[row, col].set_title(f"{name} Distribution")
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        # 5. Cumulative distribution of RGB Mean Std
        sorted_scores = np.sort(df["rgb_std_mean"])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[2, 0].plot(sorted_scores, cumulative, linewidth=2, color="purple")
        axes[2, 0].set_xlabel("RGB Mean Standard Deviation")
        axes[2, 0].set_ylabel("Cumulative Probability")
        axes[2, 0].set_title("RGB Mean Std - Cumulative Distribution")
        axes[2, 0].grid(True, alpha=0.3)

        # 6. Low-end focus (bottom 25%)
        bottom_25_threshold = np.percentile(df["rgb_std_mean"], 25)
        low_quality_data = df[df["rgb_std_mean"] <= bottom_25_threshold]
        axes[2, 1].hist(low_quality_data["rgb_std_mean"], bins=30, alpha=0.7, color="coral", edgecolor="black")
        axes[2, 1].set_xlabel("RGB Mean Standard Deviation")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].set_title("Bottom 25% RGB Quality Scores")
        axes[2, 1].grid(True, alpha=0.3)

        # 7. Threshold analysis
        thresholds = np.percentile(df["rgb_std_mean"], [1, 2, 5, 10, 15, 20])
        counts = [len(df[df["rgb_std_mean"] <= t]) for t in thresholds]
        percentages = [1, 2, 5, 10, 15, 20]

        axes[2, 2].bar(range(len(thresholds)), counts, alpha=0.7, color="gold", edgecolor="black")
        axes[2, 2].set_xticks(range(len(thresholds)))
        axes[2, 2].set_xticklabels([f"{p}%\n({t:.3f})" for p, t in zip(percentages, thresholds)])
        axes[2, 2].set_xlabel("RGB Mean Std Percentile Threshold\n(Value)")
        axes[2, 2].set_ylabel("Number of Images")
        axes[2, 2].set_title("Images Below Different RGB Thresholds")
        axes[2, 2].grid(True, alpha=0.3)

        # Add text annotations for threshold analysis
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            axes[2, 2].text(i, count + len(df) * 0.01, f"{count}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()

        # Save the plot
        output_file = self.config.DATASET_ROOT / "data_check" / "quality_score_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 RGB distribution plot saved to: {output_file}")

        plt.show()

        return output_file

    def determine_threshold(self, df):
        """
        Determine optimal threshold for excluding low-quality images based on RGB metrics
        """
        print("\n🎯 Determining optimal RGB threshold for low-quality data exclusion...")

        # Analysis of different threshold options using RGB mean std
        threshold_options = {
            "Conservative (5th percentile)": np.percentile(df["rgb_std_mean"], 5),
            "Moderate (10th percentile)": np.percentile(df["rgb_std_mean"], 10),
            "Aggressive (12th percentile)": np.percentile(df["rgb_std_mean"], 12),
        }

        print("RGB Mean Std threshold options:")
        for name, threshold in threshold_options.items():
            excluded_count = len(df[df["rgb_std_mean"] <= threshold])
            excluded_percentage = excluded_count / len(df) * 100

            # Class distribution of excluded samples
            excluded_df = df[df["rgb_std_mean"] <= threshold]
            if len(excluded_df) > 0:
                class_0_excluded = len(excluded_df[excluded_df["label"] == 0])
                class_1_excluded = len(excluded_df[excluded_df["label"] == 1])

                # Additional statistics for excluded samples
                avg_contrast = excluded_df["rgb_contrast"].mean()
                avg_brightness = excluded_df["rgb_brightness"].mean()

                print(f"\n  {name}:")
                print(f"    RGB Mean Std Threshold: {threshold:.4f}")
                print(f"    Excluded: {excluded_count} images ({excluded_percentage:.1f}%)")
                print(f"    Class 0 excluded: {class_0_excluded} ({class_0_excluded/excluded_count*100:.1f}%)")
                print(f"    Class 1 excluded: {class_1_excluded} ({class_1_excluded/excluded_count*100:.1f}%)")
                print(f"    Avg RGB Contrast: {avg_contrast:.4f}")
                print(f"    Avg RGB Brightness: {avg_brightness:.4f}")

        # Use Aggressive approach (15th percentile) as default
        recommended_threshold = threshold_options["Aggressive (12th percentile)"]

        print(f"\n🎯 Recommended RGB threshold: {recommended_threshold:.4f} (12th percentile)")
        print("   This represents a balanced approach using RGB optical data only.")

        return recommended_threshold

    def generate_exclusion_list(self, df, threshold):
        """
        Generate exclusion list based on RGB threshold
        """
        print(f"\n📝 Generating exclusion list with RGB threshold: {threshold:.4f}")

        # Find images below RGB threshold
        low_quality_mask = df["rgb_std_mean"] <= threshold
        excluded_images = df[low_quality_mask]["image_id"].tolist()

        print(f"Images to exclude: {len(excluded_images)} out of {len(df)} ({len(excluded_images)/len(df)*100:.1f}%)")

        # Class distribution of excluded images
        excluded_df = df[low_quality_mask]
        class_distribution = excluded_df["label"].value_counts()
        print(f"Excluded images by class:")
        for class_label, count in class_distribution.items():
            print(f"  Class {class_label}: {count} images ({count/len(excluded_images)*100:.1f}%)")

        # RGB statistics of excluded images
        if len(excluded_df) > 0:
            print(f"\nRGB statistics of excluded images:")
            print(f"  Avg RGB Mean Std: {excluded_df['rgb_std_mean'].mean():.4f}")
            print(f"  Avg RGB Contrast: {excluded_df['rgb_contrast'].mean():.4f}")
            print(f"  Avg RGB Brightness: {excluded_df['rgb_brightness'].mean():.4f}")
            print(f"  Avg Red Std: {excluded_df['rgb_std_red'].mean():.4f}")
            print(f"  Avg Green Std: {excluded_df['rgb_std_green'].mean():.4f}")
            print(f"  Avg Blue Std: {excluded_df['rgb_std_blue'].mean():.4f}")

        # Save exclusion list to JSON
        exclusion_data = {
            "threshold": threshold,
            "threshold_metric": "rgb_std_mean",
            "threshold_method": "rgb_optical_channels_only_15th_percentile",
            "total_images": len(df),
            "excluded_count": len(excluded_images),
            "excluded_percentage": len(excluded_images) / len(df) * 100,
            "excluded_by_class": {
                "class_0": int(class_distribution.get(0, 0)),
                "class_1": int(class_distribution.get(1, 0)),
            },
            "excluded_image_ids": excluded_images,
            "statistics": {
                "threshold_percentile": 5.0,
                "mean_excluded_rgb_std": float(excluded_df["rgb_std_mean"].mean()) if len(excluded_df) > 0 else 0.0,
                "max_excluded_rgb_std": float(excluded_df["rgb_std_mean"].max()) if len(excluded_df) > 0 else 0.0,
                "mean_excluded_contrast": float(excluded_df["rgb_contrast"].mean()) if len(excluded_df) > 0 else 0.0,
                "mean_excluded_brightness": (
                    float(excluded_df["rgb_brightness"].mean()) if len(excluded_df) > 0 else 0.0
                ),
            },
        }

        output_file = self.config.DATASET_ROOT / "data_check" / "exclude_ids.json"
        with open(output_file, "w") as f:
            json.dump(exclusion_data, f, indent=2)

        print(f"💾 RGB-based exclusion list saved to: {output_file}")

        return output_file, excluded_images


def main():
    """Main execution function"""
    print("🚀 Phase 1 - Step 2: RGB Quality Score Analysis")
    print("=" * 60)
    print("📸 Analyzing RGB-based quality scores from optical channels only")
    print("=" * 60)

    # Initialize configuration
    config = Config()

    # Initialize analyzer
    analyzer = RGBQualityAnalyzer(config)

    try:
        # Load RGB quality scores
        df = analyzer.load_quality_scores()

        # Analyze distribution
        stats = analyzer.analyze_distribution(df)

        # Visualize distribution
        plot_file = analyzer.visualize_distribution(df)

        # Determine threshold
        threshold = analyzer.determine_threshold(df)

        # Generate exclusion list
        exclusion_file, excluded_images = analyzer.generate_exclusion_list(df, threshold)

        print("\n🎉 Step 2 completed successfully!")
        print(f"📊 RGB distribution visualization: {plot_file}")
        print(f"📝 RGB-based exclusion list: {exclusion_file}")
        print(f"🚫 {len(excluded_images)} images marked for exclusion based on RGB quality")
        print("🎯 Ready for Step 3: Clean dataset statistics calculation")

        return True

    except Exception as e:
        print(f"❌ Error in RGB quality analysis: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
