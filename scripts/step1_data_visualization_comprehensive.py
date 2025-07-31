#!/usr/bin/env python3
"""
Multi-modal Landslide Detection Dataset Visualization Script
Comprehensive visualization analysis for 12-channel remote sensing data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import Optional, Tuple, List

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config


class MultiModalVisualizer:
    """
    Multi-modal landslide detection data visualizer
    Supports comprehensive visualization of optical, SAR, and difference images
    """

    def __init__(self, config):
        """
        Initialize visualizer
        Args:
            config: Project configuration object
        """
        self.config = config

        # Create output directories
        self.output_dir = config.OUTPUT_ROOT / "datavision"
        self.train_output_dir = self.output_dir / "train_data"
        self.test_output_dir = self.output_dir / "test_data"

        # Create directories
        self.train_output_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Load label data
        self.train_labels = self._load_labels()

        print(f"Visualization results will be saved to: {self.output_dir}")

    def _load_labels(self) -> dict:
        """Load training labels"""
        if self.config.TRAIN_CSV.exists():
            df = pd.read_csv(self.config.TRAIN_CSV)
            labels = dict(zip(df["ID"], df["label"]))
            print(f"Loaded {len(labels)} training labels")
            return labels
        else:
            print("Training label file not found")
            return {}

    def _normalize_channel(
        self,
        channel_data: np.ndarray,
        percentile_range: Tuple[float, float] = (2, 98),
        channel_type: str = "optical",
    ) -> np.ndarray:
        """
        Channel data normalization
        Args:
            channel_data: Single channel data
            percentile_range: Percentile range for robust normalization
            channel_type: Data type ('optical', 'sar_intensity', 'sar_diff')
        Returns:
            Normalized data [0, 1]
        """
        # Handle invalid values first
        valid_mask = np.isfinite(channel_data)
        if not np.any(valid_mask):
            return np.zeros_like(channel_data)

        data = channel_data.copy()

        if channel_type == "sar_intensity":
            percentile_range = (1, 99)
            data = np.clip(data, -70, 50)  # 基于实际范围调整
        elif channel_type == "sar_diff":
            # SAR difference data: preserve sign and use symmetric normalization
            percentile_range = (5, 95)  # Use wider range to preserve important changes
            # Handle outliers
            data = np.clip(data, -50, 50)  # Based on your channel stats
        else:
            # Optical data: original approach
            data = np.clip(data, 0, 25000)  # 基于实际数据范围调整裁剪阈值

        # Percentile normalization
        if channel_type == "sar_diff":
            # For difference data, use symmetric normalization
            abs_data = np.abs(data[valid_mask])
            p_high = np.percentile(abs_data, percentile_range[1])
            if p_high > 0:
                normalized = np.clip(data / p_high, -1, 1)
                # Convert to [0,1] range for display
                normalized = (normalized + 1) / 2
            else:
                normalized = np.ones_like(data) * 0.5
        else:
            # Standard percentile normalization for optical and SAR intensity
            p_low, p_high = np.percentile(data[valid_mask], percentile_range)
            if p_high > p_low:
                normalized = np.clip((data - p_low) / (p_high - p_low), 0, 1)
            else:
                normalized = np.zeros_like(data)

        return normalized

    def _create_rgb_image(self, data: np.ndarray, channels: List[int] = [0, 1, 2]) -> np.ndarray:
        """
        Create RGB image
        Args:
            data: Multi-channel data (H, W, C)
            channels: RGB channel index list
        Returns:
            RGB image (H, W, 3)
        """
        rgb_image = np.zeros((data.shape[0], data.shape[1], 3))

        for i, ch in enumerate(channels):
            # 根据通道类型选择合适的标准化方法
            if ch < data.shape[2]:
                rgb_image[:, :, i] = self._normalize_channel(data[:, :, ch])

        return rgb_image

    def _create_false_color_image(self, data: np.ndarray) -> np.ndarray:
        """
        Create false color image (NIR-R-G)
        Args:
            data: Multi-channel data (H, W, C)
        Returns:
            False color image (H, W, 3)
        """
        # NIR-R-G combination (channels 3-0-1)
        return self._create_rgb_image(data, channels=[3, 0, 1])

    def _create_sar_image(self, data: np.ndarray, channel: int) -> np.ndarray:
        """
        Create SAR grayscale image
        Args:
            data: Multi-channel data (H, W, C)
            channel: SAR channel index
        Returns:
            SAR grayscale image (H, W)
        """
        if channel >= data.shape[2]:
            return np.zeros((data.shape[0], data.shape[1]))

        sar_data = data[:, :, channel]

        # 根据通道类型选择合适的标准化方法
        if channel in [4, 5, 8, 9]:  # SAR intensity channels (VV, VH)
            return self._normalize_channel(sar_data, channel_type="sar_intensity")
        elif channel in [6, 7, 10, 11]:  # SAR difference channels
            return self._normalize_channel(sar_data, channel_type="sar_diff")
        else:  # Optical channels (0, 1, 2, 3)
            return self._normalize_channel(sar_data, channel_type="optical")

    def _calculate_ndvi(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI (Normalized Difference Vegetation Index)
        Args:
            data: Multi-channel data (H, W, C)
        Returns:
            NDVI image (H, W)
        """
        if data.shape[2] >= 4:
            # NIR is channel 3, Red is channel 0
            nir = data[:, :, 3].astype(np.float64)
            red = data[:, :, 0].astype(np.float64)

            # Calculate NDVI: (NIR - Red) / (NIR + Red)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            denominator = nir + red + epsilon
            ndvi = (nir - red) / denominator

            # Clip values to valid NDVI range [-1, 1]
            ndvi = np.clip(ndvi, -1, 1)

            return ndvi
        else:
            # Return zeros if insufficient channels
            return np.zeros((data.shape[0], data.shape[1]))

    def visualize_sample(self, image_id: str, data: np.ndarray, is_test: bool = False) -> None:
        """
        Visualize all 12 individual bands plus RGB, False Color, and NDVI composite images (15 total)
        Args:
            image_id: Image ID
            data: Multi-channel data (H, W, 12)
            is_test: Whether it's test data
        """
        try:
            # Create subplot layout (4x4) for 15 visualizations + 1 combined info panel
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))  # Reduced size to prevent memory issues
            fig.suptitle(
                f"Complete Multi-modal Landslide Detection Data Visualization - {image_id}",
                fontsize=16,
                fontweight="bold",
            )

            # Individual Band Visualizations (Channels 0-11)
            band_descriptions = [
                "Red (Sentinel-2)",
                "Green (Sentinel-2)",
                "Blue (Sentinel-2)",
                "Near Infrared (Sentinel-2)",
                "Descending VV (Sentinel-1)",
                "Descending VH (Sentinel-1)",
                "Descending Diff VV",
                "Descending Diff VH",
                "Ascending VV (Sentinel-1)",
                "Ascending VH (Sentinel-1)",
                "Ascending Diff VV",
                "Ascending Diff VH",
            ]

            # Color maps for different band types
            band_cmaps = [
                "Reds",  # Red
                "Greens",  # Green
                "Blues",  # Blue
                "RdYlBu_r",  # NIR
                "gray",  # SAR VV Desc
                "gray",  # SAR VH Desc
                "RdBu_r",  # VV Diff Desc
                "RdBu_r",  # VH Diff Desc
                "gray",  # SAR VV Asc
                "gray",  # SAR VH Asc
                "RdBu_r",  # VV Diff Asc
                "RdBu_r",  # VH Diff Asc
            ]

            # Display all 12 individual bands
            for band in range(12):
                row = band // 4
                col = band % 4

                band_data = self._create_sar_image(data, band)
                im = axes[row, col].imshow(band_data, cmap=band_cmaps[band])
                axes[row, col].set_title(f"Band {band}\n{band_descriptions[band]}", fontweight="bold", fontsize=9)
                axes[row, col].axis("off")
                try:
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
                except Exception as cb_error:
                    print(f"⚠️ Colorbar warning for {image_id} band {band}: {cb_error}")

            # 13. True Color RGB Composite (position [3,0])
            true_color = self._create_rgb_image(data, channels=[0, 1, 2])  # R-G-B
            axes[3, 0].imshow(true_color)
            axes[3, 0].set_title("True Color Composite\n(RGB: Bands 0-1-2)", fontweight="bold", fontsize=9)
            axes[3, 0].axis("off")

            # 14. False Color NIR-R-G Composite (position [3,1])
            false_color = self._create_false_color_image(data)
            axes[3, 1].imshow(false_color)
            axes[3, 1].set_title("False Color Composite\n(NIR-R-G: Bands 3-0-1)", fontweight="bold", fontsize=9)
            axes[3, 1].axis("off")

            # 15. NDVI (Normalized Difference Vegetation Index) (position [3,2])
            ndvi = self._calculate_ndvi(data)
            ndvi_im = axes[3, 2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            axes[3, 2].set_title("NDVI\n(NIR-Red)/(NIR+Red)", fontweight="bold", fontsize=9)
            axes[3, 2].axis("off")
            try:
                plt.colorbar(ndvi_im, ax=axes[3, 2], fraction=0.046, pad=0.04)
            except Exception as cb_error:
                print(f"⚠️ NDVI colorbar warning for {image_id}: {cb_error}")

            # 16. Combined Information Panel (position [3,3])
            stats_text = self._get_data_statistics(data)
            channel_info = (
                "Channel Mapping:\n"
                "Optical: 0:Red, 1:Green, 2:Blue, 3:NIR\n"
                "SAR Desc: 4:VV, 5:VH, 6:VV_Diff, 7:VH_Diff\n"
                "SAR Asc: 8:VV, 9:VH, 10:VV_Diff, 11:VH_Diff"
            )

            combined_text = f"Data Statistics:\n{stats_text}\n\n{channel_info}"

            axes[3, 3].text(
                0.05,
                0.95,
                combined_text,
                transform=axes[3, 3].transAxes,
                fontsize=7,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightsteelblue", alpha=0.8),
            )
            axes[3, 3].set_title("Dataset & Channel Info", fontweight="bold", fontsize=9)
            axes[3, 3].axis("off")

            # Add label information for training data
            if not is_test and image_id in self.train_labels:
                label = self.train_labels[image_id]
                label_text = "Landslide" if label == 1 else "No Landslide"
                label_color = "red" if label == 1 else "green"

                # Add label information at the top
                fig.text(
                    0.5,
                    0.98,
                    f"Label: {label_text}",
                    horizontalalignment="center",
                    fontsize=14,
                    fontweight="bold",
                    color=label_color,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=label_color, alpha=0.2),
                )

            plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Leave space for title and label

            # Save image with better error handling
            if is_test:
                save_path = self.test_output_dir / f"{image_id}.png"
            else:
                save_path = self.train_output_dir / f"{image_id}.png"

            # Use lower DPI to reduce memory usage and file size
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")

            return save_path

        except Exception as e:
            print(f"⚠️ Error in visualize_sample for {image_id}: {str(e)}")
            return None
        finally:
            # Always close the figure to free memory
            plt.close("all")
            # Force garbage collection
            import gc

            gc.collect()

    def _get_data_statistics(self, data: np.ndarray) -> str:
        """
        Get data statistics
        Args:
            data: Multi-channel data
        Returns:
            Formatted statistics string
        """
        stats = []
        stats.append(f"Data Shape: {data.shape}")
        stats.append(f"Data Type: {data.dtype}")
        stats.append(f"Value Range: [{data.min():.2f}, {data.max():.2f}]")
        stats.append(f"Mean: {data.mean():.2f}")
        stats.append(f"Std Dev: {data.std():.2f}")

        return "\n".join(stats)

    def visualize_dataset(self, data_type: str = "train", max_samples: Optional[int] = None) -> None:
        """
        Batch visualize dataset
        Args:
            data_type: Data type ('train' or 'test')
            max_samples: Maximum number of samples, None means process all samples
        """
        if data_type == "train":
            csv_file = self.config.TRAIN_CSV
            data_dir = self.config.TRAIN_DATA_DIR
            is_test = False
        else:
            csv_file = self.config.TEST_CSV
            data_dir = self.config.TEST_DATA_DIR
            is_test = True

        # Read sample list
        if not csv_file.exists():
            print(f"CSV file does not exist: {csv_file}")
            return

        df = pd.read_csv(csv_file)
        sample_ids = df["ID"].tolist()

        if max_samples:
            sample_ids = sample_ids[:max_samples]

        print(f"Processing {data_type} dataset: {len(sample_ids)} samples")

        success_count = 0
        error_count = 0

        # Process each sample
        for image_id in tqdm(sample_ids, desc=f"Visualizing {data_type} data"):
            try:
                # Load .npy file
                npy_path = data_dir / f"{image_id}.npy"

                if not npy_path.exists():
                    print(f"File does not exist: {npy_path}")
                    error_count += 1
                    continue

                # Load data
                data = np.load(npy_path)

                # Ensure data format is correct
                if data.shape != (64, 64, 12):
                    print(f"Abnormal data shape: {image_id}, shape={data.shape}")
                    error_count += 1
                    continue

                # Visualize
                save_path = self.visualize_sample(image_id, data, is_test)
                success_count += 1

            except Exception as e:
                print(f"Processing failed {image_id}: {str(e)}")
                error_count += 1
                continue

        print(f"✅ Processing completed!")
        print(f"Success: {success_count} samples")
        print(f"Failed: {error_count} samples")
        print(f"Save path: {self.output_dir}")

    def create_dataset_summary(self) -> None:
        """Create dataset summary statistics"""
        print("Creating dataset summary...")

        # Analyze training and test sets
        train_stats = self._analyze_dataset("train")
        test_stats = self._analyze_dataset("test")

        # Create summary charts
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Multi-modal Landslide Detection Dataset Summary", fontsize=16, fontweight="bold")

        # Training set label distribution
        if "label_distribution" in train_stats:
            labels = list(train_stats["label_distribution"].keys())
            counts = list(train_stats["label_distribution"].values())
            colors = ["green", "red"]

            axes[0, 0].pie(
                counts,
                labels=[f"{'No Landslide' if l==0 else 'Landslide'}\n({c} samples)" for l, c in zip(labels, counts)],
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            axes[0, 0].set_title("Training Set Label Distribution", fontweight="bold")

        # Channel statistics
        if "channel_stats" in train_stats:
            channel_names = [desc.split("(")[0].strip() for desc in self.config.CHANNEL_DESCRIPTIONS]
            means = [stats["mean"] for stats in train_stats["channel_stats"]]
            stds = [stats["std"] for stats in train_stats["channel_stats"]]

            x = np.arange(len(channel_names))
            width = 0.35

            axes[0, 1].bar(x - width / 2, means, width, label="Mean", alpha=0.8)
            axes[0, 1].bar(x + width / 2, stds, width, label="Std Dev", alpha=0.8)
            axes[0, 1].set_xlabel("Channel")
            axes[0, 1].set_ylabel("Value")
            axes[0, 1].set_title("Channel Statistics (Training Set)", fontweight="bold")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(channel_names, rotation=45, ha="right")
            axes[0, 1].legend()

        # Data range distribution
        if "channel_stats" in train_stats:
            mins = [stats["min"] for stats in train_stats["channel_stats"]]
            maxs = [stats["max"] for stats in train_stats["channel_stats"]]

            axes[0, 2].scatter(range(12), mins, label="Min Value", alpha=0.7, s=50)
            axes[0, 2].scatter(range(12), maxs, label="Max Value", alpha=0.7, s=50)
            axes[0, 2].set_xlabel("Channel Index")
            axes[0, 2].set_ylabel("Value Range")
            axes[0, 2].set_title("Channel Value Ranges", fontweight="bold")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # Dataset basic information
        axes[1, 0].text(
            0.1,
            0.5,
            f"Training Samples: {train_stats.get('total_samples', 'N/A')}\n"
            f"Test Samples: {test_stats.get('total_samples', 'N/A')}\n"
            f"Image Size: 64×64×12\n"
            f"Data Type: float64",
            transform=axes[1, 0].transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )
        axes[1, 0].set_title("Dataset Basic Information", fontweight="bold")
        axes[1, 0].axis("off")

        # Modality distribution comparison
        modality_names = ["Optical (Sentinel-2)", "SAR Descending (Sentinel-1)", "SAR Ascending (Sentinel-1)"]
        modality_channels = [4, 4, 4]  # Number of channels per modality

        axes[1, 1].pie(modality_channels, labels=modality_names, autopct="%1.1f%%", startangle=90)
        axes[1, 1].set_title("Multi-modal Channel Distribution", fontweight="bold")

        # Data quality metrics
        if "quality_metrics" in train_stats:
            quality_data = train_stats["quality_metrics"]
            metrics = list(quality_data.keys())
            values = list(quality_data.values())

            axes[1, 2].bar(metrics, values, color="lightcoral", alpha=0.8)
            axes[1, 2].set_title("Data Quality Metrics", fontweight="bold")
            axes[1, 2].set_ylabel("Percentage (%)")
            for i, v in enumerate(values):
                axes[1, 2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.3, hspace=0.3)

        # Save summary
        summary_path = self.output_dir / "dataset_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")  # Reduced DPI to avoid size issues
        plt.close()

        print(f"Dataset summary saved: {summary_path}")

    def _analyze_dataset(self, data_type: str) -> dict:
        """Analyze dataset statistics"""
        if data_type == "train":
            csv_file = self.config.TRAIN_CSV
            data_dir = self.config.TRAIN_DATA_DIR
        else:
            csv_file = self.config.TEST_CSV
            data_dir = self.config.TEST_DATA_DIR

        stats = {}

        if not csv_file.exists():
            return stats

        df = pd.read_csv(csv_file)
        stats["total_samples"] = len(df)

        # Label distribution (training set only)
        if data_type == "train" and "label" in df.columns:
            stats["label_distribution"] = df["label"].value_counts().to_dict()

        # Random sampling for channel statistics (avoid memory issues)
        sample_size = min(100, len(df))
        sample_ids = df["ID"].sample(sample_size).tolist()

        all_data = []
        valid_samples = 0

        for image_id in sample_ids:
            npy_path = data_dir / f"{image_id}.npy"
            if npy_path.exists():
                try:
                    data = np.load(npy_path)
                    if data.shape == (64, 64, 12):
                        all_data.append(data)
                        valid_samples += 1
                except:
                    continue

        if all_data:
            combined_data = np.stack(all_data, axis=0)  # (N, H, W, C)

            # Channel statistics
            channel_stats = []
            for c in range(12):
                channel_data = combined_data[:, :, :, c]
                channel_stats.append(
                    {
                        "mean": float(np.mean(channel_data)),
                        "std": float(np.std(channel_data)),
                        "min": float(np.min(channel_data)),
                        "max": float(np.max(channel_data)),
                    }
                )

            stats["channel_stats"] = channel_stats

            # Data quality metrics
            total_pixels = combined_data.size
            nan_pixels = np.isnan(combined_data).sum()
            inf_pixels = np.isinf(combined_data).sum()
            zero_pixels = (combined_data == 0).sum()

            stats["quality_metrics"] = {
                "NaN Values": (nan_pixels / total_pixels) * 100,
                "Inf Values": (inf_pixels / total_pixels) * 100,
                "Zero Values": (zero_pixels / total_pixels) * 100,
            }

        return stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-modal Landslide Detection Dataset Visualization")

    parser.add_argument("--data-type", choices=["train", "test", "both"], default="both", help="Data type to visualize")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples (for quick testing)")
    parser.add_argument("--create-summary", action="store_true", help="Create dataset summary statistics")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Specific sample IDs to visualize")

    args = parser.parse_args()

    # Initialize configuration
    config = Config()

    # Create visualizer
    visualizer = MultiModalVisualizer(config)

    print("Multi-modal Landslide Detection Dataset Visualization Tool")
    print("=" * 60)

    # Process specific samples
    if args.sample_ids:
        print(f"Visualizing specific samples: {args.sample_ids}")
        for sample_id in args.sample_ids:
            # First try to find in training set
            train_path = config.TRAIN_DATA_DIR / f"{sample_id}.npy"
            if train_path.exists():
                data = np.load(train_path)
                visualizer.visualize_sample(sample_id, data, is_test=False)
                print(f"Processed training sample: {sample_id}")
            else:
                # Find in test set
                test_path = config.TEST_DATA_DIR / f"{sample_id}.npy"
                if test_path.exists():
                    data = np.load(test_path)
                    visualizer.visualize_sample(sample_id, data, is_test=True)
                    print(f"Processed test sample: {sample_id}")
                else:
                    print(f"Sample not found: {sample_id}")
        return

    # Batch processing
    if args.data_type in ["train", "both"]:
        print("Processing training set...")
        visualizer.visualize_dataset("train", args.max_samples)

    if args.data_type in ["test", "both"]:
        print("Processing test set...")
        visualizer.visualize_dataset("test", args.max_samples)

    # Create summary
    if args.create_summary:
        visualizer.create_dataset_summary()

    print("\nData visualization completed!")
    print(f"📁 Results saved in: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
