#!/usr/bin/env python3
"""
数据分析脚本
用于深入分析多模态遥感数据集
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from utils.visualization import plot_data_distribution, visualize_multimodal_sample, setup_plotting_style


def analyze_channel_statistics(config):
    """分析各通道的统计特性"""
    print("=== 通道统计分析 ===")

    # 随机采样分析
    train_df = pd.read_csv(config.TRAIN_CSV)
    sample_ids = train_df["ID"].sample(100).values

    channel_stats = {"mean": [], "std": [], "min": [], "max": [], "p25": [], "p50": [], "p75": []}

    for channel in range(12):
        values = []
        for img_id in sample_ids:
            img_path = config.TRAIN_DATA_DIR / f"{img_id}.npy"
            image = np.load(img_path)
            values.extend(image[:, :, channel].flatten())

        values = np.array(values)

        channel_stats["mean"].append(np.mean(values))
        channel_stats["std"].append(np.std(values))
        channel_stats["min"].append(np.min(values))
        channel_stats["max"].append(np.max(values))
        channel_stats["p25"].append(np.percentile(values, 25))
        channel_stats["p50"].append(np.percentile(values, 50))
        channel_stats["p75"].append(np.percentile(values, 75))

    # 打印统计信息
    for i in range(12):
        print(f"\n通道 {i} ({config.CHANNEL_DESCRIPTIONS[i]}):")
        print(f"  均值: {channel_stats['mean'][i]:.3f}")
        print(f"  标准差: {channel_stats['std'][i]:.3f}")
        print(f"  范围: [{channel_stats['min'][i]:.3f}, {channel_stats['max'][i]:.3f}]")
        print(
            f"  四分位数: [{channel_stats['p25'][i]:.3f}, {channel_stats['p50'][i]:.3f}, {channel_stats['p75'][i]:.3f}]"
        )

    return channel_stats


def analyze_class_distribution(config):
    """分析类别分布"""
    print("\n=== 类别分布分析 ===")

    train_df = pd.read_csv(config.TRAIN_CSV)

    # 基础统计
    label_counts = train_df["label"].value_counts()
    print(f"总样本数: {len(train_df)}")
    print(f"无滑坡样本: {label_counts[0]} ({label_counts[0]/len(train_df)*100:.1f}%)")
    print(f"有滑坡样本: {label_counts[1]} ({label_counts[1]/len(train_df)*100:.1f}%)")
    print(f"类别比例: 1:{label_counts[0]/label_counts[1]:.2f}")

    # 可视化
    plot_data_distribution(train_df, save_path=config.OUTPUT_ROOT / "data_distribution.png")


def visualize_sample_images(config, num_samples=3):
    """可视化样本图像"""
    print(f"\n=== 可视化 {num_samples} 个样本图像 ===")

    train_df = pd.read_csv(config.TRAIN_CSV)

    # 分别从两个类别采样
    landslide_samples = train_df[train_df["label"] == 1]["ID"].sample(num_samples // 2 + 1).values
    no_landslide_samples = train_df[train_df["label"] == 0]["ID"].sample(num_samples // 2).values

    sample_ids = np.concatenate([landslide_samples, no_landslide_samples])

    for i, img_id in enumerate(sample_ids):
        img_path = config.TRAIN_DATA_DIR / f"{img_id}.npy"
        image = np.load(img_path)

        label = train_df[train_df["ID"] == img_id]["label"].iloc[0]
        label_text = "Landslide" if label == 1 else "No Landslide"

        print(f"样本 {i+1}: ID={img_id}, Label={label_text}")

        save_path = config.OUTPUT_ROOT / f"sample_{img_id}_{label_text}.png"
        visualize_multimodal_sample(image, config, save_path=save_path)


def analyze_data_quality(config):
    """分析数据质量"""
    print("\n=== 数据质量分析 ===")

    train_df = pd.read_csv(config.TRAIN_CSV)

    # 检查缺失文件
    missing_files = []
    corrupted_files = []

    print("检查文件完整性...")
    for img_id in train_df["ID"]:
        img_path = config.TRAIN_DATA_DIR / f"{img_id}.npy"

        if not img_path.exists():
            missing_files.append(img_id)
        else:
            try:
                image = np.load(img_path)
                if image.shape != (64, 64, 12):
                    corrupted_files.append(img_id)
            except:
                corrupted_files.append(img_id)

    print(f"缺失文件: {len(missing_files)}")
    print(f"损坏文件: {len(corrupted_files)}")

    if missing_files:
        print("缺失的文件ID:", missing_files[:10])
    if corrupted_files:
        print("损坏的文件ID:", corrupted_files[:10])

    # 检查异常值
    print("\n检查异常值...")
    sample_images = []
    sample_ids = train_df["ID"].sample(50).values

    for img_id in sample_ids:
        img_path = config.TRAIN_DATA_DIR / f"{img_id}.npy"
        image = np.load(img_path)
        sample_images.append(image)

    sample_images = np.array(sample_images)

    # 计算全局统计
    global_min = sample_images.min()
    global_max = sample_images.max()
    global_mean = sample_images.mean()
    global_std = sample_images.std()

    print(f"全局数值范围: [{global_min:.3f}, {global_max:.3f}]")
    print(f"全局均值±标准差: {global_mean:.3f}±{global_std:.3f}")

    # 检查极值
    extreme_low = sample_images < (global_mean - 5 * global_std)
    extreme_high = sample_images > (global_mean + 5 * global_std)

    print(f"极低值像素比例: {extreme_low.mean()*100:.3f}%")
    print(f"极高值像素比例: {extreme_high.mean()*100:.3f}%")


def main():
    """主分析函数"""
    # 初始化配置
    config = Config()
    config.create_dirs()

    # 设置绘图样式
    setup_plotting_style()

    print("=" * 60)
    print("多模态滑坡检测数据集分析")
    print("=" * 60)

    # 基础数据集信息
    print("\n=== 基础信息 ===")
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    print(f"训练集样本数: {len(train_df):,}")
    print(f"测试集样本数: {len(test_df):,}")
    print(f"图像尺寸: {config.IMG_HEIGHT}×{config.IMG_WIDTH}×{config.IMG_CHANNELS}")

    # 执行各项分析
    analyze_class_distribution(config)
    analyze_data_quality(config)
    channel_stats = analyze_channel_statistics(config)
    visualize_sample_images(config, num_samples=4)

    print("\n" + "=" * 60)
    print("数据分析完成！结果已保存到outputs目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
