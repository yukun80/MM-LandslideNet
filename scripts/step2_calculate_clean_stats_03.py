#!/usr/bin/env python3
"""
清洁数据集统计计算器 (RGB过滤)
在基于RGB光学通道评估排除低质量图像后，
计算清洁数据集的通道统计(均值/标准差)和类别平衡。
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config


class RGBFilteredDatasetAnalyzer:
    """
    使用基于RGB的质量过滤计算清洁数据集的统计信息
    """

    def __init__(self, config):
        """使用项目配置初始化"""
        self.config = config

        # 创建输出目录
        self.config.create_dirs()

        # 加载训练标签
        self.train_df = pd.read_csv(self.config.TRAIN_CSV)
        print(f"已加载训练元数据: {len(self.train_df)} 个样本")

        # 定义通道组以便组织统计
        self.channel_groups = {
            "optical": {
                "channels": [0, 1, 2, 3],  # 红、绿、蓝、近红外
                "name": "Sentinel-2 光学",
                "description": "红、绿、蓝、近红外",
            },
            "sar_descending": {
                "channels": [4, 5],  # VV, VH 下行
                "name": "SAR 下行",
                "description": "VV, VH 下行通道",
            },
            "sar_desc_diff": {
                "channels": [6, 7],  # Diff VV, Diff VH 下行
                "name": "SAR 下行差分",
                "description": "VV, VH 下行差分",
            },
            "sar_ascending": {
                "channels": [8, 9],  # VV, VH 上行
                "name": "SAR 上行",
                "description": "VV, VH 上行通道",
            },
            "sar_asc_diff": {
                "channels": [10, 11],  # Diff VV, Diff VH 上行
                "name": "SAR 上行差分",
                "description": "VV, VH 上行差分",
            },
        }

        print(f"📊 已定义通道组:")
        for group_name, group_info in self.channel_groups.items():
            print(f"   {group_info['name']}: 通道 {group_info['channels']}")

    def load_rgb_exclusion_list(self):
        """
        从JSON文件加载基于RGB的排除列表
        """
        exclusion_file = self.config.DATASET_ROOT / "data_check" / "exclude_ids.json"

        if not exclusion_file.exists():
            raise FileNotFoundError(f"未找到RGB排除列表: {exclusion_file}")

        with open(exclusion_file, "r", encoding="utf-8") as f:
            exclusion_data = json.load(f)

        excluded_ids = set(exclusion_data["excluded_image_ids"])
        print(f"📋 已加载基于RGB的排除列表: {len(excluded_ids)} 张图像需要排除")
        print(f"   RGB阈值: {exclusion_data['threshold']:.4f}")
        print(f"   阈值指标: {exclusion_data.get('threshold_metric', 'rgb_std_mean')}")
        print(f"   排除方法: {exclusion_data.get('threshold_method', '基于RGB')}")
        print(f"   排除百分比: {exclusion_data['excluded_percentage']:.1f}%")

        return excluded_ids, exclusion_data

    def get_clean_dataset_info(self, excluded_ids):
        """
        获取RGB过滤后清洁数据集的信息
        """
        # 过滤排除的图像
        clean_mask = ~self.train_df["ID"].isin(excluded_ids)
        clean_df = self.train_df[clean_mask].copy()

        print(f"\n🧹 RGB过滤后的清洁数据集信息:")
        print(f"   原始数据集: {len(self.train_df)} 张图像")
        print(f"   RGB排除的图像: {len(excluded_ids)} 张图像")
        print(f"   清洁数据集: {len(clean_df)} 张图像")
        print(f"   保留率: {len(clean_df)/len(self.train_df)*100:.1f}%")

        # 清洁数据集中的类别分布
        clean_class_counts = pd.Series(clean_df["label"]).value_counts().sort_index()
        print(f"\n   清洁数据集类别分布:")
        print(f"   类别0 (非滑坡): {clean_class_counts[0]} ({clean_class_counts[0]/len(clean_df)*100:.1f}%)")
        print(f"   类别1 (滑坡): {clean_class_counts[1]} ({clean_class_counts[1]/len(clean_df)*100:.1f}%)")

        # 与原始分布比较
        original_class_counts = pd.Series(self.train_df["label"]).value_counts().sort_index()
        print(f"\n   原始vs RGB清洁类别保留情况:")
        for class_label in [0, 1]:
            original_count = original_class_counts[class_label]
            clean_count = clean_class_counts[class_label]
            retention_rate = clean_count / original_count * 100
            print(f"   类别{class_label}: {clean_count}/{original_count} ({retention_rate:.1f}% 保留)")

        return clean_df

    def calculate_channel_statistics_by_group(self, clean_df, excluded_ids):
        """
        分别计算每个通道组的统计信息
        """
        print(f"\n📊 在RGB过滤数据集上按组计算通道统计...")
        print(f"正在处理 {len(clean_df)} 张清洁图像...")

        # 按通道组初始化数据收集
        group_data = {}
        for group_name, group_info in self.channel_groups.items():
            group_data[group_name] = {channel: [] for channel in group_info["channels"]}

        processed_count = 0
        failed_loads = []

        for idx, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="处理RGB清洁图像"):

            image_id = row["ID"]

            # 如果在排除列表中则跳过(双重检查)
            if image_id in excluded_ids:
                continue

            image_path = self.config.TRAIN_DATA_DIR / f"{image_id}.npy"

            try:
                # 加载图像数据
                if not image_path.exists():
                    print(f"警告: 未找到图像文件: {image_path}")
                    failed_loads.append(image_id)
                    continue

                image_data = np.load(image_path)

                # 验证形状
                if image_data.shape != (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS):
                    print(f"警告: {image_id} 的形状意外: {image_data.shape}")
                    failed_loads.append(image_id)
                    continue

                # 为每个通道组收集数据
                for group_name, group_info in self.channel_groups.items():
                    for channel in group_info["channels"]:
                        channel_data = image_data[:, :, channel].flatten()
                        group_data[group_name][channel].extend(channel_data)

                processed_count += 1

            except Exception as e:
                print(f"处理 {image_id} 时出错: {str(e)}")
                failed_loads.append(image_id)
                continue

        print(f"✅ 成功处理了 {processed_count} 张RGB清洁图像")
        if failed_loads:
            print(f"❌ 失败加载 {len(failed_loads)} 张图像")

        return group_data, processed_count

    def compute_group_statistics(self, group_data):
        """
        计算每个通道组的统计信息
        """
        print("📈 计算通道组统计...")

        group_stats = {}

        for group_name, group_info in self.channel_groups.items():
            print(f"\n🔍 正在处理 {group_info['name']} 通道...")

            group_stats[group_name] = {
                "name": group_info["name"],
                "description": group_info["description"],
                "channels": {},
            }

            for channel in group_info["channels"]:
                if len(group_data[group_name][channel]) > 0:
                    channel_values = np.array(group_data[group_name][channel])

                    # 计算综合统计
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
                        f"   通道 {channel:2d} ({self.config.CHANNEL_DESCRIPTIONS[channel]:25s}): "
                        f"均值={mean_val:8.3f}, 标准差={std_val:8.3f}"
                    )
                else:
                    print(f"警告: 通道 {channel} 没有数据")

        return group_stats

    def save_final_statistics(self, group_stats, clean_df, excluded_ids, exclusion_data, processed_count):
        """
        保存RGB过滤数据集的综合统计信息
        """
        print(f"\n💾 保存最终RGB过滤统计...")

        # 准备综合统计
        final_stats = {
            "dataset_info": {
                "original_image_count": len(self.train_df),
                "excluded_image_count": len(excluded_ids),
                "clean_image_count": len(clean_df),
                "processed_image_count": processed_count,
                "retention_rate": len(clean_df) / len(self.train_df),
                "exclusion_threshold": exclusion_data["threshold"],
                "exclusion_method": exclusion_data.get("threshold_method", "rgb_光学通道_第5百分位数"),
                "exclusion_metric": exclusion_data.get("threshold_metric", "rgb_std_mean"),
                "filtering_approach": "RGB_光学通道_仅限",
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
                "rgb_channels_used": [0, 1, 2],  # 红、绿、蓝
                "sar_channels_excluded_from_quality": [4, 5, 6, 7, 8, 9, 10, 11],
                "quality_metrics": ["rgb_std_mean", "rgb_contrast", "rgb_brightness"],
                "threshold_percentile": exclusion_data.get("statistics", {}).get("threshold_percentile", 5.0),
            },
            "processing_info": {
                "script_version": "2.0_RGB过滤",
                "processing_date": pd.Timestamp.now().isoformat(),
                "config_file": "configs/config.py",
            },
        }

        # 计算类别平衡指标
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

        # 保存到JSON
        output_file = self.config.DATASET_ROOT / "data_check" / "channel_stats.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)

        print(f"✅ 最终RGB过滤统计已保存到: {output_file}")

        # 打印综合摘要
        print(f"\n📋 最终RGB过滤数据集摘要:")
        print(f"   📸 过滤方法: 仅RGB光学通道 (通道0-2)")
        print(f"   🧹 清洁图像: {len(clean_df)} (保留率: {len(clean_df)/len(self.train_df)*100:.1f}%)")
        print(
            f"   ⚖️  类别平衡: {class_balance['class_0_count']} : {class_balance['class_1_count']} "
            f"(比例: {class_balance['imbalance_ratio']:.2f})"
        )
        print(f"   📊 处理的通道组: {len(group_stats)}")
        print(
            f"   🎯 总处理像素: {processed_count * self.config.IMG_HEIGHT * self.config.IMG_WIDTH * self.config.IMG_CHANNELS:,}"
        )

        print(f"\n📈 通道组统计摘要:")
        for group_name, group_info in group_stats.items():
            channel_count = len(group_info["channels"])
            print(f"   {group_info['name']:20s}: {channel_count} 个通道已处理")

        return output_file


def main():
    """主执行函数"""
    print("🚀 阶段1 - 步骤3: RGB过滤的清洁数据集统计")
    print("=" * 70)
    print("📸 使用基于RGB的质量评估进行数据过滤")
    print("🚫 SAR通道已从质量评估中排除(但包含在统计中)")
    print("=" * 70)

    # 初始化配置
    config = Config()

    # 检查训练数据目录是否存在
    if not config.TRAIN_DATA_DIR.exists():
        print(f"❌ 未找到训练数据目录: {config.TRAIN_DATA_DIR}")
        print("请确保训练数据(.npy文件)可用。")
        return False

    # 初始化分析器
    analyzer = RGBFilteredDatasetAnalyzer(config)

    try:
        # 加载基于RGB的排除列表
        excluded_ids, exclusion_data = analyzer.load_rgb_exclusion_list()

        # 获取清洁数据集信息
        clean_df = analyzer.get_clean_dataset_info(excluded_ids)

        # 按组计算通道统计
        group_data, processed_count = analyzer.calculate_channel_statistics_by_group(clean_df, excluded_ids)

        # 计算组统计
        group_stats = analyzer.compute_group_statistics(group_data)

        if not group_stats:
            print("❌ 没有计算通道统计!")
            return False

        # 保存最终统计
        output_file = analyzer.save_final_statistics(
            group_stats, clean_df, excluded_ids, exclusion_data, processed_count
        )

        print("\n🎉 步骤3完成!")
        print(f"📊 确定的RGB过滤通道统计: {output_file}")
        print("🎯 阶段1 基于RGB的数据清理完成!")
        print("\n✨ 相比以前方法的关键改进:")
        print("   • 仅RGB质量评估(更可靠)")
        print("   • 按通道类型分别统计")
        print("   • 在光学质量过滤时保留了SAR数据")

        return True

    except Exception as e:
        print(f"❌ RGB过滤统计计算出错: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
