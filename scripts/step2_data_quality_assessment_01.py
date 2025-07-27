"""
数据质量评估脚本 (基于RGB)
仅使用Sentinel-2光学数据的RGB通道(0-2)计算质量指标
识别低信息含量样本，避免SAR噪声干扰。
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


class RGBQualityAssessment:
    """
    基于RGB的数据质量评估，仅使用光学通道(0-2)
    避免SAR噪声对质量评估的干扰
    """

    def __init__(self, config):
        """使用项目配置初始化"""
        self.config = config

        # 创建输出目录
        self.config.create_dirs()

        # 加载训练标签
        self.train_df = pd.read_csv(self.config.TRAIN_CSV)
        print(f"已加载训练元数据: {len(self.train_df)} 个样本")

        # RGB通道索引 (Sentinel-2 光学数据)
        self.rgb_channels = [0, 1, 2]  # 红、绿、蓝
        print("📸 使用RGB通道进行质量评估:")
        for idx in self.rgb_channels:
            print(f"   通道 {idx}: {self.config.CHANNEL_DESCRIPTIONS[idx]}")

    def calculate_rgb_quality_score(self, image_data):
        """
        仅使用RGB通道计算质量分数
        参数:
            image_data: 形状为 (H, W, C) 的numpy数组
        返回:
            dict: RGB质量指标
        """
        if len(image_data.shape) != 3:
            print(f"警告: 意外的图像形状 {image_data.shape}")
            return {
                "rgb_std_red": 0.0,
                "rgb_std_green": 0.0,
                "rgb_std_blue": 0.0,
                "rgb_std_mean": 0.0,
                "rgb_contrast": 0.0,
                "rgb_brightness": 0.0,
            }

        # 提取RGB通道
        rgb_data = image_data[:, :, self.rgb_channels]  # 形状: (H, W, 3)

        # 计算各通道统计
        red_channel = rgb_data[:, :, 0]
        green_channel = rgb_data[:, :, 1]
        blue_channel = rgb_data[:, :, 2]

        # 每个RGB通道的标准差
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        # RGB标准差的平均值
        rgb_std_mean = np.mean([red_std, green_std, blue_std])

        # 额外的质量指标
        # 对比度: 灰度图像的标准差
        grayscale = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
        rgb_contrast = np.std(grayscale)

        # 亮度: 灰度图像的平均值
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
        处理所有训练图像并计算RGB质量分数
        """
        print("🔍 开始基于RGB的数据质量评估...")
        print(f"正在处理 {len(self.train_df)} 个训练样本...")

        quality_scores = []
        failed_loads = []

        for idx, row in tqdm(self.train_df.iterrows(), total=len(self.train_df), desc="评估RGB质量"):

            image_id = row["ID"]
            image_path = self.config.TRAIN_DATA_DIR / f"{image_id}.npy"

            try:
                # 加载图像数据
                if not image_path.exists():
                    print(f"警告: 未找到图像文件: {image_path}")
                    failed_loads.append(image_id)
                    continue

                image_data = np.load(image_path)

                # 计算RGB质量分数
                quality_metrics = self.calculate_rgb_quality_score(image_data)

                # 存储结果
                result = {
                    "image_id": image_id,
                    "label": row["label"],
                    "shape": f"{image_data.shape[0]}x{image_data.shape[1]}x{image_data.shape[2]}",
                }
                result.update(quality_metrics)
                quality_scores.append(result)

            except Exception as e:
                print(f"处理 {image_id} 时出错: {str(e)}")
                failed_loads.append(image_id)
                continue

        print(f"✅ 成功处理了 {len(quality_scores)} 张图像")
        if failed_loads:
            print(f"❌ 失败加载 {len(failed_loads)} 张图像")
            print("失败的图像:", failed_loads[:10], "..." if len(failed_loads) > 10 else "")

        return quality_scores

    def save_quality_scores(self, quality_scores):
        """
        将RGB质量分数保存到CSV文件
        """
        output_file = self.config.DATASET_ROOT / "data_check" / "image_quality_scores.csv"

        # 转换为DataFrame
        df = pd.DataFrame(quality_scores)

        # 按RGB平均标准差排序 (降序 - 最高质量在前)
        df = df.sort_values("rgb_std_mean", ascending=False)

        # 保存到CSV
        df.to_csv(output_file, index=False)

        print(f"💾 RGB质量分数已保存到: {output_file}")
        print(f"RGB质量分数统计:")
        print(f"  RGB平均标准差 - 均值: {df['rgb_std_mean'].mean():.4f}")
        print(f"  RGB平均标准差 - 中位数: {df['rgb_std_mean'].median():.4f}")
        print(f"  RGB平均标准差 - 最小值: {df['rgb_std_mean'].min():.4f}")
        print(f"  RGB平均标准差 - 最大值: {df['rgb_std_mean'].max():.4f}")
        print(f"  RGB平均标准差 - 标准差: {df['rgb_std_mean'].std():.4f}")

        print(f"\n  RGB对比度 - 均值: {df['rgb_contrast'].mean():.4f}")
        print(f"  RGB对比度 - 中位数: {df['rgb_contrast'].median():.4f}")

        print(f"\n  各通道统计:")
        print(f"  红色标准差   - 均值: {df['rgb_std_red'].mean():.4f}, 中位数: {df['rgb_std_red'].median():.4f}")
        print(f"  绿色标准差 - 均值: {df['rgb_std_green'].mean():.4f}, 中位数: {df['rgb_std_green'].median():.4f}")
        print(f"  蓝色标准差  - 均值: {df['rgb_std_blue'].mean():.4f}, 中位数: {df['rgb_std_blue'].median():.4f}")

        return output_file, df


def main():
    """主执行函数"""
    print("🚀 阶段1 - 步骤1: 基于RGB的数据质量评估")
    print("=" * 60)
    print("📸 仅使用Sentinel-2光学数据的RGB通道(0-2)")
    print("🚫 排除SAR通道以避免噪声干扰")
    print("=" * 60)

    # 初始化配置
    config = Config()

    # 检查训练数据目录是否存在
    if not config.TRAIN_DATA_DIR.exists():
        print(f"❌ 未找到训练数据目录: {config.TRAIN_DATA_DIR}")
        print("请确保训练数据(.npy文件)可用。")
        return False

    # 初始化评估工具
    assessor = RGBQualityAssessment(config)

    # 处理所有训练图像
    quality_scores = assessor.assess_all_training_images()

    if not quality_scores:
        print("❌ 没有图像被成功处理!")
        return False

    # 保存结果
    output_file, df = assessor.save_quality_scores(quality_scores)

    print("\n🎉 步骤1完成!")
    print(f"📊 RGB质量评估结果已保存到: {output_file}")
    print(f"📈 处理了 {len(df)} 张图像的RGB质量分数")
    print("🎯 准备步骤2: RGB质量分析和阈值确定")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
