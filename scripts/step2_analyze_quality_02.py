#!/usr/bin/env python3
"""
RGB质量分数分析脚本 - 彻底修复中文字体版本
分析基于RGB的质量分数分布，确定低质量数据阈值，
仅使用光学通道信息生成排除列表。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import matplotlib.font_manager as fm
import warnings

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config

# 全局字体变量
CHINESE_FONT = None


def configure_chinese_matplotlib():
    """配置matplotlib支持中文显示 - 彻底修复版本"""
    global CHINESE_FONT
    print("🎨 配置matplotlib中文字体...")

    # 获取所有可用字体
    all_fonts = [f.name for f in fm.fontManager.ttflist]

    # 按优先级尝试中文字体
    priority_fonts = [
        "SimHei",  # 黑体 - 最可靠
        "Noto Sans SC",  # Google Noto 简体中文
        "Noto Sans CJK SC",  # Google Noto CJK 简体中文
        "Microsoft YaHei",  # 微软雅黑
    ]

    selected_font = None
    for font in priority_fonts:
        if font in all_fonts:
            selected_font = font
            print(f"✅ 找到中文字体: {font}")
            break

    # 强制配置matplotlib - 多重保险
    if selected_font:
        CHINESE_FONT = selected_font

        # 方法1: 全局rcParams设置
        plt.rcParams.update(
            {
                "font.sans-serif": [selected_font, "DejaVu Sans"],
                "axes.unicode_minus": False,
                "font.size": 10,
                "font.family": "sans-serif",
            }
        )

        # 方法2: 强制重建字体管理器
        try:
            fm._rebuild()
        except:
            try:
                # 新版本matplotlib的方法
                fm.fontManager.__init__()
            except:
                pass

        # 方法3: 抑制字体警告（这样即使有回退也不会显示警告）
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

        print(f"🎯 字体配置完成: {selected_font}")
        return selected_font
    else:
        print("⚠️ 未找到中文字体，使用英文显示")
        CHINESE_FONT = "DejaVu Sans"
        plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})
        return "DejaVu Sans"


def apply_font_to_axes(ax):
    """为axes的所有文本元素应用中文字体"""
    if CHINESE_FONT and CHINESE_FONT != "DejaVu Sans":
        # 设置标题字体
        if ax.get_title():
            ax.set_title(ax.get_title(), fontfamily=CHINESE_FONT)

        # 设置轴标签字体
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontfamily=CHINESE_FONT)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontfamily=CHINESE_FONT)

        # 设置刻度标签字体
        for label in ax.get_xticklabels():
            label.set_fontfamily(CHINESE_FONT)
        for label in ax.get_yticklabels():
            label.set_fontfamily(CHINESE_FONT)

        # 设置图例字体
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontfamily(CHINESE_FONT)


class RGBQualityAnalyzer:
    """
    基于RGB的图像质量分数分析器和阈值确定器
    """

    def __init__(self, config):
        """使用项目配置初始化"""
        self.config = config
        # 创建输出目录
        self.config.create_dirs()

    def load_quality_scores(self):
        """从CSV文件加载RGB质量分数"""
        quality_file = self.config.DATASET_ROOT / "data_check" / "image_quality_scores.csv"

        if not quality_file.exists():
            raise FileNotFoundError(f"未找到质量分数文件: {quality_file}")

        df = pd.read_csv(quality_file)
        print(f"📊 已加载 {len(df)} 张图像的RGB质量分数")

        # 验证预期列是否存在
        expected_cols = ["rgb_std_mean", "rgb_contrast", "rgb_std_red", "rgb_std_green", "rgb_std_blue"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少预期列: {missing_cols}")

        return df

    def analyze_distribution(self, df):
        """分析RGB质量分数分布"""
        print("\n🔍 分析RGB质量分数分布...")

        # RGB平均标准差统计
        print("RGB平均标准差统计:")
        stats = df["rgb_std_mean"].describe()
        print(stats)

        # 计算阈值分析的百分位数
        percentiles = [1, 2, 5, 10, 15, 20, 25]
        print(f"\nRGB平均标准差 - 低端百分位数:")
        for p in percentiles:
            value = np.percentile(df["rgb_std_mean"], p)
            count = len(df[df["rgb_std_mean"] <= value])
            print(f"  第{p}百分位数: {value:.4f} ({count} 张图像, {count/len(df)*100:.1f}%)")

        return stats

    def visualize_distribution(self, df):
        """创建RGB质量分数分布的综合可视化"""
        print("\n📈 创建RGB质量分数分布可视化...")

        # 设置绘图样式
        plt.style.use("default")
        sns.set_palette("husl")

        # 创建3x3布局的综合图表
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))

        # 设置主标题，使用指定字体
        if CHINESE_FONT and CHINESE_FONT != "DejaVu Sans":
            fig.suptitle("RGB质量分数分布分析", fontsize=16, fontweight="bold", fontfamily=CHINESE_FONT)
        else:
            fig.suptitle("RGB Quality Score Distribution Analysis", fontsize=16, fontweight="bold")

        # 1. RGB平均标准差整体直方图
        axes[0, 0].hist(df["rgb_std_mean"], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].axvline(
            df["rgb_std_mean"].mean(), color="red", linestyle="--", label=f'均值: {df["rgb_std_mean"].mean():.4f}'
        )
        axes[0, 0].axvline(
            df["rgb_std_mean"].median(),
            color="orange",
            linestyle="--",
            label=f'中位数: {df["rgb_std_mean"].median():.4f}',
        )
        axes[0, 0].set_xlabel("RGB平均标准差")
        axes[0, 0].set_ylabel("频次")
        axes[0, 0].set_title("RGB平均标准差分布")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        apply_font_to_axes(axes[0, 0])

        # 2. 按类别的RGB平均标准差箱线图
        df_plot = df.copy()
        df_plot["类别"] = df_plot["label"].map({0: "非滑坡", 1: "滑坡"})
        sns.boxplot(data=df_plot, x="类别", y="rgb_std_mean", ax=axes[0, 1])
        axes[0, 1].set_title("按类别的RGB平均标准差")
        axes[0, 1].set_ylabel("RGB平均标准差")
        axes[0, 1].set_xlabel("类别")
        axes[0, 1].grid(True, alpha=0.3)
        apply_font_to_axes(axes[0, 1])

        # 3. RGB对比度分布
        axes[0, 2].hist(df["rgb_contrast"], bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
        axes[0, 2].axvline(
            df["rgb_contrast"].mean(), color="red", linestyle="--", label=f'均值: {df["rgb_contrast"].mean():.4f}'
        )
        axes[0, 2].set_xlabel("RGB对比度")
        axes[0, 2].set_ylabel("频次")
        axes[0, 2].set_title("RGB对比度分布")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        apply_font_to_axes(axes[0, 2])

        # 4-6. 各RGB通道分布
        colors = ["red", "green", "blue"]
        channels = ["rgb_std_red", "rgb_std_green", "rgb_std_blue"]
        channel_names = ["红色通道标准差", "绿色通道标准差", "蓝色通道标准差"]

        for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
            row, col = 1, i
            axes[row, col].hist(df[channel], bins=40, alpha=0.7, color=color, edgecolor="black")
            axes[row, col].axvline(
                df[channel].mean(), color="darkred", linestyle="--", label=f"均值: {df[channel].mean():.4f}"
            )
            axes[row, col].set_xlabel(name)
            axes[row, col].set_ylabel("频次")
            axes[row, col].set_title(f"{name}分布")
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            apply_font_to_axes(axes[row, col])

        # 7. RGB平均标准差累积分布
        sorted_scores = np.sort(df["rgb_std_mean"])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[2, 0].plot(sorted_scores, cumulative, linewidth=2, color="purple")
        axes[2, 0].set_xlabel("RGB平均标准差")
        axes[2, 0].set_ylabel("累积概率")
        axes[2, 0].set_title("RGB平均标准差 - 累积分布")
        axes[2, 0].grid(True, alpha=0.3)
        apply_font_to_axes(axes[2, 0])

        # 8. 低端25%聚焦
        bottom_25_threshold = np.percentile(df["rgb_std_mean"], 25)
        low_quality_data = df[df["rgb_std_mean"] <= bottom_25_threshold]
        axes[2, 1].hist(low_quality_data["rgb_std_mean"], bins=30, alpha=0.7, color="coral", edgecolor="black")
        axes[2, 1].set_xlabel("RGB平均标准差")
        axes[2, 1].set_ylabel("频次")
        axes[2, 1].set_title("最低25%的RGB质量分数")
        axes[2, 1].grid(True, alpha=0.3)
        apply_font_to_axes(axes[2, 1])

        # 9. 阈值分析
        thresholds = np.percentile(df["rgb_std_mean"], [1, 2, 5, 10, 15, 20])
        counts = [len(df[df["rgb_std_mean"] <= t]) for t in thresholds]
        percentages = [1, 2, 5, 10, 15, 20]

        bars = axes[2, 2].bar(range(len(thresholds)), counts, alpha=0.7, color="gold", edgecolor="black")
        axes[2, 2].set_xticks(range(len(thresholds)))
        axes[2, 2].set_xticklabels([f"{p}%\n({t:.3f})" for p, t in zip(percentages, thresholds)])
        axes[2, 2].set_xlabel("RGB平均标准差百分位阈值\n(数值)")
        axes[2, 2].set_ylabel("图像数量")
        axes[2, 2].set_title("低于不同RGB阈值的图像")
        axes[2, 2].grid(True, alpha=0.3)
        apply_font_to_axes(axes[2, 2])

        # 为阈值分析添加文本注释
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            text = axes[2, 2].text(
                i,
                count + len(df) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )
            if CHINESE_FONT and CHINESE_FONT != "DejaVu Sans":
                text.set_fontfamily(CHINESE_FONT)

        # 使用上下文管理器来抑制警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()

            # 保存图表
            output_file = self.config.DATASET_ROOT / "data_check" / "quality_score_distribution.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"📊 RGB分布图已保存到: {output_file}")

        plt.show()
        return output_file

    def determine_threshold(self, df):
        """基于RGB指标确定排除低质量图像的最优阈值"""
        print("\n🎯 确定低质量数据排除的最优RGB阈值...")

        # 使用RGB平均标准差分析不同阈值选项
        threshold_options = {
            "保守 (第5百分位数)": np.percentile(df["rgb_std_mean"], 5),
            "适中 (第10百分位数)": np.percentile(df["rgb_std_mean"], 10),
            "激进 (第15百分位数)": np.percentile(df["rgb_std_mean"], 15),
        }

        print("RGB平均标准差阈值选项:")
        for name, threshold in threshold_options.items():
            excluded_count = len(df[df["rgb_std_mean"] <= threshold])
            excluded_percentage = excluded_count / len(df) * 100

            # 排除样本的类别分布
            excluded_df = df[df["rgb_std_mean"] <= threshold]
            if len(excluded_df) > 0:
                class_0_excluded = len(excluded_df[excluded_df["label"] == 0])
                class_1_excluded = len(excluded_df[excluded_df["label"] == 1])

                # 排除样本的额外统计
                avg_contrast = excluded_df["rgb_contrast"].mean()
                avg_brightness = excluded_df.get("rgb_brightness", pd.Series([0])).mean()

                print(f"\n  {name}:")
                print(f"    RGB平均标准差阈值: {threshold:.4f}")
                print(f"    排除: {excluded_count} 张图像 ({excluded_percentage:.1f}%)")
                print(f"    类别0排除: {class_0_excluded} ({class_0_excluded/excluded_count*100:.1f}%)")
                print(f"    类别1排除: {class_1_excluded} ({class_1_excluded/excluded_count*100:.1f}%)")
                print(f"    平均RGB对比度: {avg_contrast:.4f}")
                print(f"    平均RGB亮度: {avg_brightness:.4f}")

        # 使用激进方法 (第15百分位数) 作为默认
        recommended_threshold = threshold_options["激进 (第15百分位数)"]

        print(f"\n🎯 推荐的RGB阈值: {recommended_threshold:.4f} (第15百分位数)")
        print("   这代表了仅使用RGB光学数据的平衡方法。")

        return recommended_threshold

    def generate_exclusion_list(self, df, threshold):
        """基于RGB阈值生成排除列表"""
        print(f"\n📝 使用RGB阈值生成排除列表: {threshold:.4f}")

        # 查找低于RGB阈值的图像
        low_quality_mask = df["rgb_std_mean"] <= threshold
        excluded_images = df[low_quality_mask]["image_id"].tolist()

        print(f"需要排除的图像: {len(excluded_images)} / {len(df)} ({len(excluded_images)/len(df)*100:.1f}%)")

        # 排除图像的类别分布
        excluded_df = df[low_quality_mask]
        class_distribution = excluded_df["label"].value_counts()
        print(f"按类别排除的图像:")
        for class_label, count in class_distribution.items():
            print(f"  类别{class_label}: {count} 张图像 ({count/len(excluded_images)*100:.1f}%)")

        # 保存排除列表到JSON
        exclusion_data = {
            "threshold": threshold,
            "threshold_metric": "rgb_std_mean",
            "threshold_method": "rgb_光学通道仅限_第15百分位数",
            "total_images": len(df),
            "excluded_count": len(excluded_images),
            "excluded_percentage": len(excluded_images) / len(df) * 100,
            "excluded_by_class": {
                "class_0": int(class_distribution.get(0, 0)),
                "class_1": int(class_distribution.get(1, 0)),
            },
            "excluded_image_ids": excluded_images,
        }

        output_file = self.config.DATASET_ROOT / "data_check" / "exclude_ids.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(exclusion_data, f, indent=2, ensure_ascii=False)

        print(f"💾 基于RGB的排除列表已保存到: {output_file}")

        return output_file, excluded_images


def main():
    """主执行函数"""
    print("🚀 阶段1 - 步骤2: RGB质量分数分析")
    print("=" * 60)
    print("📸 分析仅来自光学通道的RGB质量分数")
    print("=" * 60)

    # 在程序开始时配置中文字体
    selected_font = configure_chinese_matplotlib()
    print(f"🎨 当前使用字体: {selected_font}")

    # 初始化配置
    config = Config()

    # 初始化分析器
    analyzer = RGBQualityAnalyzer(config)

    # 加载RGB质量分数
    df = analyzer.load_quality_scores()

    # 分析分布
    stats = analyzer.analyze_distribution(df)

    # 可视化分布
    plot_file = analyzer.visualize_distribution(df)

    # 确定阈值
    threshold = analyzer.determine_threshold(df)

    # 生成排除列表
    exclusion_file, excluded_images = analyzer.generate_exclusion_list(df, threshold)

    print("\n🎉 步骤2完成!")
    print(f"📊 RGB分布可视化: {plot_file}")
    print(f"📝 基于RGB的排除列表: {exclusion_file}")
    print(f"🚫 基于RGB质量标记了 {len(excluded_images)} 张图像进行排除")
    print("🎯 准备步骤3: 清洁数据集统计计算")


if __name__ == "__main__":
    main()
