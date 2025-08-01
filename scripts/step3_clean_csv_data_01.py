"""
数据清洗脚本 - step3_clean_csv_data_01.py
基于quality assessment结果，直接从CSV文件中移除低质量数据

遵循三个原则：
1. 最小改动原则：生成清洁的CSV文件，简化后续训练流程
2. 单一职责原则：专注于数据清洗，不涉及其他功能
3. 渐进增强原则：基于现有quality assessment流程的输出进行清洗

使用方法：
    python scripts/step3_clean_csv_data_01.py \
        --exclude_json dataset/data_check/exclude_ids.json \
        --input_csv dataset/train.csv \
        --output_csv dataset/train_cleaned.csv
        
    # 或使用默认路径
    python scripts/step3_clean_csv_data_01.py
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from configs.config import Config

    HAVE_CONFIG = True
except ImportError:
    HAVE_CONFIG = False
    print("⚠️  Warning: configs.config not found, using manual paths")


class CSVDataCleaner:
    """
    CSV数据清洗器

    职责：
    - 读取质量评估产生的排除列表
    - 从CSV文件中移除对应的行
    - 生成清洁的CSV文件并记录清洗信息
    """

    def __init__(self):
        """初始化清洗器"""
        self.cleaning_stats = {
            "original_count": 0,
            "excluded_count": 0,
            "cleaned_count": 0,
            "excluded_by_class": {},
            "cleaned_by_class": {},
            "cleaning_timestamp": datetime.now().isoformat(),
        }

    def load_exclude_ids(self, exclude_json_path: Path) -> Tuple[List[str], Dict[str, Any]]:
        """
        加载需要排除的样本ID列表

        Args:
            exclude_json_path: 排除列表JSON文件路径

        Returns:
            Tuple[List[str], Dict]: (排除的ID列表, 排除信息元数据)
        """
        if not exclude_json_path.exists():
            raise FileNotFoundError(f"排除列表文件不存在: {exclude_json_path}")

        print(f"📖 正在加载排除列表: {exclude_json_path}")

        with open(exclude_json_path, "r", encoding="utf-8") as f:
            exclude_data = json.load(f)

        exclude_ids = exclude_data.get("excluded_image_ids", [])

        if not exclude_ids:
            print("⚠️  警告: 排除列表为空")
            return [], exclude_data

        print(f"📋 加载了 {len(exclude_ids)} 个需要排除的样本ID")

        # 打印排除信息
        if "threshold" in exclude_data:
            print(f"   🎯 排除阈值: {exclude_data['threshold']:.4f}")
        if "threshold_method" in exclude_data:
            print(f"   📏 排除方法: {exclude_data['threshold_method']}")
        if "excluded_percentage" in exclude_data:
            print(f"   📊 排除比例: {exclude_data['excluded_percentage']:.1f}%")

        return exclude_ids, exclude_data

    def load_csv_data(self, csv_path: Path) -> pd.DataFrame:
        """
        加载CSV数据文件

        Args:
            csv_path: CSV文件路径

        Returns:
            pd.DataFrame: 加载的数据
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV数据文件不存在: {csv_path}")

        print(f"📊 正在加载CSV数据: {csv_path}")

        df = pd.read_csv(csv_path)

        print(f"   📈 原始数据: {len(df)} 行")

        # 检查必要的列
        required_columns = ["ID"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")

        # 检查是否有label列（用于统计）
        has_labels = "label" in df.columns
        if has_labels:
            class_counts = df["label"].value_counts().sort_index()
            print(f"   📋 类别分布: {dict(class_counts)}")
        else:
            print("   ℹ️  未找到label列，跳过类别统计")

        self.cleaning_stats["original_count"] = len(df)
        if has_labels:
            self.cleaning_stats["original_by_class"] = df["label"].value_counts().to_dict()

        return df

    def clean_data(self, df: pd.DataFrame, exclude_ids: List[str]) -> pd.DataFrame:
        """
        清洗数据，移除排除列表中的样本

        Args:
            df: 原始数据DataFrame
            exclude_ids: 需要排除的样本ID列表

        Returns:
            pd.DataFrame: 清洗后的数据
        """
        print(f"\n🧹 开始数据清洗...")

        if not exclude_ids:
            print("   ℹ️  排除列表为空，返回原始数据")
            return df.copy()

        # 检查有多少需要排除的ID实际存在于数据中
        exclude_set = set(exclude_ids)
        existing_ids = set(df["ID"].values)
        actual_exclude_ids = exclude_set.intersection(existing_ids)
        missing_exclude_ids = exclude_set - existing_ids

        print(f"   📋 排除列表中的ID: {len(exclude_ids)}")
        print(f"   ✅ 实际存在的ID: {len(actual_exclude_ids)}")
        if missing_exclude_ids:
            print(f"   ⚠️  不存在的ID: {len(missing_exclude_ids)}")
            if len(missing_exclude_ids) <= 5:  # 只显示前5个
                print(f"      {list(missing_exclude_ids)[:5]}")

        # 执行过滤
        mask = ~df["ID"].isin(exclude_ids)
        cleaned_df = df[mask].copy().reset_index(drop=True)

        # 统计清洗结果
        excluded_count = len(df) - len(cleaned_df)
        self.cleaning_stats["excluded_count"] = excluded_count
        self.cleaning_stats["cleaned_count"] = len(cleaned_df)

        print(f"   📤 移除样本: {excluded_count}")
        print(f"   📥 保留样本: {len(cleaned_df)}")
        print(f"   📊 保留率: {len(cleaned_df)/len(df)*100:.1f}%")

        # 如果有标签，统计类别信息
        if "label" in df.columns:
            excluded_df = df[~mask]
            excluded_class_counts = excluded_df["label"].value_counts().to_dict()
            cleaned_class_counts = cleaned_df["label"].value_counts().to_dict()

            self.cleaning_stats["excluded_by_class"] = excluded_class_counts
            self.cleaning_stats["cleaned_by_class"] = cleaned_class_counts

            print(f"   📋 移除的类别分布: {excluded_class_counts}")
            print(f"   📋 保留的类别分布: {cleaned_class_counts}")

        return cleaned_df

    def save_cleaned_data(self, cleaned_df: pd.DataFrame, output_path: Path, exclude_data: Dict[str, Any]) -> Path:
        """
        保存清洗后的数据和清洗报告

        Args:
            cleaned_df: 清洗后的数据
            output_path: 输出文件路径
            exclude_data: 排除信息元数据

        Returns:
            Path: 清洗报告文件路径
        """
        print(f"\n💾 保存清洗后的数据...")

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存清洗后的CSV
        cleaned_df.to_csv(output_path, index=False)
        print(f"   ✅ 清洁数据已保存: {output_path}")

        return output_path

    def clean_csv_data(self, exclude_json_path: Path, input_csv_path: Path, output_csv_path: Path) -> Tuple[Path, Path]:
        """
        执行完整的CSV数据清洗流程

        Args:
            exclude_json_path: 排除列表JSON文件路径
            input_csv_path: 输入CSV文件路径
            output_csv_path: 输出CSV文件路径

        Returns:
            Tuple[Path, Path]: (清洁CSV文件路径, 清洗报告路径)
        """
        print("🚀 开始CSV数据清洗流程")
        print("=" * 60)

        # 步骤1: 加载排除列表
        exclude_ids, exclude_data = self.load_exclude_ids(exclude_json_path)

        # 步骤2: 加载CSV数据
        df = self.load_csv_data(input_csv_path)

        # 步骤3: 清洗数据
        cleaned_df = self.clean_data(df, exclude_ids)

        # 步骤4: 保存结果
        self.save_cleaned_data(cleaned_df, output_csv_path, exclude_data)

        print("\n🎉 CSV数据清洗完成!")
        print(
            f"📊 处理摘要: {self.cleaning_stats['original_count']} → {self.cleaning_stats['cleaned_count']} "
            f"(移除了 {self.cleaning_stats['excluded_count']} 个样本)"
        )

        return output_csv_path


def create_default_paths() -> Tuple[Path, Path, Path]:
    """
    创建默认的文件路径（基于项目配置或合理推断）

    Returns:
        Tuple[Path, Path, Path]: (exclude_json, input_csv, output_csv)
    """
    if HAVE_CONFIG:
        # 使用项目配置
        config = Config()
        exclude_json = config.DATASET_ROOT / "data_check" / "exclude_ids.json"
        input_csv = config.TRAIN_CSV
        output_csv = input_csv.parent / f"{input_csv.stem}_cleaned{input_csv.suffix}"
    else:
        # 使用默认路径推断
        project_root = Path(__file__).parent.parent
        exclude_json = project_root / "dataset" / "data_check" / "exclude_ids.json"
        input_csv = project_root / "dataset" / "train.csv"
        output_csv = project_root / "dataset" / "train_cleaned.csv"

    return exclude_json, input_csv, output_csv


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CSV数据清洗工具 - 移除低质量样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认路径
    python scripts/step3_clean_csv_data_01.py
    
    # 指定自定义路径
    python scripts/step3_clean_csv_data_01.py \
        --exclude_json dataset/data_check/exclude_ids.json \
        --input_csv dataset/list/Train.csv \
        --output_csv dataset/Train.csv
        """,
    )

    parser.add_argument("--exclude_json", type=Path, help="排除列表JSON文件路径")
    parser.add_argument("--input_csv", type=Path, help="输入CSV文件路径")
    parser.add_argument("--output_csv", type=Path, help="输出清洁CSV文件路径")
    parser.add_argument("--force", action="store_true", help="强制覆盖输出文件（如果存在）")

    args = parser.parse_args()

    # 确定文件路径
    if args.exclude_json and args.input_csv and args.output_csv:
        exclude_json = args.exclude_json
        input_csv = args.input_csv
        output_csv = args.output_csv
        print("📂 使用用户指定的文件路径")
    else:
        exclude_json, input_csv, output_csv = create_default_paths()
        print("📂 使用默认文件路径")

    print(f"   🔍 排除列表: {exclude_json}")
    print(f"   📊 输入CSV: {input_csv}")
    print(f"   💾 输出CSV: {output_csv}")

    # 检查输出文件是否存在
    if output_csv.exists() and not args.force:
        print(f"\n⚠️  输出文件已存在: {output_csv}")
        print("使用 --force 参数强制覆盖，或指定不同的输出路径")
        return False

    try:
        # 执行清洗
        cleaner = CSVDataCleaner()
        cleaned_csv = cleaner.clean_csv_data(exclude_json, input_csv, output_csv)

        print(f"\n✅ 数据清洗成功完成!")
        print(f"📄 清洁数据: {cleaned_csv}")
        print(f"\n💡 后续步骤:")
        print(f"   1. 更新训练配置中的CSV路径为: {cleaned_csv}")
        print(f"   2. 移除或注释掉 exclude_ids_file 参数")
        print(f"   3. 享受简化的训练流程! 🚀")

        return True

    except Exception as e:
        print(f"\n❌ 数据清洗失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
