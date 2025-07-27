"""
Excluded Images Visualization Manager
移动被排除的低质量图像的可视化结果到专门目录
基于 RGB 质量评估结果自动整理可视化文件
"""

import sys
import json
import shutil
from pathlib import Path
from typing import List, Tuple

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config


class ExcludedVisualizationManager:
    """管理被排除图像的可视化结果"""

    def __init__(self, config):
        """
        初始化管理器
        Args:
            config: 项目配置对象
        """
        self.config = config

        # 定义路径
        self.datavision_root = Path("dataset/datavision")
        self.train_vis_dir = self.datavision_root / "train_data"
        self.excluded_vis_dir = self.datavision_root / "excluded_image"
        self.exclude_ids_file = Path("dataset/data_check/exclude_ids.json")

        # 创建目标目录
        self.excluded_vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"📂 可视化管理器初始化完成")
        print(f"   源目录: {self.train_vis_dir}")
        print(f"   目标目录: {self.excluded_vis_dir}")
        print(f"   排除列表: {self.exclude_ids_file}")

    def load_excluded_ids(self) -> List[str]:
        """
        加载被排除的图像ID列表
        Returns:
            excluded_ids: 被排除的图像ID列表
            metadata: 排除相关的元数据
        """
        if not self.exclude_ids_file.exists():
            raise FileNotFoundError(f"排除列表文件未找到: {self.exclude_ids_file}")

        with open(self.exclude_ids_file, "r", encoding="utf-8") as f:
            exclude_data = json.load(f)

        excluded_ids = exclude_data.get("excluded_image_ids", [])

        print(f"📋 加载排除列表完成:")
        print(f"   排除图像数量: {len(excluded_ids)}")

        return excluded_ids

    def find_visualization_files(self, excluded_ids: List[str]) -> dict:
        """
        查找对应的可视化文件
        Args:
            excluded_ids: 被排除的图像ID列表
        Returns:
            文件映射字典 {image_id: visualization_file_path}
        """
        visualization_files = {}
        missing_files = []

        print(f"\n🔍 查找可视化文件...")

        for image_id in excluded_ids:
            # 可能的文件名格式
            possible_names = [
                f"{image_id}_visualization.png",
                f"{image_id}.png",
                f"{image_id}_improved_visualization.png",
            ]

            found = False
            for filename in possible_names:
                file_path = self.train_vis_dir / filename
                if file_path.exists():
                    visualization_files[image_id] = file_path
                    found = True
                    break

            if not found:
                missing_files.append(image_id)

        print(f"缺失可视化文件: {len(missing_files)} 个")

        return visualization_files

    def move_visualization_files(self, visualization_files: dict) -> dict:
        """
        移动可视化文件到排除目录
        Args:
            visualization_files: 文件映射字典
        Returns:
            移动结果统计
        """
        print(f"\n📦 开始移动可视化文件...")

        success_count = 0

        for image_id, source_path in visualization_files.items():
            # 目标文件路径
            target_path = self.excluded_vis_dir / source_path.name

            # 移动文件
            shutil.move(str(source_path), str(target_path))
            success_count += 1

            if success_count <= 5:  # 只显示前5个移动的文件
                print(f"   ✅ {source_path.name} -> {target_path.name}")
            elif success_count == 6:
                print(f"   ... 继续移动中 ...")

        # 移动结果统计
        result_stats = {
            "total_files": len(visualization_files),
            "successful_moves": success_count,
        }

        print(f"\n📊 移动完成统计:")
        print(f"   总文件数: {result_stats['total_files']}")
        print(f"   成功移动: {result_stats['successful_moves']}")

        return result_stats

    def run_management_process(self):
        """
        执行完整的可视化文件管理流程
        Returns:
            操作是否成功
        """
        print("🚀 开始排除图像可视化管理流程")
        print("=" * 60)

        # 1. 加载排除列表
        excluded_ids = self.load_excluded_ids()

        # 2. 查找可视化文件
        visualization_files = self.find_visualization_files(excluded_ids)
        missing_count = len(excluded_ids) - len(visualization_files)

        if not visualization_files:
            print("❌ 未找到任何可视化文件，操作终止")
            return False

        # 3. 移动文件
        move_stats = self.move_visualization_files(visualization_files)

        print("\n🎉 可视化文件管理完成!")
        print("=" * 60)
        print(f"📂 被排除图像的可视化文件已移动到: {self.excluded_vis_dir}")


def main():
    """主执行函数"""
    print("🎨 排除图像可视化文件管理工具")
    print("基于RGB质量评估结果自动整理低质量图像的可视化文件")
    print("=" * 60)

    # 初始化配置
    config = Config()

    # 创建管理器
    manager = ExcludedVisualizationManager(config)

    # 执行管理流程
    manager.run_management_process()


if __name__ == "__main__":
    main()
