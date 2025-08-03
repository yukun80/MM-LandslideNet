#!/usr/bin/env python3
"""
滑坡标注辅助工具

功能：
1. 批量显示待标注样本的图像
2. 交互式标注界面
3. 自动生成标准格式的annotation_results.json
4. 进度跟踪和保存

使用方法：
python lightning_landslide/src/active_learning/annotation_helper.py
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 🔧 新增：字体管理
import warnings  # 🔧 新增：警告处理
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox, simpledialog


def configure_chinese_matplotlib():
    """
    配置matplotlib支持中文显示，抑制字体警告

    遵循原则：
    1. 最小改动：只添加字体配置，不修改其他逻辑
    2. 单一职责：专门处理字体配置问题
    3. 渐进增强：在现有基础上添加字体支持
    """
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

    # 配置matplotlib字体
    if selected_font:
        plt.rcParams.update(
            {
                "font.sans-serif": [selected_font, "DejaVu Sans"],
                "axes.unicode_minus": False,
                "font.size": 10,
                "font.family": "sans-serif",
            }
        )
        print(f"🎯 字体配置完成: {selected_font}")
    else:
        print("⚠️ 未找到中文字体，使用英文显示")
        plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})

    # 🔧 关键修复：抑制字体警告
    # 这样即使有字体回退也不会显示警告
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

    return selected_font


class LandslideAnnotationTool:
    def __init__(self, iteration: int = 0):
        self.iteration = iteration
        # 🔧 新增：在初始化时配置字体
        self.selected_font = configure_chinese_matplotlib()

        self.annotation_request_file = (
            "lightning_landslide/exp/multimodal_swin_tiny_0803/active_learning/annotation_request_iter_0.json"
        )

        self.output_file = f"lightning_landslide/exp/multimodal_swin_tiny_0803/active_learning/annotation_results_iter_{iteration}.json"
        self.output_file_generic = (
            "lightning_landslide/exp/multimodal_swin_tiny_0803/active_learning/annotation_results.json"
        )
        self.image_dir = Path("dataset/datavision/test_data")

        # 加载待标注样本
        self.load_annotation_request()

        # 初始化标注结果
        self.annotations = {}
        self.current_index = 0

        # 如果已有部分标注结果，加载它们
        self.load_existing_annotations()

    def load_annotation_request(self):
        """加载标注请求文件"""
        with open(self.annotation_request_file, "r", encoding="utf-8") as f:
            request_data = json.load(f)

        self.sample_list = request_data["selected_samples"]
        self.sample_details = {item["sample_id"]: item for item in request_data["sample_details"]}

        print(f"📋 加载了 {len(self.sample_list)} 个待标注样本")

    def load_existing_annotations(self):
        """加载已有的标注结果"""
        if Path(self.output_file).exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "annotations" in data:
                    self.annotations = data["annotations"]
                    annotated_count = len([v for v in self.annotations.values() if v is not None])
                    print(f"📝 加载了 {annotated_count} 个已完成的标注")

    def show_sample_info(self, sample_id):
        """显示样本信息"""
        if sample_id in self.sample_details:
            details = self.sample_details[sample_id]
            print(f"\n" + "=" * 60)
            print(f"🔍 样本ID: {sample_id}")
            print(f"📊 不确定性分数: {details['uncertainty_score']:.6f}")
            print(f"📁 图像路径: {self.image_dir / f'{sample_id}.png'}")
            print(f"📈 进度: {self.current_index + 1}/{len(self.sample_list)}")
            print("=" * 60)

    def display_image(self, sample_id):
        """显示图像"""
        image_path = self.image_dir / f"{sample_id}.png"

        if not image_path.exists():
            print(f"⚠️ 图像文件不存在: {image_path}")
            return False

        try:
            # 使用matplotlib显示图像
            img = Image.open(image_path)

            plt.figure(figsize=(10, 8))
            plt.imshow(img)

            # 🔧 修复：确保标题使用正确的字体
            title_text = f"Sample: {sample_id}\nUncertainty: {self.sample_details[sample_id]['uncertainty_score']:.6f}"
            if self.selected_font and self.selected_font != "DejaVu Sans":
                plt.title(title_text, fontfamily=self.selected_font)
            else:
                plt.title(title_text)

            plt.axis("off")

            # 🔧 修复：标注指南也使用正确的字体
            guide_text = "标注指南:\n" "1 = 滑坡区域 (土壤暴露、植被中断)\n" "0 = 非滑坡区域 (植被正常、结构规整)"

            figtext_kwargs = {
                "x": 0.02,
                "y": 0.02,
                "s": guide_text,
                "fontsize": 10,
                "bbox": dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            }

            if self.selected_font and self.selected_font != "DejaVu Sans":
                figtext_kwargs["fontfamily"] = self.selected_font

            plt.figtext(**figtext_kwargs)

            plt.show(block=False)
            plt.pause(0.1)  # 确保图像显示

            return True

        except Exception as e:
            print(f"❌ 显示图像时出错: {e}")
            return False

    def get_annotation(self, sample_id):
        """获取用户标注"""
        while True:
            try:
                print(f"\n请为样本 {sample_id} 标注:")
                print("1 = 滑坡区域")
                print("0 = 非滑坡区域")
                print("s = 跳过此样本")
                print("q = 退出并保存")
                print("b = 返回上一个样本")

                choice = input("请输入选择 (1/0/s/q/b): ").strip().lower()

                if choice == "1":
                    return 1
                elif choice == "0":
                    return 0
                elif choice == "s":
                    return None  # 跳过
                elif choice == "q":
                    return "quit"
                elif choice == "b":
                    return "back"
                else:
                    print("❌ 无效输入，请输入 1、0、s、q 或 b")

            except KeyboardInterrupt:
                print("\n\n🛑 标注被中断")
                return "quit"

    def save_annotations(self):
        """保存标注结果（保存两个版本）"""
        # 统计标注情况
        total_samples = len(self.sample_list)
        annotated_samples = len([v for v in self.annotations.values() if v is not None])
        landslide_count = len([v for v in self.annotations.values() if v == 1])
        non_landslide_count = len([v for v in self.annotations.values() if v == 0])

        result = {
            "metadata": {
                "iteration": self.iteration,  # 🔥 新增：记录迭代次数
                "total_samples": total_samples,
                "annotated_samples": annotated_samples,
                "landslide_samples": landslide_count,
                "non_landslide_samples": non_landslide_count,
                "completion_rate": f"{annotated_samples/total_samples*100:.1f}%",
                "annotation_date": "2025-07-31",
                "annotation_method": "interactive_visual_inspection",
            },
            "annotations": self.annotations,
        }

        # 🔥 保存带iteration编号的文件（与ActiveRetrainer匹配）
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 🔥 同时保存通用文件（向后兼容）
        with open(self.output_file_generic, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 标注结果已保存到:")
        print(f"  📁 {self.output_file}")
        print(f"  📁 {self.output_file_generic}")
        print(f"📊 标注统计: {annotated_samples}/{total_samples} 完成 ({annotated_samples/total_samples*100:.1f}%)")
        print(f"📊 滑坡样本: {landslide_count}, 非滑坡样本: {non_landslide_count}")

    def run_annotation(self):
        """运行交互式标注"""
        print("🚀 开始滑坡标注任务")
        print("💡 提示：可以随时按 Ctrl+C 中断并保存当前进度")

        # 按不确定性分数排序（高到低）
        sorted_samples = []
        for sample_id in self.sample_list:
            if sample_id in self.sample_details:
                uncertainty = self.sample_details[sample_id]["uncertainty_score"]
                sorted_samples.append((sample_id, uncertainty))

        sorted_samples.sort(key=lambda x: x[1], reverse=True)

        self.current_index = 0

        while self.current_index < len(sorted_samples):
            sample_id, uncertainty = sorted_samples[self.current_index]

            # 显示样本信息
            self.show_sample_info(sample_id)

            # 如果已经标注过，显示现有标注
            if sample_id in self.annotations and self.annotations[sample_id] is not None:
                current_label = self.annotations[sample_id]
                print(f"📝 当前标注: {current_label} ({'滑坡' if current_label == 1 else '非滑坡'})")

            # 显示图像
            if self.display_image(sample_id):
                # 获取标注
                annotation = self.get_annotation(sample_id)

                if annotation == "quit":
                    break
                elif annotation == "back":
                    if self.current_index > 0:
                        self.current_index -= 1
                        plt.close("all")  # 关闭当前图像
                        continue
                    else:
                        print("📍 已经是第一个样本")
                        continue
                elif annotation is not None:
                    self.annotations[sample_id] = annotation
                    label_text = "滑坡区域" if annotation == 1 else "非滑坡区域"
                    print(f"✅ 标注完成: {sample_id} -> {annotation} ({label_text})")

                # 关闭当前图像
                plt.close("all")

            self.current_index += 1

            # 每10个样本自动保存一次
            if self.current_index % 10 == 0:
                self.save_annotations()
                print(f"💾 进度已自动保存 ({self.current_index}/{len(sorted_samples)})")

        # 最终保存
        self.save_annotations()
        print("🎉 标注任务完成！")


def main():
    iteration = 0
    tool = LandslideAnnotationTool(iteration=iteration)
    tool.run_annotation()


if __name__ == "__main__":
    main()
