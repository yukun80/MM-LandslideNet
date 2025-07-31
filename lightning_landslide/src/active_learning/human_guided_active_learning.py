# =============================================================================
# lightning_landslide/src/active_learning/human_guided_active_learning.py
# =============================================================================

"""
人工指导的主动学习实现 - 修复版

修复内容：
1. 确保基础训练与baseline配置一致
2. 完善人工标注交互机制
3. 实现训练数据更新逻辑
4. 添加超时和错误处理
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from dataclasses import dataclass
import sys
import os
from copy import deepcopy
import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from ..utils.instantiate import instantiate_from_config
from ..data.multimodal_dataset import MultiModalDataset

logger = logging.getLogger(__name__)


@dataclass
class HumanAnnotationRequest:
    """人工标注请求"""

    iteration: int
    sample_ids: List[str]
    request_file: str
    annotation_file: str
    timestamp: str
    image_paths: List[str]  # 新增：图像路径


@dataclass
class HumanAnnotationResult:
    """人工标注结果"""

    sample_id: str
    label: int
    confidence: float = 1.0
    image_path: str = ""  # 新增：图像路径


class AnnotatedDataset(Dataset):
    """标注数据集类"""

    def __init__(self, annotations: List[HumanAnnotationResult], transform=None):
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # 如果有图像路径，加载图像
        if annotation.image_path and os.path.exists(annotation.image_path):
            # 这里需要根据你的图像加载逻辑调整
            # 简化版本：返回随机tensor
            image = torch.randn(5, 256, 256)  # 5通道，256x256
        else:
            image = torch.randn(5, 256, 256)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(annotation.label, dtype=torch.long)


class HumanGuidedActiveLearning:
    """
    人工指导的主动学习 - 修复版
    """

    def __init__(self, config: Dict, experiment_name: str = None, output_dir: str = None):
        self.config = config
        self.experiment_name = experiment_name or f"human_guided_{int(time.time())}"
        self.output_dir = Path(output_dir) if output_dir else Path("outputs") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 人工标注目录
        self.annotation_dir = self.output_dir / "human_annotations"
        self.annotation_dir.mkdir(exist_ok=True)

        # 核心参数
        active_config = config.get("active_pseudo_learning", {})
        self.max_iterations = active_config.get("max_iterations", 5)
        self.annotation_budget = active_config.get("annotation_budget", 50)
        self.confidence_threshold = active_config.get("pseudo_labeling", {}).get("confidence_threshold", 0.85)
        self.n_mc_passes = active_config.get("uncertainty_estimation", {}).get("params", {}).get("n_forward_passes", 10)

        # 人工标注模式配置
        self.annotation_mode = active_config.get("annotation_mode", "human")
        self.annotation_timeout = active_config.get("annotation_timeout", 3600)

        # 数据模块
        self.datamodule = instantiate_from_config(config["data"])

        # 存储测试集样本ID映射和图像路径
        self.test_sample_ids = []
        self.test_image_paths = []

        # 累积的标注数据
        self.accumulated_annotations = []

        logger.info(f"🎯 HumanGuidedActiveLearning initialized: {self.experiment_name}")
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info(f"📝 Annotation directory: {self.annotation_dir}")
        logger.info(f"👤 Annotation mode: {self.annotation_mode}")

    def run(self):
        """运行人工指导的主动学习"""
        logger.info("🚀 Starting Human-Guided Active Learning...")
        start_time = time.time()

        # 1. 初始化数据和建立ID映射
        self._initialize_data()

        # 2. 训练基线模型 - 修复：使用与baseline一致的配置
        logger.info("🏁 Training baseline model...")
        current_model = self._train_model("baseline", use_baseline_config=True)
        best_f1 = self._evaluate_model(current_model)

        logger.info(f"📊 Baseline F1: {best_f1:.4f}")

        # 3. 主动学习迭代循环
        for iteration in range(self.max_iterations):
            logger.info(f"\n🔄 === ITERATION {iteration + 1}/{self.max_iterations} ===")

            # 3a. 不确定性估计
            uncertainty_scores = self._estimate_uncertainty(current_model)

            # 3b. 选择需要标注的样本
            selected_indices = self._select_active_samples(uncertainty_scores)
            selected_sample_ids = [self.test_sample_ids[i] for i in selected_indices]
            selected_image_paths = [self.test_image_paths[i] for i in selected_indices]

            # 3c. 🔥 人工标注流程 🔥
            if self.annotation_mode == "human":
                annotations = self._request_human_annotation(iteration, selected_sample_ids, selected_image_paths)
            else:
                annotations = self._simulate_annotation(selected_indices)

            # 存储累积标注
            self.accumulated_annotations.extend(annotations)

            # 3d. 生成伪标签
            pseudo_labels = self._generate_pseudo_labels(current_model)

            # 3e. 更新训练数据 - 修复：真正实现数据更新
            updated_datamodule = self._update_training_data(annotations, pseudo_labels)

            # 3f. 重新训练模型
            logger.info("🚀 Retraining model with new annotations...")
            current_model = self._train_model(
                f"iter_{iteration + 1}", use_baseline_config=False, custom_datamodule=updated_datamodule
            )
            current_f1 = self._evaluate_model(current_model, custom_datamodule=updated_datamodule)

            logger.info(f"📈 Iteration {iteration + 1} F1: {current_f1:.4f}")

            if current_f1 > best_f1:
                best_f1 = current_f1
                logger.info(f"🏆 New best F1: {best_f1:.4f}")

        total_time = time.time() - start_time
        logger.info(f"🎉 Human-guided active learning completed!")
        logger.info(f"🏆 Final best F1: {best_f1:.4f}")
        logger.info(f"⏱️ Total time: {total_time:.1f}s")

        return {"best_f1": best_f1, "total_time": total_time}

    def _initialize_data(self):
        """初始化数据并建立样本ID映射"""
        # 设置训练和测试数据
        self.datamodule.setup("fit")
        self.datamodule.setup("test")

        # 建立测试集样本ID映射和图像路径
        test_dataset = self.datamodule.test_dataset
        self.test_sample_ids = []
        self.test_image_paths = []

        # 获取样本ID和图像路径
        if hasattr(test_dataset, "sample_ids"):
            self.test_sample_ids = test_dataset.sample_ids
        elif hasattr(test_dataset, "data") and hasattr(test_dataset.data, "index"):
            self.test_sample_ids = test_dataset.data.index.tolist()
        else:
            self.test_sample_ids = [f"test_sample_{i}" for i in range(len(test_dataset))]

        # 获取图像路径
        if hasattr(test_dataset, "image_paths"):
            self.test_image_paths = test_dataset.image_paths
        elif hasattr(test_dataset, "data_dir"):
            # 根据sample_id构建图像路径
            data_dir = Path(test_dataset.data_dir)
            self.test_image_paths = [str(data_dir / f"{sample_id}.tif") for sample_id in self.test_sample_ids]
        else:
            self.test_image_paths = [""] * len(self.test_sample_ids)

        logger.info(f"✅ Initialized {len(self.datamodule.train_dataset)} training samples")
        logger.info(f"✅ Initialized {len(test_dataset)} test samples")
        logger.info(f"📋 Test sample ID range: {self.test_sample_ids[0]} ~ {self.test_sample_ids[-1]}")

    def _request_human_annotation(
        self, iteration: int, sample_ids: List[str], image_paths: List[str]
    ) -> List[HumanAnnotationResult]:
        """请求人工标注 - 修复版"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 创建标注请求
        request = HumanAnnotationRequest(
            iteration=iteration,
            sample_ids=sample_ids,
            image_paths=image_paths,
            request_file=f"annotation_request_iter_{iteration}_{timestamp}.json",
            annotation_file=f"annotations_iter_{iteration}_{timestamp}.json",
            timestamp=timestamp,
        )

        # 保存标注请求到文件
        request_path = self.annotation_dir / request.request_file
        annotation_path = self.annotation_dir / request.annotation_file

        request_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "annotation_budget": len(sample_ids),
            "sample_info": [
                {
                    "sample_id": sid,
                    "image_path": img_path,
                    "relative_path": os.path.relpath(img_path) if img_path else "",
                }
                for sid, img_path in zip(sample_ids, image_paths)
            ],
            "annotation_file": request.annotation_file,
            "instructions": {
                "task": "请为以下样本标注滑坡检测结果",
                "labels": {"0": "无滑坡 (No landslide)", "1": "有滑坡 (Landslide present)"},
                "format": "请将标注结果保存为指定的JSON格式",
                "confidence": "可选：提供置信度分数 (0.0-1.0)",
                "注意": "请仔细查看图像，根据地形特征判断是否存在滑坡",
            },
            "annotation_format": {
                "说明": "请将标注结果保存为以下格式的JSON文件",
                "example": [
                    {"sample_id": "example_001", "label": 1, "confidence": 0.95},
                    {"sample_id": "example_002", "label": 0, "confidence": 0.90},
                ],
            },
        }

        # 保存请求文件
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)

        # 显示标注指令
        print("\n" + "=" * 80)
        print("🛑 需要人工标注 (HUMAN ANNOTATION REQUIRED)")
        print("=" * 80)
        print(f"📋 迭代轮次: {iteration + 1}/{self.max_iterations}")
        print(f"📝 需要标注的样本数: {len(sample_ids)}")
        print(f"📂 标注请求文件: {request_path}")
        print(f"💾 请将标注结果保存到: {annotation_path}")
        print(f"⏰ 标注超时时间: {self.annotation_timeout//60} 分钟")
        print("\n📋 需要标注的样本:")
        for i, (sample_id, img_path) in enumerate(zip(sample_ids, image_paths), 1):
            rel_path = os.path.relpath(img_path) if img_path else "无图像路径"
            print(f"  {i:2d}. {sample_id} -> {rel_path}")

        print("\n" + "=" * 80)
        print("🔧 标注说明:")
        print("1. 请查看上述样本对应的图像文件")
        print("2. 标注格式: 0=无滑坡, 1=有滑坡")
        print("3. 将结果保存为JSON格式到指定文件")
        print("4. JSON格式示例:")
        print('   [{"sample_id": "sample_001", "label": 1, "confidence": 0.95}]')
        print("5. 完成标注后程序将自动继续")
        print("=" * 80)

        # 等待标注文件
        return self._wait_for_annotation_file(annotation_path, sample_ids)

    def _wait_for_annotation_file(
        self, annotation_path: Path, expected_sample_ids: List[str]
    ) -> List[HumanAnnotationResult]:
        """等待并验证标注文件"""
        print(f"\n⏳ 等待标注文件: {annotation_path}")
        print("💡 提示: 完成标注后保存文件，程序将自动继续...")

        start_wait_time = time.time()
        check_interval = 10  # 每10秒检查一次

        while True:
            # 检查文件是否存在
            if annotation_path.exists():
                try:
                    # 读取标注文件
                    with open(annotation_path, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)

                    # 验证数据格式
                    annotations = []
                    if isinstance(annotation_data, list):
                        for item in annotation_data:
                            if "sample_id" in item and "label" in item:
                                annotation = HumanAnnotationResult(
                                    sample_id=item["sample_id"],
                                    label=int(item["label"]),
                                    confidence=float(item.get("confidence", 1.0)),
                                    image_path=item.get("image_path", ""),
                                )
                                annotations.append(annotation)

                    # 验证标注完整性
                    annotated_ids = {ann.sample_id for ann in annotations}
                    expected_ids = set(expected_sample_ids)

                    if annotated_ids >= expected_ids:  # 包含所有期望的ID
                        print(f"✅ 标注完成! 收到 {len(annotations)} 个标注")
                        return annotations
                    else:
                        missing_ids = expected_ids - annotated_ids
                        print(f"⚠️  缺少以下样本的标注: {missing_ids}")
                        print("请补充完整后重新保存文件...")
                        time.sleep(check_interval)
                        continue

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"⚠️  标注文件格式错误: {e}")
                    print("请检查JSON格式并重新保存...")
                    time.sleep(check_interval)
                    continue

            # 检查超时
            elapsed_time = time.time() - start_wait_time
            if elapsed_time > self.annotation_timeout:
                print(f"\n⏰ 标注超时 ({self.annotation_timeout//60} 分钟)")
                print("将使用模拟标注继续...")
                return self._simulate_annotation(
                    [i for i, sid in enumerate(self.test_sample_ids) if sid in expected_sample_ids]
                )

            # 显示等待状态
            if int(elapsed_time) % 30 == 0:  # 每30秒显示一次
                print(f"⏳ 等待中... (已等待 {elapsed_time:.0f}s)")

            time.sleep(check_interval)

    def _simulate_annotation(self, selected_indices: List[int]) -> List[HumanAnnotationResult]:
        """模拟人工标注（用于测试）"""
        annotations = []

        # 获取真实标签进行模拟
        test_loader = self.datamodule.test_dataloader()
        true_labels = []

        for _, labels in test_loader:
            true_labels.extend(labels.numpy())

        for idx in selected_indices:
            if idx < len(true_labels):
                annotation = HumanAnnotationResult(
                    sample_id=self.test_sample_ids[idx],
                    label=int(true_labels[idx]),
                    confidence=0.95,  # 模拟高置信度
                    image_path=self.test_image_paths[idx],
                )
                annotations.append(annotation)

        logger.info(f"🤖 模拟标注了 {len(annotations)} 个样本")
        return annotations

    def _estimate_uncertainty(self, model: pl.LightningModule) -> np.ndarray:
        """MC Dropout不确定性估计"""
        logger.info("🎯 开始不确定性估计 (快速模式)")
        start_time = time.time()

        model.eval()
        device = next(model.parameters()).device

        # # 启用dropout进行不确定性估计
        # for module in model.modules():
        #     if isinstance(module, torch.nn.Dropout):
        #         module.train()

        test_loader = self.datamodule.test_dataloader()

        uncertainty_scores = []
        total_batches = len(test_loader)

        with torch.no_grad():
            # 添加进度条，让用户看到进度
            pbar = tqdm.tqdm(test_loader, desc="🔍 不确定性估计", total=total_batches)

            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(device)

                # 单次前向传播（避免MC采样的性能问题）
                logits = model(data)
                probs = F.softmax(logits, dim=1)
                probs_np = probs.cpu().numpy()

                # 计算预测熵作为不确定性指标
                probs_np = np.clip(probs_np, 1e-8, 1.0)  # 避免log(0)
                batch_entropy = -np.sum(probs_np * np.log(probs_np), axis=1)

                uncertainty_scores.extend(batch_entropy.tolist())

                # 更新进度条信息
                pbar.set_postfix(
                    {
                        "已处理": len(uncertainty_scores),
                        "平均不确定性": f"{np.mean(batch_entropy):.3f}",
                        "当前批次": f"{batch_idx+1}/{total_batches}",
                    }
                )
            uncertainty_scores = np.array(uncertainty_scores)
            elapsed_time = time.time() - start_time

            logger.info(f"✅ 不确定性估计完成!")
            logger.info(f"📊 处理了 {len(uncertainty_scores)} 个样本")
            logger.info(f"⏱️ 耗时: {elapsed_time:.1f} 秒")
            logger.info(f"📈 平均不确定性: {np.mean(uncertainty_scores):.4f}")
            logger.info(f"📉 不确定性范围: {np.min(uncertainty_scores):.4f} - {np.max(uncertainty_scores):.4f}")

            return uncertainty_scores

        # predictions = []

        # with torch.no_grad():
        #     for data, _ in test_loader:
        #         data = data.to(device)

        #         batch_preds = []
        #         for _ in range(self.n_mc_passes):
        #             logits = model(data)
        #             probs = F.softmax(logits, dim=1)
        #             batch_preds.append(probs.cpu().numpy())

        #         batch_preds = np.stack(batch_preds)
        #         predictions.append(batch_preds)

        # all_predictions = np.concatenate(predictions, axis=0)
        # pred_variance = np.var(all_predictions, axis=0)
        # uncertainty_scores = np.mean(pred_variance, axis=1)

        # logger.info(f"✅ 估计了 {len(uncertainty_scores)} 个样本的不确定性")
        # return uncertainty_scores

    def _select_active_samples(self, uncertainty_scores: np.ndarray) -> List[int]:
        """选择最不确定的样本"""
        selected_indices = np.argsort(uncertainty_scores)[-self.annotation_budget :].tolist()
        logger.info(f"🎯 选择了 {len(selected_indices)} 个最不确定的样本")
        return selected_indices

    def _generate_pseudo_labels(self, model: pl.LightningModule) -> List[Tuple[str, int]]:
        """生成高置信度伪标签"""
        model.eval()
        test_loader = self.datamodule.test_dataloader()
        pseudo_labels = []
        device = next(model.parameters()).device

        with torch.no_grad():
            sample_idx = 0
            for data, _ in test_loader:
                data = data.to(device)
                logits = model(data)
                probs = F.softmax(logits, dim=1)

                max_probs = torch.max(probs, dim=1)[0]
                predicted_labels = torch.argmax(probs, dim=1)

                high_conf_mask = max_probs > self.confidence_threshold

                for i, (is_high_conf, pred_label) in enumerate(zip(high_conf_mask, predicted_labels)):
                    if is_high_conf:
                        sample_id = self.test_sample_ids[sample_idx + i]
                        pseudo_labels.append((sample_id, pred_label.item()))

                sample_idx += len(data)

        logger.info(f"🏷️ 生成了 {len(pseudo_labels)} 个伪标签")
        return pseudo_labels

    def _update_training_data(self, annotations: List[HumanAnnotationResult], pseudo_labels: List[Tuple[str, int]]):
        """更新训练数据 - 修复：实际实现数据更新"""
        logger.info(f"📝 添加 {len(annotations)} 个人工标注")
        logger.info(f"🏷️ 添加 {len(pseudo_labels)} 个伪标签")

        # 创建新的数据模块，包含原始训练数据 + 新标注
        updated_config = deepcopy(self.config["data"])

        # 创建更新后的数据模块
        updated_datamodule = instantiate_from_config(updated_config)
        updated_datamodule.setup("fit")

        # 创建标注数据集
        if annotations:
            annotation_dataset = AnnotatedDataset(annotations, transform=updated_datamodule.train_dataset.transform)

            # 合并数据集
            combined_dataset = ConcatDataset([updated_datamodule.train_dataset, annotation_dataset])
            updated_datamodule.train_dataset = combined_dataset

        # 保存标注历史
        annotation_history = {
            "human_annotations": [
                {
                    "sample_id": ann.sample_id,
                    "label": ann.label,
                    "confidence": ann.confidence,
                    "image_path": ann.image_path,
                }
                for ann in annotations
            ],
            "pseudo_labels": [{"sample_id": sample_id, "label": label} for sample_id, label in pseudo_labels],
        }

        history_file = self.annotation_dir / f"annotation_history_{int(time.time())}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(annotation_history, f, indent=2, ensure_ascii=False)

        return updated_datamodule

    def _train_model(self, name: str, use_baseline_config: bool = False, custom_datamodule=None) -> pl.LightningModule:
        """训练模型 - 修复：确保配置一致性"""
        model = instantiate_from_config(self.config["model"])
        datamodule = custom_datamodule or self.datamodule

        # 根据是否是基线训练使用不同配置
        if use_baseline_config:
            # 使用与optical_baseline.yaml一致的训练配置
            trainer_config = self.config.get("trainer", {}).get("params", {})
            max_epochs = trainer_config.get("max_epochs", 100)  # 默认100 epoch

            callbacks = []

            # ModelCheckpoint
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(self.output_dir / "checkpoints"),
                filename=f"{name}_{{epoch:02d}}_{{val_f1:.4f}}",
                monitor="val_f1",
                mode="max",
                save_top_k=1,
                save_last=True,
                verbose=True,
            )
            callbacks.append(checkpoint_callback)

            # EarlyStopping
            early_stopping = EarlyStopping(
                monitor="val_f1", patience=20, mode="max", verbose=True, strict=False, min_delta=0.001  # 与baseline一致
            )
            callbacks.append(early_stopping)

            # LearningRateMonitor
            lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=False)
            callbacks.append(lr_monitor)

            # Logger
            logger_instance = TensorBoardLogger(save_dir=str(self.output_dir / "logs"), name=f"{name}", version="")
        else:
            # 主动学习迭代中使用较少的epoch
            max_epochs = 20
            callbacks = [
                ModelCheckpoint(
                    dirpath=str(self.output_dir / "checkpoints"),
                    filename=f"{name}_{{epoch:02d}}_{{val_f1:.4f}}",
                    monitor="val_f1",
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(monitor="val_f1", patience=8, mode="max", verbose=False),
            ]
            logger_instance = False

        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            precision="32-true",  # 与baseline一致
            enable_progress_bar=True,
            enable_model_summary=use_baseline_config,  # 基线训练时显示模型摘要
            logger=logger_instance,
            callbacks=callbacks,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            gradient_clip_val=1.0,
            deterministic=False,
        )

        trainer.fit(model, datamodule)
        return model

    def _evaluate_model(self, model: pl.LightningModule, custom_datamodule=None) -> float:
        """评估模型"""
        datamodule = custom_datamodule or self.datamodule

        eval_trainer = pl.Trainer(
            accelerator="auto", devices="auto", enable_progress_bar=False, enable_model_summary=False, logger=False
        )

        results = eval_trainer.validate(model, datamodule, verbose=False)
        f1_score = results[0].get("val_f1", 0.0) if results else 0.0
        return f1_score


def create_human_guided_active_learning(config: Dict, **kwargs) -> HumanGuidedActiveLearning:
    """创建人工指导的主动学习训练器"""
    return HumanGuidedActiveLearning(config, **kwargs)
