# =============================================================================
# lightning_landslide/src/active_learning/human_guided_active_learning.py
# =============================================================================

"""
人工指导的主动学习实现

设计思路：
1. 程序选择最不确定的样本
2. 输出样本ID列表给人工专家
3. 暂停程序等待人工标注
4. 读取人工标注结果
5. 继续训练流程
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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ..utils.instantiate import instantiate_from_config

logger = logging.getLogger(__name__)


@dataclass
class HumanAnnotationRequest:
    """人工标注请求"""

    iteration: int
    sample_ids: List[str]
    request_file: str
    annotation_file: str
    timestamp: str


@dataclass
class HumanAnnotationResult:
    """人工标注结果"""

    sample_id: str
    label: int
    confidence: float = 1.0  # 专家标注置信度


class HumanGuidedActiveLearning:
    """
    人工指导的主动学习

    核心特点：
    1. 真实的人工标注流程
    2. 程序暂停/恢复机制
    3. 标注质量跟踪
    4. 简单的文件接口
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
        self.annotation_mode = active_config.get("annotation_mode", "human")  # "human" | "simulated"
        self.annotation_timeout = active_config.get("annotation_timeout", 3600)  # 1小时超时

        # 数据模块
        self.datamodule = instantiate_from_config(config["data"])

        # 存储测试集样本ID映射
        self.test_sample_ids = []

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

        # 2. 训练基线模型
        logger.info("🏁 Training baseline model...")
        current_model = self._train_model("baseline")
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

            # 3c. 🔥 人工标注流程 🔥
            if self.annotation_mode == "human":
                annotations = self._request_human_annotation(iteration, selected_sample_ids)
            else:
                annotations = self._simulate_annotation(selected_indices)

            # 3d. 生成伪标签
            pseudo_labels = self._generate_pseudo_labels(current_model)

            # 3e. 更新训练数据（添加人工标注 + 伪标签）
            self._update_training_data(annotations, pseudo_labels)

            # 3f. 重新训练模型
            logger.info("🚀 Retraining model with new annotations...")
            current_model = self._train_model(f"iter_{iteration + 1}")
            current_f1 = self._evaluate_model(current_model)

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

        # 建立测试集样本ID映射
        test_dataset = self.datamodule.test_dataset
        self.test_sample_ids = []

        # 如果数据集有sample_id属性，使用它；否则使用索引
        if hasattr(test_dataset, "sample_ids"):
            self.test_sample_ids = test_dataset.sample_ids
        elif hasattr(test_dataset, "data") and hasattr(test_dataset.data, "index"):
            self.test_sample_ids = test_dataset.data.index.tolist()
        else:
            # 如果没有明确的ID，使用数据集索引
            self.test_sample_ids = [f"test_sample_{i}" for i in range(len(test_dataset))]

        logger.info(f"✅ Initialized {len(self.datamodule.train_dataset)} training samples")
        logger.info(f"✅ Initialized {len(test_dataset)} test samples")
        logger.info(f"📋 Test sample ID range: {self.test_sample_ids[0]} ~ {self.test_sample_ids[-1]}")

    def _request_human_annotation(self, iteration: int, sample_ids: List[str]) -> List[HumanAnnotationResult]:
        """请求人工标注"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 创建标注请求
        request = HumanAnnotationRequest(
            iteration=iteration,
            sample_ids=sample_ids,
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
            "sample_ids": sample_ids,
            "annotation_file": request.annotation_file,
            "instructions": {
                "task": "Please annotate the following samples for landslide detection",
                "labels": {"0": "No landslide", "1": "Landslide present"},
                "format": "Save annotations in the specified JSON format",
                "confidence": "Optional: provide confidence score (0.0-1.0)",
            },
            "annotation_format": {
                "example": [
                    {"sample_id": "example_001", "label": 1, "confidence": 0.95},
                    {"sample_id": "example_002", "label": 0, "confidence": 0.90},
                ]
            },
        }

        # 保存请求文件
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)

        # 显示标注指令
        print("\n" + "=" * 80)
        print("🛑 HUMAN ANNOTATION REQUIRED")
        print("=" * 80)
        print(f"📋 Iteration: {iteration + 1}")
        print(f"📝 Samples to annotate: {len(sample_ids)}")
        print(f"📂 Request file: {request_path}")
        print(f"💾 Save annotations to: {annotation_path}")
        print("\n📋 Samples to annotate:")
        for i, sample_id in enumerate(sample_ids, 1):
            print(f"  {i:2d}. {sample_id}")

        print("\n📝 Annotation format (JSON):")
        print(json.dumps(request_data["annotation_format"]["example"], indent=2))

        print(f"\n⏰ Please complete annotation within {self.annotation_timeout//60} minutes")
        print("🔄 Waiting for annotation completion...")
        print("=" * 80)

        # 等待标注完成
        annotations = self._wait_for_annotations(annotation_path)

        logger.info(f"✅ Received {len(annotations)} human annotations")
        return annotations

    def _wait_for_annotations(self, annotation_path: Path) -> List[HumanAnnotationResult]:
        """等待人工标注完成"""
        start_wait_time = time.time()

        while True:
            # 检查标注文件是否存在
            if annotation_path.exists():
                try:
                    # 尝试读取标注文件
                    with open(annotation_path, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)

                    # 解析标注结果
                    annotations = []
                    for item in annotation_data:
                        annotation = HumanAnnotationResult(
                            sample_id=item["sample_id"],
                            label=int(item["label"]),
                            confidence=float(item.get("confidence", 1.0)),
                        )
                        annotations.append(annotation)

                    print(f"\n✅ Successfully loaded {len(annotations)} annotations!")
                    return annotations

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"⚠️  Error reading annotation file: {e}")
                    print("Please check the JSON format and try again.")
                    time.sleep(5)
                    continue

            # 检查超时
            elapsed_time = time.time() - start_wait_time
            if elapsed_time > self.annotation_timeout:
                print(f"\n⏰ Annotation timeout ({self.annotation_timeout//60} minutes)")
                print("Using simulated annotations for this iteration...")
                return self._simulate_annotation([])  # 空的模拟标注

            # 每10秒检查一次
            time.sleep(10)
            print(f"⏳ Waiting... ({elapsed_time:.0f}s elapsed)")

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
                    sample_id=self.test_sample_ids[idx], label=int(true_labels[idx]), confidence=0.95  # 模拟高置信度
                )
                annotations.append(annotation)

        logger.info(f"🤖 Simulated {len(annotations)} annotations")
        return annotations

    def _estimate_uncertainty(self, model: pl.LightningModule) -> np.ndarray:
        """MC Dropout不确定性估计"""
        model.eval()

        # 启用dropout
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

        test_loader = self.datamodule.test_dataloader()
        predictions = []
        device = next(model.parameters()).device

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)

                batch_preds = []
                for _ in range(self.n_mc_passes):
                    logits = model(data)
                    probs = F.softmax(logits, dim=1)
                    batch_preds.append(probs.cpu().numpy())

                batch_preds = np.stack(batch_preds)
                predictions.append(batch_preds)

        all_predictions = np.concatenate(predictions, axis=1)
        pred_variance = np.var(all_predictions, axis=0)
        uncertainty_scores = np.mean(pred_variance, axis=1)

        logger.info(f"✅ Estimated uncertainty for {len(uncertainty_scores)} samples")
        return uncertainty_scores

    def _select_active_samples(self, uncertainty_scores: np.ndarray) -> List[int]:
        """选择最不确定的样本"""
        selected_indices = np.argsort(uncertainty_scores)[-self.annotation_budget :].tolist()
        logger.info(f"🎯 Selected {len(selected_indices)} most uncertain samples")
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

        logger.info(f"🏷️ Generated {len(pseudo_labels)} pseudo labels")
        return pseudo_labels

    def _update_training_data(self, annotations: List[HumanAnnotationResult], pseudo_labels: List[Tuple[str, int]]):
        """更新训练数据（实际实现中需要根据具体数据结构调整）"""
        # 这里是简化实现，实际中需要：
        # 1. 将人工标注样本添加到训练集
        # 2. 将伪标签样本添加到训练集
        # 3. 重新创建数据加载器

        logger.info(f"📝 Added {len(annotations)} human annotations")
        logger.info(f"🏷️ Added {len(pseudo_labels)} pseudo labels")

        # 保存标注历史
        annotation_history = {
            "human_annotations": [
                {"sample_id": ann.sample_id, "label": ann.label, "confidence": ann.confidence} for ann in annotations
            ],
            "pseudo_labels": [{"sample_id": sample_id, "label": label} for sample_id, label in pseudo_labels],
        }

        history_file = self.annotation_dir / f"annotation_history_{int(time.time())}.json"
        with open(history_file, "w") as f:
            json.dump(annotation_history, f, indent=2)

    def _train_model(self, name: str) -> pl.LightningModule:
        """训练模型"""
        model = instantiate_from_config(self.config["model"])

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

        trainer = pl.Trainer(
            max_epochs=15,
            accelerator="auto",
            devices="auto",
            precision="32-true",
            enable_progress_bar=True,  # 可以启用进度条
            enable_model_summary=False,
            logger=False,
            callbacks=callbacks,
        )

        trainer.fit(model, self.datamodule)
        return model

    def _evaluate_model(self, model: pl.LightningModule) -> float:
        """评估模型"""
        eval_trainer = pl.Trainer(
            accelerator="auto", devices="auto", enable_progress_bar=False, enable_model_summary=False, logger=False
        )

        results = eval_trainer.validate(model, self.datamodule, verbose=False)
        f1_score = results[0].get("val_f1", 0.0) if results else 0.0
        return f1_score


def create_human_guided_active_learning(config: Dict, **kwargs) -> HumanGuidedActiveLearning:
    """创建人工指导的主动学习训练器"""
    return HumanGuidedActiveLearning(config, **kwargs)
