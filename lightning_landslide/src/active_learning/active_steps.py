# =============================================================================
# lightning_landslide/src/active_learning/active_steps.py
# 主动学习步骤模块 - 重构版
# =============================================================================

"""
主动学习步骤分解实现

⚠️  重要说明：
本模块只实现主动学习的步骤2-5，不包含从头开始的模型训练。
必须先通过 `python main.py train config.yaml` 完成基础训练（步骤1）。

遵循三个原则：
1. 最小改动原则：重用现有训练流程和数据模块
2. 单一职责原则：每个类只负责一个特定步骤
3. 渐进增强原则：在现有代码基础上添加功能

新的命令体系：
- 步骤1：python main.py train config.yaml                    # 基础训练（必须先执行）
- 步骤2：python main.py uncertainty_estimation config.yaml   # 不确定性估计
- 步骤3：python main.py sample_selection config.yaml         # 样本选择
- 步骤4：人工标注（离线完成）
- 步骤5：python main.py retrain config.yaml                  # 模型fine-tuning

核心设计：
- 所有步骤都基于步骤1产生的检查点文件
- 步骤5是fine-tuning，不是从头训练
- 每个步骤都有完整的状态管理和错误恢复
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dataclasses import dataclass
import pickle
from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..utils.instantiate import instantiate_from_config

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningState:
    """主动学习状态管理"""

    # 基本信息
    experiment_name: str
    checkpoint_path: str
    iteration: int = 0

    # 数据信息
    unlabeled_pool: List[str] = None  # 未标注样本ID列表
    labeled_samples: List[str] = None  # 已标注样本ID列表
    annotation_history: List[Dict] = None  # 标注历史

    # 结果信息
    uncertainty_scores: Dict[str, float] = None  # 不确定性分数
    selected_samples: List[str] = None  # 当前选中的样本
    pseudo_labels: List[Dict] = None  # 伪标签结果

    def __post_init__(self):
        if self.unlabeled_pool is None:
            self.unlabeled_pool = []
        if self.labeled_samples is None:
            self.labeled_samples = []
        if self.annotation_history is None:
            self.annotation_history = []
        # 🔧 新增：初始化缺失的字段
        if self.selected_samples is None:
            self.selected_samples = []
        if self.pseudo_labels is None:
            self.pseudo_labels = []
        # 🔧 新增：初始化 uncertainty_scores（如果需要）
        if self.uncertainty_scores is None:
            self.uncertainty_scores = {}

    def save(self, save_path: Path):
        """保存状态到文件"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Active learning state saved to: {save_path}")

    @classmethod
    def load(cls, load_path: Path):
        """从文件加载状态"""
        if not load_path.exists():
            raise FileNotFoundError(f"State file not found: {load_path}")

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        # 🔧 新增：确保加载的状态也有正确的字段初始化
        if not hasattr(state, "selected_samples") or state.selected_samples is None:
            state.selected_samples = []
        if not hasattr(state, "pseudo_labels") or state.pseudo_labels is None:
            state.pseudo_labels = []
        if not hasattr(state, "uncertainty_scores") or state.uncertainty_scores is None:
            state.uncertainty_scores = {}

        logger.info(f"Active learning state loaded from: {load_path}")
        return state

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, experiment_name: str):
        """从检查点创建初始状态"""
        return cls(experiment_name=experiment_name, checkpoint_path=checkpoint_path, iteration=0)


class BaseActiveStep(ABC):
    """主动学习步骤基类"""

    def __init__(self, config: Dict[str, Any], state_path: Optional[str] = None):
        self.config = config
        self.state_path = Path(state_path) if state_path else None
        self.state = self._load_or_create_state()

        # 设置输出目录
        self.output_dir = Path(config.get("outputs", {}).get("experiment_dir", "outputs"))
        self.active_dir = self.output_dir / "active_learning"
        self.active_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_state(self) -> ActiveLearningState:
        """加载或创建状态"""
        if self.state_path and self.state_path.exists():
            return ActiveLearningState.load(self.state_path)
        else:
            # 从配置创建初始状态
            experiment_name = self.config.get("experiment_name", "active_learning")
            checkpoint_path = self._find_best_checkpoint()
            return ActiveLearningState.from_checkpoint(checkpoint_path, experiment_name)

    def _find_best_checkpoint(self) -> str:
        """寻找最佳检查点文件"""
        # 从配置或默认位置寻找检查点
        checkpoint_path = self.config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return checkpoint_path

        # 在实验目录下寻找最佳检查点
        exp_dir = Path(self.config.get("outputs", {}).get("experiment_dir", "outputs"))
        checkpoint_dir = exp_dir / "checkpoints"

        if checkpoint_dir.exists():
            # 🔧 修复：正确的检查点搜索模式
            # 尝试多种常见的检查点命名模式
            patterns = "best-*.ckpt"  # 通用best模式

            best_ckpts = list(checkpoint_dir.glob(patterns))
            if best_ckpts:
                logger.info(f"Found {len(best_ckpts)} checkpoints with pattern: {patterns}")

                # 选择最新的文件
                latest_ckpt = max(best_ckpts, key=lambda x: x.stat().st_mtime)
                logger.info(f"Found checkpoint: {latest_ckpt}")
                return str(latest_ckpt)

        raise FileNotFoundError("No valid checkpoint found. Please run baseline training first.")

    def save_state(self):
        """保存当前状态"""
        state_file = self.active_dir / f"state_iter_{self.state.iteration}.pkl"
        self.state.save(state_file)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """运行当前步骤"""
        pass


class UncertaintyEstimator(BaseActiveStep):
    """步骤2：不确定性估计"""

    def __init__(self, config: Dict[str, Any], state_path: Optional[str] = None):
        super().__init__(config, state_path)

        # 不确定性估计配置
        uncertainty_config = config.get("active_pseudo_learning", {}).get("uncertainty_estimation", {})
        self.method = uncertainty_config.get("method", "mc_dropout")
        self.n_forward_passes = uncertainty_config.get("params", {}).get("n_forward_passes", 10)

        logger.info(f"🔍 UncertaintyEstimator initialized")
        logger.info(f"📊 Method: {self.method}, Forward passes: {self.n_forward_passes}")

    def run(self) -> Dict[str, Any]:
        """运行不确定性估计"""
        logger.info("🔍 Starting uncertainty estimation...")

        # 1. 加载模型
        model = self._load_model()

        # 2. 获取未标注数据
        datamodule = self._setup_datamodule()

        # 3. 估计不确定性
        uncertainty_scores = self._estimate_uncertainty(model, datamodule)

        # 4. 更新状态
        self.state.uncertainty_scores = uncertainty_scores
        self.save_state()

        # 5. 保存结果
        results_path = self.active_dir / f"uncertainty_scores_iter_{self.state.iteration}.json"
        with open(results_path, "w") as f:
            json.dump(uncertainty_scores, f, indent=2)

        logger.info(f"✅ Uncertainty estimation completed")
        logger.info(f"📁 Results saved to: {results_path}")
        logger.info(f"📊 Estimated uncertainty for {len(uncertainty_scores)} samples")

        return {
            "uncertainty_scores": uncertainty_scores,
            "results_path": str(results_path),
            "num_samples": len(uncertainty_scores),
        }

    def _load_model(self) -> pl.LightningModule:
        """加载模型 - GPU优化版本"""
        logger.info(f"📥 Loading model from: {self.state.checkpoint_path}")

        # 重用现有的模型实例化逻辑
        model = instantiate_from_config(self.config["model"])

        # 🔧 修复1：检查GPU可用性并设置正确的设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 Using device: {device}")

        # 🔧 修复2：根据设备类型加载检查点
        if device == "cuda":
            checkpoint = torch.load(self.state.checkpoint_path, map_location="cuda")
        else:
            checkpoint = torch.load(self.state.checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["state_dict"])

        # 🔧 修复3：明确移动模型到GPU
        model = model.to(device)
        model.eval()

        # 设置dropout为训练模式（用于MC Dropout）
        if self.method == "mc_dropout":
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        return model

    def _setup_datamodule(self):
        """设置数据模块"""
        # 重用现有的数据模块
        datamodule = instantiate_from_config(self.config["data"])
        datamodule.setup("test")  # 使用测试集作为未标注池
        return datamodule

    def _estimate_uncertainty(self, model: pl.LightningModule, datamodule) -> Dict[str, float]:
        """
        🔥 修复版本：使用真实sample ID的不确定性估计

        关键修复：
        1. 直接从数据集获取真实sample ID
        2. 替换虚拟ID生成逻辑
        3. 确保结果可追溯到原始数据
        """
        logger.info(f"🔄 Running {self.method} with {self.n_forward_passes} forward passes...")
        logger.info("🔍 FIXED VERSION: Using real sample IDs from dataset")

        # 🔧 修复1：确保设备正确
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        device = next(model.parameters()).device
        test_loader = datamodule.test_dataloader()
        uncertainty_scores = {}

        # 激活dropout用于MC Dropout
        if self.method == "mc_dropout":
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        total_batches = len(test_loader)
        logger.info(f"📊 Total batches to process: {total_batches}")

        # 🔥 关键修复：获取数据集以访问真实ID
        test_dataset = datamodule.test_dataset

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                try:
                    # 解析batch数据
                    if isinstance(batch, list) and len(batch) >= 2:
                        images, labels = batch[0], batch[1]
                        batch_size = images.size(0)
                    else:
                        logger.warning(f"⚠️ Unexpected batch format at {batch_idx}")
                        continue

                    # 🔥 核心修复：计算真实的数据集索引并获取真实ID
                    batch_start_idx = batch_idx * test_loader.batch_size
                    real_sample_ids = []

                    for i in range(batch_size):
                        dataset_idx = batch_start_idx + i
                        if dataset_idx < len(test_dataset):
                            # 直接从数据集获取真实ID
                            real_id = test_dataset.data_index.iloc[dataset_idx]["ID"]
                            real_sample_ids.append(real_id)
                        else:
                            # 安全回退
                            real_sample_ids.append(f"sample_{batch_idx}_{i}")

                    # 验证获取的真实ID
                    if batch_idx == 0:  # 只在第一批显示示例
                        logger.info(f"✅ Real IDs example: {real_sample_ids[:3]}")
                        logger.info(f"🔍 ID format validation: {real_sample_ids[0].startswith('ID_')}")

                    images = images.to(device)
                    batch_predictions = []

                    # MC Dropout前向传播
                    for pass_idx in range(self.n_forward_passes):
                        output = model(images)
                        if output.dim() == 2 and output.size(1) == 1:
                            output = output.squeeze(1)

                        probs = torch.sigmoid(output)
                        batch_predictions.append(probs.cpu().numpy())

                    # 计算不确定性
                    batch_predictions = np.array(batch_predictions)  # [n_passes, batch_size]

                    for i, real_id in enumerate(real_sample_ids):
                        try:
                            sample_pred = batch_predictions[:, i]  # [n_passes]

                            # 计算方差和熵
                            mean_pred = np.mean(sample_pred)
                            variance = np.var(sample_pred)

                            # 二分类熵
                            p = np.clip(mean_pred, 1e-8, 1 - 1e-8)
                            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))

                            # 组合不确定性
                            uncertainty = variance + 0.1 * entropy
                            uncertainty_scores[str(real_id)] = float(uncertainty)

                        except Exception as e:
                            logger.error(f"❌ Error computing uncertainty for sample {real_id}: {e}")
                            continue

                    # 🔥 改进的进度报告
                    if (batch_idx + 1) % 10 == 0:
                        progress_pct = (batch_idx + 1) / total_batches * 100
                        samples_processed = len(uncertainty_scores)

                        # 统计真实ID数量
                        real_id_count = sum(1 for k in uncertainty_scores.keys() if k.startswith("ID_"))
                        fake_id_count = samples_processed - real_id_count

                        logger.info(f"🔄 Progress: {progress_pct:.1f}% ({batch_idx+1}/{total_batches} batches)")
                        logger.info(
                            f"📊 Processed: {samples_processed} samples ({real_id_count} real IDs, {fake_id_count} fallback IDs)"
                        )

                    # 内存清理
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                    continue

        # 最终验证
        if len(uncertainty_scores) == 0:
            raise ValueError("No samples processed successfully")

        # 🔥 结果验证和统计
        total_samples = len(uncertainty_scores)
        real_id_samples = sum(1 for k in uncertainty_scores.keys() if k.startswith("ID_"))
        fallback_samples = total_samples - real_id_samples

        logger.info(f"📊 Final statistics:")
        logger.info(f"📊 - Total samples processed: {total_samples}")
        logger.info(f"📊 - Real IDs: {real_id_samples} ({real_id_samples/total_samples*100:.1f}%)")
        logger.info(f"📊 - Fallback IDs: {fallback_samples} ({fallback_samples/total_samples*100:.1f}%)")

        # 显示真实ID示例
        real_ids = [k for k in uncertainty_scores.keys() if k.startswith("ID_")][:5]
        if real_ids:
            logger.info(f"📝 Real ID examples: {real_ids}")
        else:
            logger.warning("⚠️ No real IDs found! All samples using fallback IDs.")

        logger.info(f"📊 Successfully processed {len(uncertainty_scores)} samples")
        return uncertainty_scores


class SampleSelector(BaseActiveStep):
    """步骤3：样本选择"""

    def __init__(self, config: Dict[str, Any], state_path: Optional[str] = None):
        super().__init__(config, state_path)

        # 样本选择配置
        active_config = config.get("active_pseudo_learning", {})
        self.annotation_budget = active_config.get("annotation_budget", 50)

        selection_config = active_config.get("active_learning", {})
        self.strategies = selection_config.get(
            "strategies", {"uncertainty": 0.6, "diversity": 0.3, "cluster_based": 0.1}
        )

        logger.info(f"🎯 SampleSelector initialized")
        logger.info(f"📝 Budget: {self.annotation_budget}")
        logger.info(f"🎲 Strategies: {self.strategies}")

    def run(self) -> Dict[str, Any]:
        """运行样本选择"""
        logger.info("🎯 Starting sample selection...")

        # 1. 检查是否有不确定性分数
        if not self.state.uncertainty_scores:
            # 尝试从文件加载
            uncertainty_file = self.active_dir / f"uncertainty_scores_iter_{self.state.iteration}.json"
            if uncertainty_file.exists():
                with open(uncertainty_file, "r") as f:
                    self.state.uncertainty_scores = json.load(f)
                logger.info(f"📥 Loaded uncertainty scores from: {uncertainty_file}")
            else:
                raise ValueError("No uncertainty scores found. Please run uncertainty estimation first.")

        # 2. 选择样本
        selected_samples = self._select_samples()

        # 3. 更新状态
        self.state.selected_samples = selected_samples
        self.save_state()

        # 4. 生成标注请求文件
        annotation_file = self._generate_annotation_request(selected_samples)

        logger.info(f"✅ Sample selection completed")
        logger.info(f"📝 Selected {len(selected_samples)} samples for annotation")
        logger.info(f"📁 Annotation request saved to: {annotation_file}")

        return {
            "selected_samples": selected_samples,
            "annotation_file": str(annotation_file),
            "num_selected": len(selected_samples),
        }

    def _select_samples(self) -> List[str]:
        """选择样本进行标注"""
        uncertainty_scores = self.state.uncertainty_scores

        # 过滤已标注的样本
        available_samples = {
            sample_id: score
            for sample_id, score in uncertainty_scores.items()
            if sample_id not in self.state.labeled_samples
        }

        if len(available_samples) < self.annotation_budget:
            logger.warning(f"Available samples ({len(available_samples)}) < budget ({self.annotation_budget})")
            return list(available_samples.keys())

        # 基于不确定性选择
        uncertainty_weight = self.strategies.get("uncertainty", 1.0)
        if uncertainty_weight > 0:
            # 选择不确定性最高的样本
            sorted_samples = sorted(available_samples.items(), key=lambda x: x[1], reverse=True)
            n_uncertainty = int(self.annotation_budget * uncertainty_weight)
            selected_by_uncertainty = [sample_id for sample_id, _ in sorted_samples[:n_uncertainty]]
        else:
            selected_by_uncertainty = []

        # 随机选择剩余样本（简化的多样性策略）
        remaining_budget = self.annotation_budget - len(selected_by_uncertainty)
        if remaining_budget > 0:
            remaining_samples = [
                sample_id for sample_id in available_samples.keys() if sample_id not in selected_by_uncertainty
            ]

            if len(remaining_samples) <= remaining_budget:
                selected_random = remaining_samples
            else:
                import random

                random.seed(42)
                selected_random = random.sample(remaining_samples, remaining_budget)
        else:
            selected_random = []

        selected_samples = selected_by_uncertainty + selected_random

        logger.info(f"📊 Selection breakdown:")
        logger.info(f"  🎯 By uncertainty: {len(selected_by_uncertainty)}")
        logger.info(f"  🎲 Random/diversity: {len(selected_random)}")

        return selected_samples

    def _generate_annotation_request(self, selected_samples: List[str]) -> Path:
        """生成标注请求文件"""
        annotation_request = {
            "iteration": self.state.iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": self.state.experiment_name,
            "selected_samples": selected_samples,
            "annotation_budget": self.annotation_budget,
            "instructions": {
                "task": "请为选中的样本标注是否为滑坡区域",
                "labels": {"0": "非滑坡区域", "1": "滑坡区域"},
                "format": "请在annotation_results.json中提供标注结果",
            },
            "sample_details": [],
        }

        # 添加样本详细信息
        for sample_id in selected_samples:
            uncertainty = self.state.uncertainty_scores.get(sample_id, 0.0)
            sample_info = {
                "sample_id": sample_id,
                "uncertainty_score": uncertainty,
                "image_path": f"dataset/test_data/{sample_id}",  # 根据实际路径调整
                "label": None,  # 待标注
                "confidence": None,  # 可选：标注者信心
            }
            annotation_request["sample_details"].append(sample_info)

        # 保存标注请求
        request_file = self.active_dir / f"annotation_request_iter_{self.state.iteration}.json"
        with open(request_file, "w", encoding="utf-8") as f:
            json.dump(annotation_request, f, indent=2, ensure_ascii=False)

        return request_file


class ActiveRetrainer(BaseActiveStep):
    """步骤5：模型fine-tuning（基于已有检查点，不是从头训练）"""

    def __init__(self, config: Dict[str, Any], state_path: Optional[str] = None, annotation_file: Optional[str] = None):
        super().__init__(config, state_path)
        self.annotation_file = annotation_file

        # 重训练配置
        active_config = config.get("active_pseudo_learning", {})
        self.pseudo_config = active_config.get("pseudo_labeling", {})

        logger.info(f"🔄 ActiveRetrainer initialized (Fine-tuning mode)")
        logger.info(f"📥 Will load from checkpoint: {self.state.checkpoint_path}")
        logger.info(f"⚠️  Note: This will fine-tune existing model, NOT train from scratch")

    def run(self) -> Dict[str, Any]:
        """运行模型重训练（步骤5：基于已有检查点进行fine-tuning）"""
        logger.info("🔄 Starting model fine-tuning with new annotations and pseudo labels...")
        logger.info("⚠️  Note: This is fine-tuning from existing checkpoint, NOT training from scratch")

        # 1. 加载标注结果
        annotations = self._load_annotations()

        # 2. 加载伪标签（不再生成，而是从文件加载）
        pseudo_labels = self._generate_pseudo_labels()  # 方法名保持不变，但内部逻辑改为加载

        # 3. 更新训练数据
        updated_datamodule = self._update_training_data(annotations, pseudo_labels)

        # 4. Fine-tune模型（基于已有检查点）
        new_model, training_results = self._retrain_model(updated_datamodule)

        # 5. 更新状态
        self.state.annotation_history.append(
            {
                "iteration": self.state.iteration,
                "annotations": annotations,
                "num_pseudo_labels": len(pseudo_labels),
                "training_results": training_results,
                "fine_tuning": True,
                "pseudo_labels_source": "loaded_from_file",  # 标记来源
            }
        )

        # 更新迭代计数
        old_iteration = self.state.iteration
        self.state.iteration += 1
        self.save_state()

        logger.info(f"✅ Model fine-tuning completed (iteration {old_iteration} -> {self.state.iteration})")
        logger.info(f"📊 Used {len(annotations)} human annotations")
        logger.info(f"🏷️ Used {len(pseudo_labels)} pseudo labels (loaded from file)")

        return {
            "num_annotations": len(annotations),
            "num_pseudo_labels": len(pseudo_labels),
            "training_results": training_results,
            "new_checkpoint": training_results.get("best_checkpoint"),
            "iteration": self.state.iteration,
            "fine_tuning": True,
            "pseudo_labels_source": "loaded_from_file",
        }

    def _load_annotations(self) -> List[Dict]:
        """加载人工标注结果"""
        if self.annotation_file:
            annotation_path = Path(self.annotation_file)
        else:
            # 寻找最新的标注文件
            annotation_path = self.active_dir / f"annotation_results_iter_{self.state.iteration}.json"

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation_data = json.load(f)

        # 解析标注结果
        annotations = []

        if "annotations" in annotation_data:
            for sample_id, label in annotation_data["annotations"].items():
                if label is not None:  # 跳过 None 值（未标注的样本）
                    annotations.append(
                        {
                            "sample_id": sample_id,
                            "label": label,
                            "confidence": 0.9,  # 人工标注默认置信度为0.9
                        }
                    )
        else:
            raise ValueError(
                f"Unsupported annotation format in {annotation_path}. "
                f"Expected 'sample_details', 'annotations' dict, or direct list format."
            )

        logger.info(f"📥 Loaded {len(annotations)} annotations from: {annotation_path}")
        return annotations

    def _generate_pseudo_labels(self) -> List[Dict]:
        """
        生成伪标签 - 完整实现版本

        核心思路：
        1. 利用步骤2中的不确定性评估结果
        2. 选择确定性强的样本（低不确定性）
        3. 使用当前最佳模型进行推理
        4. 根据置信度阈值生成伪标签

        遵循三个原则：
        - 最小改动：重用现有的模型加载和数据处理逻辑
        - 单一职责：只负责伪标签生成
        - 渐进增强：在现有uncertainty_scores基础上实现
        """
        logger.info("🏷️ Loading pseudo labels from previous step...")

        # 1. 寻找伪标签文件
        pseudo_labels_file = self._find_pseudo_labels_file()

        if not pseudo_labels_file:
            logger.warning("⚠️ No pseudo labels file found, proceeding without pseudo labels")
            return []

        # 2. 加载伪标签数据
        try:
            with open(pseudo_labels_file, "r", encoding="utf-8") as f:
                pseudo_labels_data = json.load(f)

            # 提取伪标签列表
            if "pseudo_labels" in pseudo_labels_data:
                pseudo_labels = pseudo_labels_data["pseudo_labels"]
            else:
                # 兼容直接为列表格式的文件
                pseudo_labels = pseudo_labels_data if isinstance(pseudo_labels_data, list) else []

            logger.info(f"📥 Loaded {len(pseudo_labels)} pseudo labels from: {pseudo_labels_file}")

            # 3. 记录统计信息
            if "statistics" in pseudo_labels_data:
                stats = pseudo_labels_data["statistics"]
                logger.info(f"📊 Pseudo label statistics:")
                logger.info(f"  - Class distribution: {stats.get('class_distribution', {})}")
                logger.info(f"  - Average confidence: {stats.get('overall_avg_confidence', 0):.3f}")
                logger.info(f"  - Average uncertainty: {stats.get('overall_avg_uncertainty', 0):.3f}")

            return pseudo_labels

        except Exception as e:
            logger.error(f"❌ Error loading pseudo labels from {pseudo_labels_file}: {e}")
            logger.warning("⚠️ Proceeding without pseudo labels")
            return []

    def _find_pseudo_labels_file(self) -> Optional[Path]:
        """
        寻找伪标签文件

        搜索优先级：
        1. 当前迭代的伪标签文件
        2. 最新的伪标签文件
        3. 指定路径的伪标签文件（如果配置中有）
        """
        # 1. 优先查找当前迭代的文件
        current_iter_file = self.active_dir / f"pseudo_labels_iter_{self.state.iteration}.json"
        if current_iter_file.exists():
            logger.info(f"📁 Found current iteration pseudo labels: {current_iter_file}")
            return current_iter_file

        # 2. 查找最新的伪标签文件
        pseudo_files = list(self.active_dir.glob("pseudo_labels_iter_*.json"))
        if pseudo_files:
            # 按迭代号排序，取最新的
            latest_file = sorted(pseudo_files, key=lambda x: int(x.stem.split("_")[-1]))[-1]
            logger.info(f"📁 Found latest pseudo labels: {latest_file}")
            return latest_file

        # 3. 检查配置中是否指定了伪标签文件路径
        pseudo_file_path = self.pseudo_config.get("pseudo_labels_file")
        if pseudo_file_path:
            pseudo_file = Path(pseudo_file_path)
            if pseudo_file.exists():
                logger.info(f"📁 Found configured pseudo labels: {pseudo_file}")
                return pseudo_file
            else:
                logger.warning(f"⚠️ Configured pseudo labels file not found: {pseudo_file}")

        # 4. 未找到任何伪标签文件
        logger.warning("⚠️ No pseudo labels file found")
        logger.info("💡 Tip: Run 'python main.py pseudo_labeling config.yaml' first to generate pseudo labels")
        return None

    def _update_training_data(self, annotations: List[Dict], pseudo_labels: List[Dict]):
        """
        更新训练数据 - 正确版本，确保无数据泄露

        核心原则：
        1. 原验证集样本绝对不能进入训练集（数据泄露）
        2. 原训练集样本保持在训练集中
        3. 新标注样本可以按策略分配到训练集和验证集
        """
        logger.info("📊 Updating training data...")

        # 1. 读取原始分割文件（从dataset目录）

        original_train_split = Path("dataset/train_split.csv")
        original_val_split = Path("dataset/val_split.csv")

        if not original_train_split.exists() or not original_val_split.exists():
            raise FileNotFoundError(
                f"❌ Original split files not found! " f"Expected: {original_train_split} and {original_val_split}"
            )

        # 2. 加载原始分割
        original_train_df = pd.read_csv(original_train_split)
        original_val_df = pd.read_csv(original_val_split)

        logger.info(f"📂 Original train split: {len(original_train_df)} samples")
        logger.info(f"📂 Original val split: {len(original_val_df)} samples")

        # 4. 合并新样本（标注 + 伪标签）
        new_samples = []

        # 添加人工标注样本
        for ann in annotations:
            new_samples.append({"sample_id": ann["sample_id"], "label": ann["label"], "source": "annotation"})

        # 添加伪标签样本
        for pseudo in pseudo_labels:
            new_samples.append({"sample_id": pseudo["sample_id"], "label": pseudo["label"], "source": "pseudo_label"})

        logger.info(
            f"📝 Processing {len(new_samples)} new samples ({len(annotations)} annotations + {len(pseudo_labels)} pseudo labels)"
        )

        # 5. 分配新样本到训练集和验证集
        # 策略：按80/20比例分配新样本（可以调整）
        np.random.seed(self.config.get("seed", 3407))  # 确保可重现
        new_sample_indices = np.random.permutation(len(new_samples))

        # 计算分配数量
        val_ratio = 0.2  # 20%新样本进入验证集
        n_new_val = int(len(new_samples) * val_ratio)
        n_new_train = len(new_samples) - n_new_val

        new_val_indices = new_sample_indices[:n_new_val]
        new_train_indices = new_sample_indices[n_new_val:]

        logger.info(f"📊 New sample allocation: {n_new_train} → train, {n_new_val} → val")

        # 6. 构建增强的训练集和验证集
        enhanced_train_rows = []
        enhanced_val_rows = []

        # 保留所有原始训练样本
        for _, row in original_train_df.iterrows():
            enhanced_train_rows.append(row.to_dict())

        # 保留所有原始验证样本
        for _, row in original_val_df.iterrows():
            enhanced_val_rows.append(row.to_dict())

        # 添加新样本到训练集
        for idx in new_train_indices:
            sample = new_samples[idx]
            enhanced_train_rows.append({"ID": sample["sample_id"], "label": sample["label"]})

        # 添加新样本到验证集
        for idx in new_val_indices:
            sample = new_samples[idx]
            enhanced_val_rows.append({"ID": sample["sample_id"], "label": sample["label"]})

        # 7. 创建DataFrames
        enhanced_train_df = pd.DataFrame(enhanced_train_rows)
        enhanced_val_df = pd.DataFrame(enhanced_val_rows)

        # 8. 🔧 关键修复：创建包含所有数据的完整CSV文件
        all_enhanced_data = []
        all_enhanced_data.extend(enhanced_train_rows)
        all_enhanced_data.extend(enhanced_val_rows)
        all_enhanced_df = pd.DataFrame(all_enhanced_data)

        # 去重（防止重复）
        all_enhanced_df = all_enhanced_df.drop_duplicates(subset=["ID"], keep="first")

        # 10. 保存文件到active_learning目录
        # 🔧 关键修复：保存完整数据集作为训练CSV
        complete_enhanced_csv = self.active_dir / f"complete_enhanced_iter_{self.state.iteration}.csv"
        all_enhanced_df.to_csv(complete_enhanced_csv, index=False)

        # 保存分割文件
        active_train_split = self.active_dir / "train_split.csv"
        active_val_split = self.active_dir / "val_split.csv"

        enhanced_train_df.to_csv(active_train_split, index=False)
        enhanced_val_df.to_csv(active_val_split, index=False)

        # 11. 创建数据路径映射文件
        mapping_file = self._create_data_path_mapping(annotations, pseudo_labels)

        # 12. 🔧 关键修复：使用完整数据集CSV作为train_csv
        enhanced_config = self.config["data"].copy()
        enhanced_config["params"]["train_csv"] = str(complete_enhanced_csv)  # 完整数据集
        enhanced_config["params"]["cross_directory_mapping"] = str(mapping_file)

        # 创建数据模块
        datamodule = instantiate_from_config(enhanced_config)

        # 13. 统计信息
        logger.info(f"📊 Enhanced dataset statistics:")
        logger.info(f"  - Original train: {len(original_train_df)} samples")
        logger.info(f"  - Original val: {len(original_val_df)} samples")
        logger.info(f"  - New samples: {len(new_samples)} samples")
        logger.info(f"    └─ Added to train: {n_new_train} samples")
        logger.info(f"    └─ Added to val: {n_new_val} samples")
        logger.info(f"  - Enhanced train: {len(enhanced_train_df)} samples (+{n_new_train})")
        logger.info(f"  - Enhanced val: {len(enhanced_val_df)} samples (+{n_new_val})")
        logger.info(f"  - Complete dataset: {len(all_enhanced_df)} samples")

        return datamodule

    def _create_data_path_mapping(self, annotations: List[Dict], pseudo_labels: List[Dict]) -> Path:
        """创建数据路径映射文件，告诉数据加载器新样本的位置"""
        import json

        # 创建路径映射
        path_mapping = {}
        test_data_dir = Path(self.config["data"]["params"]["test_data_dir"])

        # 添加人工标注样本的路径映射
        for ann in annotations:
            sample_id = ann["sample_id"]
            source_file = test_data_dir / f"{sample_id}.npy"

            if source_file.exists():
                path_mapping[sample_id] = str(source_file)
                logger.info(f"  📍 Mapped {sample_id} -> {source_file}")
            else:
                logger.warning(f"  ⚠️ Source file not found for {sample_id}: {source_file}")

        # 添加伪标签样本的路径映射
        for pseudo in pseudo_labels:
            sample_id = pseudo["sample_id"]
            source_file = test_data_dir / f"{sample_id}.npy"

            if source_file.exists():
                path_mapping[sample_id] = str(source_file)
                logger.info(f"  📍 Mapped pseudo {sample_id} -> {source_file}")
            else:
                logger.warning(f"  ⚠️ Source file not found for pseudo {sample_id}: {source_file}")

        # 保存映射文件
        mapping_file = self.active_dir / f"data_path_mapping_iter_{self.state.iteration}.json"
        with open(mapping_file, "w") as f:
            json.dump(path_mapping, f, indent=2)

        logger.info(f"📁 Data path mapping saved: {mapping_file}")
        logger.info(f"📁 Successfully mapped {len(path_mapping)} samples from test directory")

        return mapping_file

    def _create_sample_link(self, sample_id: str, source_dir: Path, target_dir: Path):
        """为单个样本创建符号链接或复制文件"""
        source_file = source_dir / f"{sample_id}.npy"
        target_file = target_dir / f"{sample_id}.npy"

        if not source_file.exists():
            logger.warning(f"  Source file not found: {source_file}")
            return False

        if target_file.exists():
            logger.debug(f"  Target already exists: {target_file}")
            return True

        # 尝试创建符号链接（Linux/Mac）
        target_file.symlink_to(source_file)
        logger.debug(f"  Created symlink: {sample_id}")
        return True

    def _create_enhanced_training_csv(self, annotations: List[Dict], pseudo_labels: List[Dict]) -> Path:
        """创建包含新标注数据的增强训练CSV文件"""
        # 加载原始训练CSV
        original_csv = self.config["data"]["params"]["train_csv"]
        original_df = pd.read_csv(original_csv)

        logger.info(f"📊 Original training data: {len(original_df)} samples")

        # 准备新标注数据
        new_rows = []

        # 1. 添加人工标注数据
        for ann in annotations:
            sample_id = ann["sample_id"]
            label = ann["label"]

            new_row = {
                "ID": sample_id,  # 匹配CSV格式中的"ID"列
                "label": label,
            }
            new_rows.append(new_row)
            logger.debug(f"  Added annotation: {sample_id} -> {label}")

        # 2. 添加伪标签数据
        for pseudo in pseudo_labels:
            sample_id = pseudo["sample_id"]
            label = pseudo["label"]
            confidence = pseudo.get("confidence", 0.0)

            new_row = {
                "ID": sample_id,
                "label": label,
            }
            new_rows.append(new_row)
            logger.debug(f"  Added pseudo label: {sample_id} -> {label} (conf: {confidence:.3f})")

        # 3. 合并数据
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            enhanced_df = pd.concat([original_df, new_df], ignore_index=True)

            # 去重（以防重复添加）
            before_dedup = len(enhanced_df)
            enhanced_df = enhanced_df.drop_duplicates(subset=["ID"], keep="last")
            after_dedup = len(enhanced_df)

            if before_dedup != after_dedup:
                logger.info(f"📊 Removed {before_dedup - after_dedup} duplicate samples")
        else:
            enhanced_df = original_df.copy()
            logger.warning("⚠️ No new samples to add!")

        # 4. 保存增强的CSV文件
        enhanced_csv_path = self.active_dir / f"enhanced_train_iter_{self.state.iteration}.csv"
        enhanced_df.to_csv(enhanced_csv_path, index=False)

        # 5. 统计信息
        added_annotations = len([ann for ann in annotations])
        added_pseudo = len([pl for pl in pseudo_labels])

        logger.info(f"📊 Enhanced training data: {len(enhanced_df)} samples (+{len(new_rows)} new)")
        logger.info(f"  - Human annotations: {added_annotations}")
        logger.info(f"  - Pseudo labels: {added_pseudo}")

        # 6. 显示类别分布
        if "label" in enhanced_df.columns:
            class_counts = enhanced_df["label"].value_counts().sort_index()
            logger.info(f"📊 Class distribution: {dict(class_counts)}")

        return enhanced_csv_path

    def _retrain_model(self, datamodule) -> Tuple[pl.LightningModule, Dict[str, Any]]:
        """重训练模型（基于已有检查点进行fine-tuning）"""
        logger.info("🔄 Fine-tuning model from existing checkpoint...")

        # 重用现有的模型配置
        model = instantiate_from_config(self.config["model"])

        # 🔧 修复：手动加载检查点权重，而不是恢复完整训练状态
        logger.info(f"📥 Loading weights from checkpoint: {self.state.checkpoint_path}")
        checkpoint = torch.load(self.state.checkpoint_path, map_location="cpu")

        # 只加载模型权重，不恢复训练状态
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("✅ Successfully loaded model weights")
        else:
            logger.warning("⚠️ No 'state_dict' found in checkpoint, loading as direct state dict")
            model.load_state_dict(checkpoint)

        # 配置重训练的训练器（使用较少epoch）
        trainer_config = self.config.get("trainer", {}).get("params", {}).copy()
        trainer_config["max_epochs"] = 50  # 重训练使用较少epoch
        trainer_config["enable_model_summary"] = False  # 重训练时不显示模型摘要

        # 🔧 修复：先创建训练器（不包含callbacks）
        trainer = pl.Trainer(**trainer_config)

        # 🔧 修复：单独创建重训练专用的回调
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

        callbacks = [
            ModelCheckpoint(
                dirpath=str(self.active_dir / "retrain_checkpoints"),
                filename=f"retrain_iter_{self.state.iteration}_{{epoch:02d}}_{{val_f1:.4f}}",
                monitor="val_f1",
                mode="max",
                save_top_k=1,
                verbose=False,
            ),
            EarlyStopping(monitor="val_f1", patience=10, mode="max", verbose=False),  # 重训练时更快早停
        ]

        # 🔧 修复：直接将回调对象赋值给训练器实例，而不是配置对象
        trainer.callbacks = callbacks

        # 🔥 关键修复：不使用ckpt_path参数，从全新的训练状态开始（但使用预加载的权重）
        logger.info("🚀 Starting fine-tuning from epoch 0 (with pre-loaded weights)")
        trainer.fit(model, datamodule)  # 不传递ckpt_path，重新开始训练计数

        # 更新状态中的检查点路径为新的最佳模型
        new_checkpoint = callbacks[0].best_model_path
        if new_checkpoint:
            self.state.checkpoint_path = new_checkpoint
            logger.info(f"📤 Updated checkpoint path: {new_checkpoint}")

        # 返回训练结果
        training_results = {
            "best_checkpoint": new_checkpoint,
            "previous_checkpoint": self.state.checkpoint_path,
            "retrain_epochs": trainer_config["max_epochs"],
            "final_metrics": {},  # 可以添加最终验证指标
        }

        logger.info("✅ Model fine-tuning completed")
        return model, training_results


# =============================================================================
# 在 active_steps.py 中添加独立的伪标签生成类
# =============================================================================


class PseudoLabelGenerator(BaseActiveStep):
    """
    步骤3.5：伪标签生成（可选的独立步骤）

    设计思路：
    - 在uncertainty_estimation之后，sample_selection之前执行
    - 生成高质量的伪标签，为模型提供更多训练数据
    - 与主动学习流程完美集成
    """

    def __init__(self, config: Dict[str, Any], state_path: Optional[str] = None):
        super().__init__(config, state_path)

        # 伪标签配置
        self.pseudo_config = config.get("active_pseudo_learning", {}).get("pseudo_labeling", {})
        self.confidence_threshold = self.pseudo_config.get("confidence_threshold", 0.85)
        self.uncertainty_threshold = self.pseudo_config.get("uncertainty_threshold", 0.1)
        self.max_pseudo_samples = self.pseudo_config.get("max_pseudo_samples", 500)

        logger.info(f"🏷️ PseudoLabelGenerator initialized")
        logger.info(f"📊 Confidence threshold: {self.confidence_threshold}")
        logger.info(f"📊 Uncertainty threshold: {self.uncertainty_threshold}")
        logger.info(f"📊 Max pseudo samples: {self.max_pseudo_samples}")

    def run(self) -> Dict[str, Any]:
        """运行伪标签生成"""
        logger.info("🏷️ Starting pseudo label generation...")

        # 1. 检查前置条件
        if not self.state.uncertainty_scores:
            # 尝试从文件加载
            uncertainty_file = self.active_dir / f"uncertainty_scores_iter_{self.state.iteration}.json"
            if uncertainty_file.exists():
                with open(uncertainty_file, "r") as f:
                    self.state.uncertainty_scores = json.load(f)
                logger.info(f"📥 Loaded uncertainty scores from: {uncertainty_file}")
            else:
                raise ValueError("No uncertainty scores found. Please run uncertainty estimation first.")

        # 2. 生成伪标签
        pseudo_labels = self._generate_pseudo_labels()

        # 3. 更新状态
        self.state.pseudo_labels = pseudo_labels  # 新增状态字段
        self.save_state()

        # 4. 保存伪标签文件
        pseudo_labels_file = self._save_pseudo_labels(pseudo_labels)

        logger.info(f"✅ Pseudo label generation completed")
        logger.info(f"🏷️ Generated {len(pseudo_labels)} pseudo labels")
        logger.info(f"📁 Pseudo labels saved to: {pseudo_labels_file}")

        return {
            "pseudo_labels": pseudo_labels,
            "pseudo_labels_file": str(pseudo_labels_file),
            "num_pseudo_labels": len(pseudo_labels),
        }

    def _generate_pseudo_labels(self) -> List[Dict]:
        """
        生成伪标签 - 完整实现版本

        核心思路：
        1. 利用步骤2中的不确定性评估结果
        2. 选择确定性强的样本（低不确定性）
        3. 使用当前最佳模型进行推理
        4. 根据置信度阈值生成伪标签

        遵循三个原则：
        - 最小改动：重用现有的模型加载和数据处理逻辑
        - 单一职责：只负责伪标签生成
        - 渐进增强：在现有uncertainty_scores基础上实现
        """
        logger.info("🏷️ Generating pseudo labels...")

        # 1. 检查是否有不确定性分数
        if not self.state.uncertainty_scores:
            logger.warning("⚠️ No uncertainty scores found, skipping pseudo label generation")
            return []

        # 2. 筛选确定性强的样本（与选择的高不确定性样本互补）
        candidate_samples = self._select_high_confidence_samples()

        if not candidate_samples:
            logger.info("📊 No suitable samples for pseudo labeling")
            return []

        # 3. 加载模型进行推理
        model = self._load_model()
        datamodule = self._setup_datamodule()

        # 4. 对候选样本进行推理
        pseudo_labels = self._inference_pseudo_labels(model, datamodule, candidate_samples)

        logger.info(f"🏷️ Generated {len(pseudo_labels)} pseudo labels from {len(candidate_samples)} candidates")

        return pseudo_labels

    def _select_high_confidence_samples(self) -> List[str]:
        """
        选择高置信度样本作为伪标签候选

        策略：
        1. 选择不确定性低的样本（与主动学习选择的高不确定性样本形成互补）
        2. 排除已经被选中进行人工标注的样本
        3. 应用类别平衡策略
        """
        uncertainty_scores = self.state.uncertainty_scores
        uncertainty_threshold = self.pseudo_config.get("uncertainty_threshold", 0.1)

        # 🔧 修复：添加防护性检查，确保字段不为 None
        labeled_samples = self.state.labeled_samples if self.state.labeled_samples is not None else []
        selected_samples = self.state.selected_samples if self.state.selected_samples is not None else []

        # 排除已标注和已选择的样本
        excluded_samples = set(labeled_samples + selected_samples)

        # 筛选低不确定性样本
        candidate_samples = [
            sample_id
            for sample_id, uncertainty in uncertainty_scores.items()
            if uncertainty <= uncertainty_threshold and sample_id not in excluded_samples
        ]

        logger.info(
            f"📊 Found {len(candidate_samples)} low-uncertainty candidates (threshold: {uncertainty_threshold})"
        )
        logger.info(
            f"📊 Excluded {len(excluded_samples)} samples (labeled: {len(labeled_samples)}, selected: {len(selected_samples)})"
        )

        # 如果样本过多，按不确定性排序选择最确定的
        max_pseudo_samples = self.pseudo_config.get("max_pseudo_samples", 500)
        if len(candidate_samples) > max_pseudo_samples:
            # 按不确定性从低到高排序，选择最确定的样本
            sorted_candidates = sorted(candidate_samples, key=lambda x: uncertainty_scores[x])
            candidate_samples = sorted_candidates[:max_pseudo_samples]
            logger.info(f"📊 Limited to top {max_pseudo_samples} most certain samples")

        return candidate_samples

    def _load_model(self) -> pl.LightningModule:
        """加载模型 - 复用 UncertaintyEstimator 的实现"""
        logger.info(f"📥 Loading model from: {self.state.checkpoint_path}")

        # 重用现有的模型实例化逻辑
        model = instantiate_from_config(self.config["model"])

        # 检查GPU可用性并设置正确的设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 Using device: {device}")

        # 根据设备类型加载检查点
        if device == "cuda":
            checkpoint = torch.load(self.state.checkpoint_path, map_location="cuda")
        else:
            checkpoint = torch.load(self.state.checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["state_dict"])

        # 明确移动模型到GPU
        model = model.to(device)
        model.eval()

        return model

    def _setup_datamodule(self):
        """设置数据模块 - 复用 UncertaintyEstimator 的实现"""
        # 重用现有的数据模块
        datamodule = instantiate_from_config(self.config["data"])
        datamodule.setup("test")  # 使用测试集作为未标注池
        return datamodule

    def _inference_pseudo_labels(
        self, model: pl.LightningModule, datamodule, candidate_samples: List[str]
    ) -> List[Dict]:
        """
        对候选样本进行推理生成伪标签

        Args:
            model: 已加载的模型
            datamodule: 数据模块
            candidate_samples: 候选样本ID列表

        Returns:
            包含伪标签信息的字典列表
        """
        logger.info(f"🔮 Running inference on {len(candidate_samples)} candidate samples...")

        pseudo_labels = []
        confidence_threshold = self.pseudo_config.get("confidence_threshold", 0.85)
        device = next(model.parameters()).device

        # 创建候选样本的数据加载器
        candidate_dataloader = self._create_candidate_dataloader(datamodule, candidate_samples)

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(candidate_dataloader):
                # 🔧 修复：正确处理不同的batch格式
                try:
                    if isinstance(batch, dict):
                        # Dict格式：{key: tensor, ...}
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        images = batch.get("image") or batch.get("images") or batch.get("data")
                    elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                        # List/Tuple格式：[images, labels] 或 [images]
                        images = batch[0].to(device)
                        if len(batch) > 1:
                            # 如果有标签，也移动到设备上（虽然在推理中不需要）
                            labels = batch[1].to(device) if isinstance(batch[1], torch.Tensor) else batch[1]
                    else:
                        # 直接是tensor格式
                        images = batch.to(device)

                    if images is None:
                        logger.warning(f"⚠️ Could not extract images from batch at index {batch_idx}")
                        continue

                    # 🔧 验证图像tensor的格式
                    if len(images.shape) != 4:  # 应该是 [batch_size, channels, height, width]
                        logger.warning(f"⚠️ Unexpected image shape: {images.shape} at batch {batch_idx}")
                        continue

                    batch_size = images.shape[0]
                    logger.debug(f"🔄 Processing batch {batch_idx}: {batch_size} samples, shape: {images.shape}")

                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                    continue

                try:
                    # 模型推理
                    outputs = model(images)

                    # 🔧 修复：提取预测概率的逻辑
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    elif isinstance(outputs, dict) and "logits" in outputs:
                        logits = outputs["logits"]
                    elif isinstance(outputs, dict) and "predictions" in outputs:
                        logits = outputs["predictions"]
                    else:
                        # 直接是logits tensor
                        logits = outputs

                    # 🔧 处理不同的logits格式
                    if logits.dim() == 1:
                        # 二分类单输出：[batch_size] -> [batch_size, 2]
                        probabilities = torch.stack([1 - torch.sigmoid(logits), torch.sigmoid(logits)], dim=1)
                    elif logits.dim() == 2 and logits.shape[1] == 1:
                        # 二分类单输出：[batch_size, 1] -> [batch_size, 2]
                        sigmoid_probs = torch.sigmoid(logits.squeeze(1))
                        probabilities = torch.stack([1 - sigmoid_probs, sigmoid_probs], dim=1)
                    elif logits.dim() == 2 and logits.shape[1] == 2:
                        # 二分类双输出：[batch_size, 2]
                        probabilities = torch.softmax(logits, dim=1)
                    else:
                        logger.warning(f"⚠️ Unexpected logits shape: {logits.shape}")
                        continue

                    # 处理每个样本
                    for i in range(batch_size):
                        sample_idx = batch_idx * candidate_dataloader.batch_size + i
                        if sample_idx >= len(candidate_samples):
                            break

                        sample_id = candidate_samples[sample_idx]
                        prob = probabilities[i]

                        # 获取最高置信度的类别
                        max_prob, predicted_class = torch.max(prob, dim=0)
                        confidence = max_prob.item()

                        # 检查是否满足置信度阈值
                        if confidence >= confidence_threshold:
                            pseudo_labels.append(
                                {
                                    "sample_id": sample_id,
                                    "label": predicted_class.item(),
                                    "confidence": confidence,
                                    "uncertainty": self.state.uncertainty_scores.get(sample_id, 0.0),
                                    "source": "pseudo_label",
                                }
                            )

                            # 🔧 调试信息
                            if len(pseudo_labels) <= 5:  # 只显示前几个
                                logger.debug(
                                    f"  ✅ Added pseudo label: {sample_id} -> class {predicted_class.item()} (conf: {confidence:.3f})"
                                )

                except Exception as e:
                    logger.error(f"❌ Error during model inference for batch {batch_idx}: {e}")
                    continue

            # 应用类别平衡策略
            if self.pseudo_config.get("use_class_balance", True) and pseudo_labels:
                pseudo_labels = self._apply_class_balance(pseudo_labels)

            logger.info(f"✅ Generated {len(pseudo_labels)} high-confidence pseudo labels")
            logger.info(f"📊 Confidence threshold: {confidence_threshold}")

            return pseudo_labels

    def _create_candidate_dataloader(self, datamodule, candidate_samples: List[str]) -> DataLoader:
        """
        为候选样本创建数据加载器

        重用现有的数据处理逻辑，确保与训练流程一致
        """
        from torch.utils.data import Subset

        # 获取完整的测试数据集（作为未标注池）
        full_dataset = datamodule.test_dataset

        # 创建样本ID到索引的映射
        sample_to_idx = {}
        for idx in range(len(full_dataset)):
            sample_id = full_dataset.data_index.iloc[idx]["ID"]
            sample_to_idx[sample_id] = idx

        # 获取候选样本的索引
        candidate_indices = []
        for sample_id in candidate_samples:
            if sample_id in sample_to_idx:
                candidate_indices.append(sample_to_idx[sample_id])
            else:
                logger.warning(f"⚠️ Sample {sample_id} not found in dataset")

        if not candidate_indices:
            raise ValueError("No valid candidate samples found in dataset")

        # 创建子集
        candidate_subset = Subset(full_dataset, candidate_indices)

        # 创建数据加载器
        candidate_dataloader = DataLoader(
            candidate_subset,
            batch_size=self.config.get("data", {}).get("batch_size", 32),
            shuffle=False,  # 保持顺序以便匹配sample_id
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            pin_memory=True,
        )

        logger.info(f"📦 Created dataloader for {len(candidate_indices)} candidate samples")
        return candidate_dataloader

    def _apply_class_balance(self, pseudo_labels: List[Dict]) -> List[Dict]:
        """
        应用类别平衡策略

        确保伪标签在不同类别间相对平衡，避免模型偏向某一类别
        """
        if not pseudo_labels:
            return pseudo_labels

        # 统计各类别的伪标签数量
        class_counts = {}
        for pseudo in pseudo_labels:
            label = pseudo["label"]
            class_counts[label] = class_counts.get(label, 0) + 1

        logger.info(f"📊 Original pseudo label distribution: {class_counts}")

        # 计算平衡后的目标数量（取最小类别的数量）
        min_count = min(class_counts.values())
        max_pseudo_per_class = max(min_count, self.pseudo_config.get("min_pseudo_per_class", 10))

        # 按类别分组并限制数量
        balanced_pseudo_labels = []
        class_samples = {label: [] for label in class_counts.keys()}

        # 将伪标签按类别分组
        for pseudo in pseudo_labels:
            class_samples[pseudo["label"]].append(pseudo)

        # 每个类别选择最高置信度的样本
        for label, samples in class_samples.items():
            # 按置信度排序
            sorted_samples = sorted(samples, key=lambda x: x["confidence"], reverse=True)
            selected_samples = sorted_samples[:max_pseudo_per_class]
            balanced_pseudo_labels.extend(selected_samples)

        # 统计平衡后的分布
        final_counts = {}
        for pseudo in balanced_pseudo_labels:
            label = pseudo["label"]
            final_counts[label] = final_counts.get(label, 0) + 1

        logger.info(f"📊 Balanced pseudo label distribution: {final_counts}")

        return balanced_pseudo_labels

    def _save_pseudo_labels(self, pseudo_labels: List[Dict]) -> Path:
        """保存伪标签到文件"""
        pseudo_labels_file = self.active_dir / f"pseudo_labels_iter_{self.state.iteration}.json"

        # 创建详细的伪标签文件
        pseudo_labels_data = {
            "iteration": self.state.iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": self.state.experiment_name,
            "config": {
                "confidence_threshold": self.confidence_threshold,
                "uncertainty_threshold": self.uncertainty_threshold,
                "max_pseudo_samples": self.max_pseudo_samples,
            },
            "pseudo_labels": pseudo_labels,
            "statistics": self._compute_pseudo_label_stats(pseudo_labels),
        }

        with open(pseudo_labels_file, "w", encoding="utf-8") as f:
            json.dump(pseudo_labels_data, f, indent=2, ensure_ascii=False)

        return pseudo_labels_file

    def _compute_pseudo_label_stats(self, pseudo_labels: List[Dict]) -> Dict[str, Any]:
        """计算伪标签统计信息"""
        if not pseudo_labels:
            return {}

        # 类别分布
        class_counts = {}
        confidence_sum = {}
        uncertainty_sum = {}

        for pseudo in pseudo_labels:
            label = pseudo["label"]
            confidence = pseudo["confidence"]
            uncertainty = pseudo["uncertainty"]

            class_counts[label] = class_counts.get(label, 0) + 1
            confidence_sum[label] = confidence_sum.get(label, 0) + confidence
            uncertainty_sum[label] = uncertainty_sum.get(label, 0) + uncertainty

        # 计算平均值
        avg_confidence = {label: confidence_sum[label] / class_counts[label] for label in class_counts}
        avg_uncertainty = {label: uncertainty_sum[label] / class_counts[label] for label in class_counts}

        return {
            "class_distribution": class_counts,
            "average_confidence": avg_confidence,
            "average_uncertainty": avg_uncertainty,
            "total_samples": len(pseudo_labels),
            "overall_avg_confidence": sum(p["confidence"] for p in pseudo_labels) / len(pseudo_labels),
            "overall_avg_uncertainty": sum(p["uncertainty"] for p in pseudo_labels) / len(pseudo_labels),
        }


# =============================================================================
# 在 ActiveLearningStepManager 中添加新的方法
# =============================================================================


class ActiveLearningStepManager:
    """主动学习步骤管理器 - 添加伪标签生成支持"""

    @staticmethod
    def run_uncertainty_estimation(config: Dict[str, Any], state_path: Optional[str] = None) -> Dict[str, Any]:
        """运行不确定性估计步骤"""
        estimator = UncertaintyEstimator(config, state_path)
        return estimator.run()

    @staticmethod
    def run_pseudo_labeling(config: Dict[str, Any], state_path: Optional[str] = None) -> Dict[str, Any]:
        """运行伪标签生成步骤（新增）"""
        generator = PseudoLabelGenerator(config, state_path)
        return generator.run()

    @staticmethod
    def run_sample_selection(config: Dict[str, Any], state_path: Optional[str] = None) -> Dict[str, Any]:
        """运行样本选择步骤"""
        selector = SampleSelector(config, state_path)
        return selector.run()

    @staticmethod
    def run_retraining(
        config: Dict[str, Any], state_path: Optional[str] = None, annotation_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """运行模型重训练步骤"""
        retrainer = ActiveRetrainer(config, state_path, annotation_file)
        return retrainer.run()
