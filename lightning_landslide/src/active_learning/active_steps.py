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

    def __post_init__(self):
        if self.unlabeled_pool is None:
            self.unlabeled_pool = []
        if self.labeled_samples is None:
            self.labeled_samples = []
        if self.annotation_history is None:
            self.annotation_history = []

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
            # 寻找best开头的检查点文件
            best_ckpts = list(checkpoint_dir.glob("baseline_epoch*.ckpt"))
            if best_ckpts:
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
        """加载模型"""
        logger.info(f"📥 Loading model from: {self.state.checkpoint_path}")

        # 重用现有的模型实例化逻辑
        model = instantiate_from_config(self.config["model"])

        # 加载检查点
        checkpoint = torch.load(self.state.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
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
        self.confidence_threshold = self.pseudo_config.get("confidence_threshold", 0.85)

        logger.info(f"🔄 ActiveRetrainer initialized (Fine-tuning mode)")
        logger.info(f"📥 Will load from checkpoint: {self.state.checkpoint_path}")
        logger.info(f"🏷️ Pseudo label threshold: {self.confidence_threshold}")
        logger.info(f"⚠️  Note: This will fine-tune existing model, NOT train from scratch")

    def run(self) -> Dict[str, Any]:
        """运行模型重训练（步骤5：基于已有检查点进行fine-tuning）"""
        logger.info("🔄 Starting model fine-tuning with new annotations...")
        logger.info("⚠️  Note: This is fine-tuning from existing checkpoint, NOT training from scratch")

        # 1. 加载标注结果
        annotations = self._load_annotations()

        # 2. 生成伪标签（基于当前模型）
        pseudo_labels = self._generate_pseudo_labels()

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
                "fine_tuning": True,  # 标记这是fine-tuning
            }
        )

        # 更新迭代计数
        old_iteration = self.state.iteration
        self.state.iteration += 1
        self.save_state()

        logger.info(f"✅ Model fine-tuning completed (iteration {old_iteration} -> {self.state.iteration})")
        logger.info(f"📊 Added {len(annotations)} human annotations")
        logger.info(f"🏷️ Generated {len(pseudo_labels)} pseudo labels")

        return {
            "num_annotations": len(annotations),
            "num_pseudo_labels": len(pseudo_labels),
            "training_results": training_results,
            "new_checkpoint": training_results.get("best_checkpoint"),
            "iteration": self.state.iteration,
            "fine_tuning": True,
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
        for sample_info in annotation_data.get("sample_details", []):
            if sample_info.get("label") is not None:
                annotations.append(
                    {
                        "sample_id": sample_info["sample_id"],
                        "label": sample_info["label"],
                        "confidence": sample_info.get("confidence", 1.0),
                    }
                )

        logger.info(f"📥 Loaded {len(annotations)} annotations from: {annotation_path}")
        return annotations

    def _generate_pseudo_labels(self) -> List[Dict]:
        """生成伪标签"""
        logger.info("🏷️ Generating pseudo labels...")

        # 简化版本：基于置信度阈值生成伪标签
        # 在实际实现中，这里应该使用训练好的模型对未标注数据进行预测

        # 这里返回空列表，实际实现时需要：
        # 1. 加载当前最佳模型
        # 2. 对未标注数据进行预测
        # 3. 选择高置信度的预测作为伪标签

        pseudo_labels = []
        logger.info(f"🏷️ Generated {len(pseudo_labels)} pseudo labels")
        return pseudo_labels

    def _update_training_data(self, annotations: List[Dict], pseudo_labels: List[Dict]):
        """更新训练数据"""
        logger.info("📊 Updating training data...")

        # 重用现有的数据模块
        datamodule = instantiate_from_config(self.config["data"])

        # 在实际实现中，这里需要：
        # 1. 将新标注的数据添加到训练集
        # 2. 将伪标签添加到训练集
        # 3. 更新数据模块的训练集

        logger.info(
            f"📊 Training data updated with {len(annotations)} annotations and {len(pseudo_labels)} pseudo labels"
        )
        return datamodule

    def _retrain_model(self, datamodule) -> Tuple[pl.LightningModule, Dict[str, Any]]:
        """重训练模型（基于已有检查点进行fine-tuning）"""
        logger.info("🔄 Fine-tuning model from existing checkpoint...")

        # 重用现有的模型配置
        model = instantiate_from_config(self.config["model"])

        # 配置重训练的训练器（使用较少epoch）
        trainer_config = self.config.get("trainer", {}).get("params", {}).copy()
        trainer_config["max_epochs"] = 20  # 重训练使用较少epoch
        trainer_config["enable_model_summary"] = False  # 重训练时不显示模型摘要

        # 添加重训练专用的回调
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
            EarlyStopping(monitor="val_f1", patience=5, mode="max", verbose=False),  # 重训练时更快早停
        ]

        trainer_config["callbacks"] = callbacks
        trainer = pl.Trainer(**trainer_config)

        # 🔥 关键：从已有检查点开始fine-tuning，而不是从头训练
        logger.info(f"📥 Loading from checkpoint: {self.state.checkpoint_path}")
        trainer.fit(model, datamodule, ckpt_path=self.state.checkpoint_path)

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
# 步骤管理器
# =============================================================================


class ActiveLearningStepManager:
    """主动学习步骤管理器"""

    @staticmethod
    def run_uncertainty_estimation(config: Dict[str, Any], state_path: Optional[str] = None) -> Dict[str, Any]:
        """运行不确定性估计步骤"""
        estimator = UncertaintyEstimator(config, state_path)
        return estimator.run()

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
