# =============================================================================
# lightning_landslide/src/active_learning/data_management.py
# =============================================================================

"""
数据管理和人工标注接口

这个模块提供了主动学习过程中复杂的数据状态管理功能：
1. 数据版本控制和追踪
2. 伪标签数据集的创建和管理
3. 人工标注接口的抽象和实现
4. 数据质量控制和验证
5. 增量数据集的高效管理
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import shutil
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """数据样本的统一表示"""

    sample_id: str
    file_path: str
    label: Optional[int] = None
    confidence: Optional[float] = None
    source: str = "original"  # "original", "pseudo", "annotated"
    iteration: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DatasetVersion:
    """数据集版本信息"""

    version_id: str
    iteration: int
    timestamp: str
    total_samples: int
    original_samples: int
    pseudo_samples: int
    annotated_samples: int
    class_distribution: Dict[int, int]
    description: str
    performance_snapshot: Dict[str, float] = None

    def __post_init__(self):
        if self.performance_snapshot is None:
            self.performance_snapshot = {}


class BaseAnnotationInterface(ABC):
    """
    人工标注接口基类

    这个抽象类定义了人工标注的标准接口，
    可以对接不同的标注工具或系统。
    """

    @abstractmethod
    def request_annotation(self, sample_ids: List[str], sample_paths: List[str]) -> Dict[str, int]:
        """
        请求人工标注

        Args:
            sample_ids: 样本ID列表
            sample_paths: 样本文件路径列表

        Returns:
            {sample_id: label} 的标注结果字典
        """
        pass

    @abstractmethod
    def get_annotation_status(self, request_id: str) -> str:
        """获取标注状态: 'pending', 'completed', 'failed'"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查标注服务是否可用"""
        pass


class SimulatedAnnotationInterface(BaseAnnotationInterface):
    """
    模拟标注接口

    用于测试和演示，通过预定义的真实标签来模拟人工标注过程。
    在实际部署中，应该替换为真实的标注接口。
    """

    def __init__(self, ground_truth_file: str, simulation_delay: float = 0.1):
        """
        初始化模拟标注接口

        Args:
            ground_truth_file: 真实标签文件路径
            simulation_delay: 模拟标注延迟（秒）
        """
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.simulation_delay = simulation_delay
        self.annotation_requests = {}

        logger.info(f"📝 Simulated annotation interface initialized with {len(self.ground_truth)} ground truth labels")

    def _load_ground_truth(self, file_path: str) -> Dict[str, int]:
        """加载真实标签"""
        try:
            if Path(file_path).suffix == ".csv":
                df = pd.read_csv(file_path)
                # 假设CSV格式为: ID, label
                return dict(zip(df["ID"].astype(str), df["label"]))
            elif Path(file_path).suffix == ".json":
                with open(file_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Unsupported ground truth file format: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return {}

    def request_annotation(self, sample_ids: List[str], sample_paths: List[str]) -> Dict[str, int]:
        """模拟标注过程"""
        import time

        time.sleep(self.simulation_delay)  # 模拟标注时间

        annotations = {}
        for sample_id in sample_ids:
            if sample_id in self.ground_truth:
                annotations[sample_id] = self.ground_truth[sample_id]
            else:
                # 如果没有真实标签，随机分配（仅用于演示）
                annotations[sample_id] = np.random.randint(0, 2)
                logger.warning(f"No ground truth for {sample_id}, using random label")

        logger.info(f"📝 Simulated annotation completed for {len(annotations)} samples")
        return annotations

    def get_annotation_status(self, request_id: str) -> str:
        return "completed"  # 模拟标注立即完成

    def is_available(self) -> bool:
        return True


class WebAnnotationInterface(BaseAnnotationInterface):
    """
    Web标注接口

    对接基于Web的标注平台，如LabelStudio、Prodigy等。
    这里提供了一个通用的框架。
    """

    def __init__(self, api_endpoint: str, api_key: str, project_id: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.project_id = project_id
        self.pending_requests = {}

    def request_annotation(self, sample_ids: List[str], sample_paths: List[str]) -> Dict[str, int]:
        """通过Web API请求标注"""
        # 这里应该实现实际的API调用
        logger.info(f"🌐 Requesting web annotation for {len(sample_ids)} samples")

        # 示例实现 - 实际中需要根据具体API修改
        # import requests
        #
        # payload = {
        #     "samples": [{"id": sid, "path": path} for sid, path in zip(sample_ids, sample_paths)],
        #     "project_id": self.project_id
        # }
        #
        # response = requests.post(
        #     f"{self.api_endpoint}/annotation/request",
        #     json=payload,
        #     headers={"Authorization": f"Bearer {self.api_key}"}
        # )
        #
        # return response.json()["annotations"]

        # 暂时返回空字典，实际实现时需要完善
        return {}

    def get_annotation_status(self, request_id: str) -> str:
        # 实际实现中查询API状态
        return "pending"

    def is_available(self) -> bool:
        # 实际实现中检查API连接
        return False


class EnhancedDataManager:
    """
    增强的数据管理器

    提供完整的数据版本控制、质量管理和增量更新功能。
    """

    def __init__(self, base_config: Dict, output_dir: Path, annotation_interface: BaseAnnotationInterface = None):
        """
        初始化数据管理器

        Args:
            base_config: 基础数据配置
            output_dir: 输出目录
            annotation_interface: 标注接口实例
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.annotation_interface = annotation_interface

        # 数据状态管理
        self.data_samples = {}  # {sample_id: DataSample}
        self.dataset_versions = []
        self.current_version = None

        # 目录结构
        self.data_dir = self.output_dir / "data_management"
        self.versions_dir = self.data_dir / "versions"
        self.annotations_dir = self.data_dir / "annotations"
        self.metadata_dir = self.data_dir / "metadata"

        self._setup_directories()
        self._initialize_original_data()

        logger.info("📊 Enhanced data manager initialized")

    def _setup_directories(self):
        """创建必要的目录结构"""
        for dir_path in [self.data_dir, self.versions_dir, self.annotations_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _initialize_original_data(self):
        """初始化原始数据"""
        logger.info("📂 Initializing original training data...")

        # 加载原始训练数据信息
        train_csv = Path(self.base_config["data"]["params"]["train_csv"])
        if train_csv.exists():
            df = pd.read_csv(train_csv)

            for _, row in df.iterrows():
                sample = DataSample(
                    sample_id=str(row["ID"]),
                    file_path=str(Path(self.base_config["data"]["params"]["train_data_dir"]) / f"{row['ID']}.npy"),
                    label=int(row["label"]) if "label" in row else None,
                    source="original",
                    iteration=0,
                )
                self.data_samples[sample.sample_id] = sample

        # 创建初始版本
        self._create_version_snapshot(0, "Initial training data")

        logger.info(f"✅ Loaded {len(self.data_samples)} original samples")

    def add_pseudo_labels(
        self, pseudo_samples: List[Dict], iteration: int, confidence_scores: Dict[str, float] = None
    ) -> int:
        """
        添加伪标签样本

        Args:
            pseudo_samples: 伪标签样本列表
            iteration: 当前迭代轮次
            confidence_scores: 置信度分数字典

        Returns:
            成功添加的样本数量
        """
        logger.info(f"🏷️ Adding {len(pseudo_samples)} pseudo-labeled samples for iteration {iteration}")

        added_count = 0
        confidence_scores = confidence_scores or {}

        for sample_info in pseudo_samples:
            sample_id = sample_info["sample_id"]
            predicted_label = sample_info["predicted_label"]

            # 检查样本是否已存在
            if sample_id in self.data_samples and self.data_samples[sample_id].source != "original":
                logger.warning(f"Sample {sample_id} already exists with source {self.data_samples[sample_id].source}")
                continue

            # 创建伪标签样本
            sample = DataSample(
                sample_id=sample_id,
                file_path=self._get_test_sample_path(sample_id),
                label=predicted_label,
                confidence=confidence_scores.get(sample_id, sample_info.get("confidence", 0.0)),
                source="pseudo",
                iteration=iteration,
                metadata={
                    "quality_score": sample_info.get("quality_score", 0.0),
                    "uncertainty": sample_info.get("uncertainty", 0.0),
                },
            )

            self.data_samples[sample_id] = sample
            added_count += 1

        logger.info(f"✅ Successfully added {added_count} pseudo-labeled samples")
        return added_count

    def request_annotations(self, sample_ids: List[str], iteration: int, timeout: float = 300.0) -> Dict[str, int]:
        """
        请求人工标注

        Args:
            sample_ids: 需要标注的样本ID列表
            iteration: 当前迭代轮次
            timeout: 超时时间（秒）

        Returns:
            标注结果字典
        """
        if not self.annotation_interface or not self.annotation_interface.is_available():
            logger.warning("No annotation interface available, skipping annotation request")
            return {}

        logger.info(f"📝 Requesting annotations for {len(sample_ids)} samples")

        # 准备样本路径
        sample_paths = [self._get_test_sample_path(sid) for sid in sample_ids]

        # 发起标注请求
        try:
            annotations = self.annotation_interface.request_annotation(sample_ids, sample_paths)

            # 保存标注结果
            self._save_annotation_results(annotations, iteration)

            # 更新数据样本
            self._update_samples_with_annotations(annotations, iteration)

            logger.info(f"✅ Received annotations for {len(annotations)} samples")
            return annotations

        except Exception as e:
            logger.error(f"❌ Annotation request failed: {e}")
            return {}

    def _update_samples_with_annotations(self, annotations: Dict[str, int], iteration: int):
        """用标注结果更新样本"""
        for sample_id, label in annotations.items():
            if sample_id in self.data_samples:
                # 更新现有样本
                sample = self.data_samples[sample_id]
                sample.label = label
                sample.source = "annotated"
                sample.iteration = iteration
            else:
                # 创建新的标注样本
                sample = DataSample(
                    sample_id=sample_id,
                    file_path=self._get_test_sample_path(sample_id),
                    label=label,
                    source="annotated",
                    iteration=iteration,
                )
                self.data_samples[sample_id] = sample

    def create_combined_dataset(self, iteration: int) -> "CombinedDataset":
        """
        创建合并的数据集

        Args:
            iteration: 当前迭代轮次

        Returns:
            包含所有数据的组合数据集
        """
        logger.info(f"🔗 Creating combined dataset for iteration {iteration}")

        # 筛选有效样本（有标签的样本）
        valid_samples = [sample for sample in self.data_samples.values() if sample.label is not None]

        # 按来源分类
        by_source = defaultdict(list)
        for sample in valid_samples:
            by_source[sample.source].append(sample)

        logger.info(f"📊 Dataset composition:")
        for source, samples in by_source.items():
            logger.info(f"  {source}: {len(samples)} samples")

        # 创建版本快照
        self._create_version_snapshot(iteration, f"Combined dataset for iteration {iteration}")

        return CombinedDataset(valid_samples, self.base_config)

    def _create_version_snapshot(self, iteration: int, description: str):
        """创建数据版本快照"""
        version_id = f"v{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 统计数据
        by_source = defaultdict(int)
        class_dist = defaultdict(int)

        for sample in self.data_samples.values():
            if sample.label is not None:
                by_source[sample.source] += 1
                class_dist[sample.label] += 1

        version = DatasetVersion(
            version_id=version_id,
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            total_samples=sum(by_source.values()),
            original_samples=by_source["original"],
            pseudo_samples=by_source["pseudo"],
            annotated_samples=by_source["annotated"],
            class_distribution=dict(class_dist),
            description=description,
        )

        self.dataset_versions.append(version)
        self.current_version = version

        # 保存版本信息
        version_file = self.versions_dir / f"{version_id}.json"
        with open(version_file, "w") as f:
            json.dump(asdict(version), f, indent=2)

        # 保存样本详情
        samples_file = self.versions_dir / f"{version_id}_samples.pkl"
        with open(samples_file, "wb") as f:
            pickle.dump(dict(self.data_samples), f)

        logger.info(f"📸 Created version snapshot: {version_id}")

    def _get_test_sample_path(self, sample_id: str) -> str:
        """获取测试样本的文件路径"""
        test_data_dir = Path(self.base_config["data"]["params"]["test_data_dir"])
        return str(test_data_dir / f"{sample_id}.npy")

    def _save_annotation_results(self, annotations: Dict[str, int], iteration: int):
        """保存标注结果"""
        annotation_file = self.annotations_dir / f"iteration_{iteration}_annotations.json"

        annotation_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "annotations": annotations,
            "count": len(annotations),
        }

        with open(annotation_file, "w") as f:
            json.dump(annotation_data, f, indent=2)

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        by_source = defaultdict(int)
        by_iteration = defaultdict(int)
        class_dist = defaultdict(int)

        for sample in self.data_samples.values():
            if sample.label is not None:
                by_source[sample.source] += 1
                by_iteration[sample.iteration] += 1
                class_dist[sample.label] += 1

        return {
            "total_samples": len([s for s in self.data_samples.values() if s.label is not None]),
            "by_source": dict(by_source),
            "by_iteration": dict(by_iteration),
            "class_distribution": dict(class_dist),
            "versions_count": len(self.dataset_versions),
            "current_version": self.current_version.version_id if self.current_version else None,
        }

    def export_training_data(self, output_path: Path, iteration: int = None):
        """
        导出训练数据为标准格式

        Args:
            output_path: 输出路径
            iteration: 指定迭代轮次，None表示导出所有数据
        """
        logger.info(f"📤 Exporting training data to {output_path}")

        # 筛选数据
        if iteration is not None:
            samples = [s for s in self.data_samples.values() if s.label is not None and s.iteration <= iteration]
        else:
            samples = [s for s in self.data_samples.values() if s.label is not None]

        # 创建DataFrame
        data_records = []
        for sample in samples:
            record = {
                "ID": sample.sample_id,
                "label": sample.label,
                "source": sample.source,
                "iteration": sample.iteration,
                "confidence": sample.confidence or 0.0,
                "file_path": sample.file_path,
            }
            if sample.metadata:
                record.update(sample.metadata)
            data_records.append(record)

        df = pd.DataFrame(data_records)
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Exported {len(samples)} samples to {output_path}")


class CombinedDataset(data.Dataset):
    """
    组合数据集类

    将原始数据、伪标签数据和新标注数据合并为一个统一的数据集。
    """

    def __init__(self, samples: List[DataSample], base_config: Dict):
        """
        初始化组合数据集

        Args:
            samples: 数据样本列表
            base_config: 基础配置
        """
        self.samples = samples
        self.base_config = base_config

        # 创建索引映射
        self.sample_ids = [s.sample_id for s in samples]
        self.labels = [s.label for s in samples]

        # 数据变换（复用原有的变换逻辑）
        self.transform = self._create_transform()

        logger.info(f"🔗 CombinedDataset created with {len(samples)} samples")
        self._log_composition()

    def _create_transform(self):
        """创建数据变换"""
        # 这里应该复用原有的数据变换逻辑
        # 简化实现，实际中需要根据具体的变换配置
        return None

    def _log_composition(self):
        """记录数据集组成"""
        by_source = defaultdict(int)
        by_class = defaultdict(int)

        for sample in self.samples:
            by_source[sample.source] += 1
            by_class[sample.label] += 1

        logger.info("📊 Dataset composition:")
        for source, count in by_source.items():
            logger.info(f"  {source}: {count} samples")

        logger.info("📊 Class distribution:")
        for cls, count in by_class.items():
            logger.info(f"  class {cls}: {count} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]

        try:
            # 加载数据文件
            if Path(sample.file_path).exists():
                data = np.load(sample.file_path)
                data = torch.from_numpy(data).float()
            else:
                logger.warning(f"File not found: {sample.file_path}, using dummy data")
                data = torch.zeros((5, 64, 64))  # 默认形状

            # 标签
            label = torch.tensor(sample.label, dtype=torch.long)

            # 应用变换
            if self.transform:
                data = self.transform(data)

            return data, label

        except Exception as e:
            logger.error(f"Error loading sample {sample.sample_id}: {e}")
            # 返回dummy数据
            return torch.zeros((5, 64, 64)), torch.tensor(0, dtype=torch.long)


def create_annotation_interface(interface_type: str, **kwargs) -> BaseAnnotationInterface:
    """
    工厂函数：创建标注接口

    Args:
        interface_type: 接口类型 ("simulated", "web", "manual")
        **kwargs: 接口特定参数

    Returns:
        标注接口实例
    """
    interfaces = {
        "simulated": SimulatedAnnotationInterface,
        "web": WebAnnotationInterface,
    }

    if interface_type not in interfaces:
        raise ValueError(f"Unknown annotation interface type: {interface_type}")

    return interfaces[interface_type](**kwargs)


def create_enhanced_data_manager(
    base_config: Dict, output_dir: Path, annotation_config: Dict = None
) -> EnhancedDataManager:
    """
    工厂函数：创建增强数据管理器

    Args:
        base_config: 基础数据配置
        output_dir: 输出目录
        annotation_config: 标注接口配置

    Returns:
        数据管理器实例
    """
    annotation_interface = None

    if annotation_config:
        annotation_interface = create_annotation_interface(
            interface_type=annotation_config.get("type", "simulated"), **annotation_config.get("params", {})
        )

    return EnhancedDataManager(
        base_config=base_config, output_dir=output_dir, annotation_interface=annotation_interface
    )
