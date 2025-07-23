import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    所有数据模块的抽象基类

    这个基类标准化了数据处理的完整流程。就像工厂的流水线一样，
    每个环节都有标准的操作程序，这样无论处理什么类型的数据，
    都能保证质量和效率。

    Lightning数据处理的生命周期：
    1. prepare_data(): 数据下载和预处理（只在主进程执行一次）
    2. setup(): 创建数据集对象（每个进程都执行）
    3. train/val/test_dataloader(): 创建数据加载器

    为什么要这样分阶段？
    - prepare_data()只在主进程执行，避免多进程重复下载
    - setup()在每个进程执行，确保每个GPU都有自己的数据集对象
    - dataloader()方法按需创建，支持动态配置
    """

    def __init__(self, cfg: DictConfig):
        """
        初始化数据模块

        Args:
            cfg: 完整的配置对象，包含数据相关的所有参数
        """
        super().__init__()
        self.cfg = cfg

        # 提取通用数据配置
        # 这些是所有数据模块都需要的基本参数
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.get("num_workers", 4)
        self.pin_memory = cfg.data.get("pin_memory", True)
        self.shuffle_train = cfg.data.get("shuffle_train", True)

        # 数据路径
        self.data_dir = Path(cfg.data.data_dir)

        # 数据集对象（在setup方法中初始化）
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 数据统计信息（用于调试和监控）
        self._data_stats = {}

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Batch size: {self.batch_size}, Workers: {self.num_workers}")

    @abstractmethod
    def prepare_data(self) -> None:
        """
        数据准备阶段（全局执行一次）

        这个方法的设计理念是"准备数据，但不创建数据集对象"。

        典型任务：
        - 下载原始数据（如果还没下载）
        - 验证数据文件完整性
        - 执行一次性的数据预处理（如格式转换）
        - 创建必要的目录结构

        重要提醒：
        1. 这个方法只在主进程中执行一次
        2. 不要在这里访问self.trainer或self.model
        3. 不要创建数据集对象或进行随机操作
        4. 应该是幂等的（多次执行结果相同）

        为什么这样设计？
        在多GPU训练时，我们不希望每个进程都重复下载数据，
        这会造成资源浪费和潜在的文件冲突。
        """
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """
        设置数据集（每个进程执行）

        这是数据处理的核心阶段，在这里我们创建实际的数据集对象。

        参数stage的含义：
        - 'fit': 训练和验证阶段
        - 'test': 测试阶段
        - 'predict': 预测阶段
        - None: 所有阶段

        典型任务：
        - 创建训练、验证、测试数据集
        - 应用数据变换和增强
        - 进行数据集划分
        - 加载预处理的数据文件

        设计要点：
        这个方法在每个GPU进程中都会执行，所以：
        1. 可以访问self.trainer和self.model
        2. 可以进行随机操作（每个进程有独立随机状态）
        3. 应该根据stage参数只创建需要的数据集

        Args:
            stage: 训练阶段标识符
        """
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """
        创建训练数据加载器

        这个方法返回用于训练的DataLoader。设计时需要考虑：

        1. 数据顺序：训练时通常需要shuffle=True
        2. 数据增强：训练时应用数据增强
        3. 批次大小：根据GPU内存和训练策略调整
        4. 工作进程：合理设置num_workers避免性能瓶颈

        Returns:
            配置好的训练数据加载器
        """
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """
        创建验证数据加载器

        验证数据加载器的配置通常与训练不同：

        1. 数据顺序：通常不需要shuffle
        2. 数据增强：通常只应用基本的标准化
        3. 批次大小：可以使用更大的批次（无需梯度）

        Returns:
            配置好的验证数据加载器
        """
        pass

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        创建测试数据加载器（可选实现）

        不是所有项目都有独立的测试集，所以这个方法是可选的。

        Returns:
            测试数据加载器，如果没有测试集则返回None
        """
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,  # 测试时不丢弃最后一个不完整批次
            )
        return None

    def predict_dataloader(self) -> Optional[DataLoader]:
        """
        创建预测数据加载器（可选实现）

        预测阶段通常使用测试集，但可能有不同的配置需求。

        Returns:
            预测数据加载器
        """
        return self.test_dataloader()

    # 以下是辅助方法，提供常用的数据处理功能

    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        这个方法提供数据集的基本统计信息，有助于：
        1. 调试数据加载问题
        2. 监控数据质量
        3. 优化训练配置
        4. 生成实验报告

        Returns:
            包含数据统计信息的字典
        """
        info = {
            "data_module": self.__class__.__name__,
            "data_dir": str(self.data_dir),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

        # 添加数据集大小信息
        if self.train_dataset is not None:
            info["train_size"] = len(self.train_dataset)
        if self.val_dataset is not None:
            info["val_size"] = len(self.val_dataset)
        if self.test_dataset is not None:
            info["test_size"] = len(self.test_dataset)

        # 添加自定义统计信息
        info.update(self._data_stats)

        return info

    def validate_data_paths(self, required_paths: List[str]) -> None:
        """
        验证必需的数据文件是否存在

        这是一个实用的辅助方法，帮助在训练开始前就发现数据问题。

        Args:
            required_paths: 必需文件的路径列表

        Raises:
            FileNotFoundError: 如果必需文件不存在
        """
        missing_paths = []

        for path_str in required_paths:
            path = Path(path_str)
            if not path.exists():
                missing_paths.append(path_str)

        if missing_paths:
            error_msg = f"Missing required data files: {missing_paths}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("All required data files validated successfully")

    def compute_class_weights(self, dataset: Dataset) -> Dict[int, float]:
        """
        计算类别权重（处理类别不平衡）

        这个方法对于您的滑坡检测任务特别有用，因为滑坡样本
        通常比非滑坡样本少得多。

        Args:
            dataset: 数据集对象

        Returns:
            类别权重字典 {class_id: weight}
        """
        if not hasattr(dataset, "__getitem__") or not hasattr(dataset, "__len__"):
            logger.warning("Dataset does not support class weight computation")
            return {}

        # 统计各类别样本数量
        class_counts = {}
        for i in range(len(dataset)):
            try:
                _, label = dataset[i]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_counts[label] = class_counts.get(label, 0) + 1
            except Exception as e:
                logger.warning(f"Error accessing sample {i}: {e}")
                break

        if not class_counts:
            return {}

        # 计算权重（反比于频率）
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)

        class_weights = {}
        for class_id, count in class_counts.items():
            weight = total_samples / (num_classes * count)
            class_weights[class_id] = weight

        logger.info(f"Computed class weights: {class_weights}")
        return class_weights

    def get_sample_for_debug(self, split: str = "train", index: int = 0) -> Tuple[Any, Any]:
        """
        获取调试样本

        这个方法用于快速检查数据加载是否正常，特别有用于：
        1. 验证数据预处理管道
        2. 检查数据格式和范围
        3. 调试数据增强效果

        Args:
            split: 数据集分割类型 ('train', 'val', 'test')
            index: 样本索引

        Returns:
            (data, label) 元组
        """
        dataset = None
        if split == "train" and self.train_dataset is not None:
            dataset = self.train_dataset
        elif split == "val" and self.val_dataset is not None:
            dataset = self.val_dataset
        elif split == "test" and self.test_dataset is not None:
            dataset = self.test_dataset

        if dataset is None:
            raise ValueError(f"Dataset for split '{split}' is not available")

        if index >= len(dataset):
            raise ValueError(f"Index {index} out of range for dataset size {len(dataset)}")

        return dataset[index]

    def summary(self) -> None:
        """打印数据模块摘要"""
        info = self.get_data_info()
        print("\n" + "=" * 50)
        print(f"Data Module: {info['data_module']}")
        print("=" * 50)
        print(f"Data directory: {info['data_dir']}")
        print(f"Batch size: {info['batch_size']}")
        print(f"Number of workers: {info['num_workers']}")

        if "train_size" in info:
            print(f"Training samples: {info['train_size']:,}")
        if "val_size" in info:
            print(f"Validation samples: {info['val_size']:,}")
        if "test_size" in info:
            print(f"Test samples: {info['test_size']:,}")

        print("=" * 50 + "\n")


import torch
