# =============================================================================
# lightning_landslide/src/active_learning/utils.py
# =============================================================================

"""
主动学习工具函数

提供主动学习过程中常用的工具函数和辅助类。
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
from functools import wraps
import psutil
import gc

logger = logging.getLogger(__name__)


class Timer:
    """简单的计时器类"""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"⏱️ {self.name} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"⏱️ {self.name} completed in {duration:.2f}s")

    def elapsed(self) -> float:
        """获取已经过的时间"""
        if self.start_time is None:
            return 0.0
        current_time = time.time()
        return current_time - self.start_time


class MemoryMonitor:
    """内存使用监控器"""

    def __init__(self, name: str = "MemoryMonitor"):
        self.name = name
        self.initial_memory = None

    def __enter__(self):
        self.initial_memory = self._get_memory_usage()
        logger.debug(f"🧠 {self.name} - Initial memory: {self.initial_memory:.1f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_memory = self._get_memory_usage()
        memory_diff = current_memory - self.initial_memory
        logger.info(f"🧠 {self.name} - Memory change: {memory_diff:+.1f} MB (Current: {current_memory:.1f} MB)")

    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def current_usage(self) -> float:
        """获取当前内存使用量"""
        return self._get_memory_usage()


def memory_cleanup():
    """清理内存"""
    logger.debug("🧹 Performing memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def timing_decorator(func):
    """计时装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(f"{func.__name__}"):
            return func(*args, **kwargs)

    return wrapper


def memory_monitor_decorator(func):
    """内存监控装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryMonitor(f"{func.__name__}"):
            result = func(*args, **kwargs)
            memory_cleanup()
            return result

    return wrapper


def safe_json_serialize(obj: Any) -> str:
    """安全的JSON序列化"""

    def json_serializer(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.integer, np.floating)):
            return o.item()
        elif isinstance(o, Path):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        elif hasattr(o, "__dict__"):
            return o.__dict__
        else:
            return str(o)

    try:
        return json.dumps(obj, default=json_serializer, indent=2)
    except Exception as e:
        logger.warning(f"JSON serialization failed: {e}")
        return str(obj)


def calculate_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    计算类别权重以处理不平衡数据

    Args:
        labels: 标签列表

    Returns:
        类别权重字典
    """
    from collections import Counter
    import numpy as np

    label_counts = Counter(labels)
    total_samples = len(labels)
    n_classes = len(label_counts)

    # 计算平衡权重
    weights = {}
    for label, count in label_counts.items():
        weight = total_samples / (n_classes * count)
        weights[label] = weight

    logger.info(f"📊 Calculated class weights: {weights}")
    return weights


def stratified_split_indices(
    labels: List[int], test_size: float = 0.2, random_state: int = None
) -> Tuple[List[int], List[int]]:
    """
    分层抽样分割数据索引

    Args:
        labels: 标签列表
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        (训练索引, 测试索引) 元组
    """
    from sklearn.model_selection import train_test_split

    indices = list(range(len(labels)))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=random_state
    )

    return train_indices, test_indices


def compute_confidence_intervals(scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    计算置信区间

    Args:
        scores: 分数列表
        confidence: 置信水平

    Returns:
        (均值, 下界, 上界) 元组
    """
    import scipy.stats as stats

    scores = np.array(scores)
    mean = np.mean(scores)
    sem = stats.sem(scores)  # 标准误差

    # 计算置信区间
    alpha = 1 - confidence
    df = len(scores) - 1
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    margin_error = t_critical * sem
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    return mean, lower_bound, upper_bound


def normalize_uncertainty_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    归一化不确定性分数

    Args:
        scores: 不确定性分数数组
        method: 归一化方法 ("minmax", "zscore", "robust")

    Returns:
        归一化后的分数
    """
    if method == "minmax":
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.zeros_like(scores)

    elif method == "zscore":
        mean, std = scores.mean(), scores.std()
        if std > 0:
            return (scores - mean) / std
        else:
            return np.zeros_like(scores)

    elif method == "robust":
        q25, q75 = np.percentile(scores, [25, 75])
        median = np.median(scores)
        iqr = q75 - q25
        if iqr > 0:
            return (scores - median) / iqr
        else:
            return np.zeros_like(scores)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_experiment_summary(results: Dict[str, Any], output_path: Path) -> str:
    """
    创建实验摘要

    Args:
        results: 实验结果字典
        output_path: 输出路径

    Returns:
        摘要文件路径
    """
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": results.get("experiment_name", "unknown"),
            "version": results.get("version", "unknown"),
        },
        "performance_summary": {
            "best_performance": max(results.get("performance_history", {}).get("val_f1", [0])),
            "final_performance": (
                results.get("performance_history", {}).get("val_f1", [0])[-1]
                if results.get("performance_history", {}).get("val_f1")
                else 0
            ),
            "convergence_iteration": results.get("convergence_iteration", 0),
            "total_training_time": results.get("total_training_time", 0),
        },
        "data_summary": {
            "initial_samples": (
                results.get("data_usage_history", {}).get("training_samples", [0])[0]
                if results.get("data_usage_history", {}).get("training_samples")
                else 0
            ),
            "final_samples": (
                results.get("data_usage_history", {}).get("training_samples", [0])[-1]
                if results.get("data_usage_history", {}).get("training_samples")
                else 0
            ),
            "pseudo_labels_used": (
                results.get("data_usage_history", {}).get("pseudo_labels", [0])[-1]
                if results.get("data_usage_history", {}).get("pseudo_labels")
                else 0
            ),
            "new_annotations": (
                results.get("data_usage_history", {}).get("new_annotations", [0])[-1]
                if results.get("data_usage_history", {}).get("new_annotations")
                else 0
            ),
        },
    }

    # 保存摘要
    summary_path = output_path / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"📋 Experiment summary saved: {summary_path}")
    return str(summary_path)


def validate_data_paths(config: Dict[str, Any]) -> bool:
    """
    验证数据路径的有效性

    Args:
        config: 配置字典

    Returns:
        验证是否通过
    """
    data_config = config.get("data", {}).get("params", {})

    required_paths = [
        ("train_data_dir", "训练数据目录"),
        ("test_data_dir", "测试数据目录"),
        ("train_csv", "训练标签文件"),
        ("test_csv", "测试标签文件"),
    ]

    all_valid = True

    for path_key, description in required_paths:
        if path_key not in data_config:
            logger.error(f"❌ Missing {description} in configuration")
            all_valid = False
            continue

        path = Path(data_config[path_key])
        if not path.exists():
            logger.error(f"❌ {description} not found: {path}")
            all_valid = False
        else:
            logger.debug(f"✅ {description} found: {path}")

    return all_valid


def estimate_training_time(config: Dict[str, Any], sample_size: int = 100) -> float:
    """
    估算训练时间

    Args:
        config: 配置字典
        sample_size: 估算样本大小

    Returns:
        预估训练时间（秒）
    """
    # 简单的时间估算公式
    base_time_per_epoch = 60  # 秒
    max_epochs = config.get("trainer", {}).get("params", {}).get("max_epochs", 30)
    max_iterations = config.get("active_pseudo_learning", {}).get("max_iterations", 5)

    # 考虑主动学习的额外开销
    active_learning_overhead = 1.5  # 50%的额外开销

    estimated_time = base_time_per_epoch * max_epochs * max_iterations * active_learning_overhead

    logger.info(f"⏰ Estimated training time: {estimated_time/60:.1f} minutes")
    return estimated_time


def create_debug_checkpoint(data: Dict[str, Any], checkpoint_dir: Path, name: str = "debug") -> str:
    """
    创建调试检查点

    Args:
        data: 要保存的数据
        checkpoint_dir: 检查点目录
        name: 检查点名称

    Returns:
        检查点文件路径
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{name}_checkpoint_{timestamp}.pkl"

    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"🐛 Debug checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    except Exception as e:
        logger.error(f"Failed to save debug checkpoint: {e}")
        return ""


def load_debug_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    加载调试检查点

    Args:
        checkpoint_path: 检查点文件路径

    Returns:
        加载的数据
    """
    try:
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)

        logger.info(f"🐛 Debug checkpoint loaded: {checkpoint_path}")
        return data

    except Exception as e:
        logger.error(f"Failed to load debug checkpoint: {e}")
        return {}
