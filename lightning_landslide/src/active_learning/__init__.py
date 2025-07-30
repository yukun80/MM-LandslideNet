# =============================================================================
# lightning_landslide/src/active_learning/__init__.py
# =============================================================================

"""
MM-LandslideNet 主动学习模块

这个模块提供了完整的主动学习+伪标签融合功能，包括：
- 不确定性估计
- 伪标签生成
- 主动学习样本选择
- 数据管理和版本控制
- 可视化分析
- 完整的训练流程

主要组件：
- UncertaintyEstimator: 多种不确定性估计方法
- PseudoLabelGenerator: 智能伪标签生成
- ActiveLearningSelector: 主动学习样本选择
- ActivePseudoTrainer: 融合训练器
- ActiveKFoldTrainer: K折+主动学习训练器
- EnhancedDataManager: 数据管理器
- ActiveLearningVisualizer: 可视化工具
"""

__version__ = "1.0.0"
__author__ = "MM-LandslideNet Team"

import logging
from typing import Dict, Any, Optional
import warnings

# 设置日志
logger = logging.getLogger(__name__)

# 核心组件导入
try:
    from .uncertainty_estimator import (
        BaseUncertaintyEstimator,
        MCDropoutEstimator,
        DeepEnsembleEstimator,
        HybridUncertaintyEstimator,
        TemperatureScaling,
        UncertaintyResults,
        create_uncertainty_estimator,
    )

    logger.debug("✓ Uncertainty estimation components loaded")
except ImportError as e:
    logger.error(f"Failed to import uncertainty estimation components: {e}")
    raise

try:
    from .pseudo_label_generator import (
        PseudoLabelGenerator,
        PseudoLabelSample,
        PseudoLabelResults,
        AdaptiveThresholdScheduler,
        ClassBalanceController,
        create_pseudo_label_generator,
    )

    logger.debug("✓ Pseudo labeling components loaded")
except ImportError as e:
    logger.error(f"Failed to import pseudo labeling components: {e}")
    raise

try:
    from .active_learning_selector import (
        BaseActiveLearningStrategy,
        UncertaintyStrategy,
        DiversityStrategy,
        ClusterBasedStrategy,
        QueryByCommitteeStrategy,
        HybridActiveLearningSelector,
        ActiveLearningQuery,
        ActiveLearningResults,
        create_active_learning_selector,
    )

    logger.debug("✓ Active learning selection components loaded")
except ImportError as e:
    logger.error(f"Failed to import active learning selection components: {e}")
    raise

try:
    from .active_pseudo_trainer import (
        ActivePseudoTrainer,
        ActivePseudoTrainingResults,
        IterationResults,
        create_active_pseudo_trainer,
    )

    logger.debug("✓ Active pseudo training components loaded")
except ImportError as e:
    logger.error(f"Failed to import active pseudo training components: {e}")
    raise

try:
    from .data_management import (
        EnhancedDataManager,
        DataSample,
        DatasetVersion,
        CombinedDataset,
        BaseAnnotationInterface,
        SimulatedAnnotationInterface,
        WebAnnotationInterface,
        create_annotation_interface,
        create_enhanced_data_manager,
    )

    logger.debug("✓ Data management components loaded")
except ImportError as e:
    logger.error(f"Failed to import data management components: {e}")
    raise

try:
    from .visualization import ActiveLearningVisualizer, VisualizationConfig, create_visualizer

    logger.debug("✓ Visualization components loaded")
except ImportError as e:
    logger.error(f"Failed to import visualization components: {e}")
    raise

# 可选组件导入（K折训练器）
try:
    from ..training.active_kfold_trainer import ActiveKFoldTrainer, ActiveKFoldResults, create_active_kfold_trainer

    logger.debug("✓ Active K-fold training components loaded")
except ImportError as e:
    logger.warning(f"Active K-fold trainer not available: {e}")
    ActiveKFoldTrainer = None
    ActiveKFoldResults = None
    create_active_kfold_trainer = None


# 公共接口
__all__ = [
    # 核心组件
    "ActivePseudoTrainer",
    "create_active_pseudo_trainer",
    # 不确定性估计
    "BaseUncertaintyEstimator",
    "MCDropoutEstimator",
    "DeepEnsembleEstimator",
    "HybridUncertaintyEstimator",
    "TemperatureScaling",
    "UncertaintyResults",
    "create_uncertainty_estimator",
    # 伪标签生成
    "PseudoLabelGenerator",
    "PseudoLabelSample",
    "PseudoLabelResults",
    "AdaptiveThresholdScheduler",
    "ClassBalanceController",
    "create_pseudo_label_generator",
    # 主动学习选择
    "BaseActiveLearningStrategy",
    "UncertaintyStrategy",
    "DiversityStrategy",
    "ClusterBasedStrategy",
    "QueryByCommitteeStrategy",
    "HybridActiveLearningSelector",
    "ActiveLearningQuery",
    "ActiveLearningResults",
    "create_active_learning_selector",
    # 数据管理
    "EnhancedDataManager",
    "DataSample",
    "DatasetVersion",
    "CombinedDataset",
    "BaseAnnotationInterface",
    "SimulatedAnnotationInterface",
    "WebAnnotationInterface",
    "create_annotation_interface",
    "create_enhanced_data_manager",
    # 可视化
    "ActiveLearningVisualizer",
    "VisualizationConfig",
    "create_visualizer",
    # 结果类型
    "ActivePseudoTrainingResults",
    "IterationResults",
]

# 添加K折相关组件（如果可用）
if ActiveKFoldTrainer is not None:
    __all__.extend(["ActiveKFoldTrainer", "ActiveKFoldResults", "create_active_kfold_trainer"])


class ActiveLearningError(Exception):
    """主动学习模块的基础异常类"""

    pass


def create_active_learning_pipeline(
    config: Dict[str, Any], experiment_name: str = None, output_dir: str = None, enable_kfold: bool = False
) -> Any:
    """
    便捷函数：创建完整的主动学习流水线

    Args:
        config: 完整配置字典
        experiment_name: 实验名称
        output_dir: 输出目录
        enable_kfold: 是否启用K折交叉验证

    Returns:
        训练器实例

    Raises:
        ActiveLearningError: 配置错误或组件创建失败
    """
    try:
        if enable_kfold and ActiveKFoldTrainer is not None:
            logger.info("🔄 Creating Active K-fold training pipeline")
            return create_active_kfold_trainer(config=config, experiment_name=experiment_name, output_dir=output_dir)
        else:
            logger.info("🎯 Creating Active pseudo-label training pipeline")
            return create_active_pseudo_trainer(config=config, experiment_name=experiment_name, output_dir=output_dir)

    except Exception as e:
        raise ActiveLearningError(f"Failed to create active learning pipeline: {e}") from e


def validate_active_learning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证主动学习配置的有效性

    Args:
        config: 配置字典

    Returns:
        验证和修正后的配置

    Raises:
        ActiveLearningError: 配置验证失败
    """
    logger.info("🔍 Validating active learning configuration...")

    # 深拷贝配置以避免修改原始配置
    import copy

    validated_config = copy.deepcopy(config)

    # 检查必需的顶级配置节
    required_sections = ["model", "data", "trainer"]
    for section in required_sections:
        if section not in validated_config:
            raise ActiveLearningError(f"Missing required configuration section: {section}")

    # 检查主动学习配置
    if "active_pseudo_learning" not in validated_config:
        logger.warning("No active_pseudo_learning section found, using defaults")
        validated_config["active_pseudo_learning"] = {}

    apl_config = validated_config["active_pseudo_learning"]

    # 验证和设置默认值
    default_apl_config = {
        "max_iterations": 5,
        "convergence_threshold": 0.01,
        "min_improvement_iterations": 2,
        "annotation_budget": 50,
        "uncertainty_estimation": {
            "method": "mc_dropout",
            "params": {"n_forward_passes": 30, "use_temperature_scaling": True},
        },
        "pseudo_labeling": {
            "confidence_threshold": 0.9,
            "uncertainty_threshold": 0.1,
            "use_adaptive_threshold": True,
            "use_class_balance": True,
        },
        "active_learning": {
            "budget_per_iteration": 50,
            "strategies": {"uncertainty": 0.5, "diversity": 0.3, "cluster_based": 0.2},
        },
    }

    # 递归合并默认配置
    def merge_configs(default: Dict, user: Dict) -> Dict:
        for key, value in default.items():
            if key not in user:
                user[key] = value
            elif isinstance(value, dict) and isinstance(user[key], dict):
                merge_configs(value, user[key])
        return user

    apl_config = merge_configs(default_apl_config, apl_config)
    validated_config["active_pseudo_learning"] = apl_config

    # 验证参数范围
    validations = [
        ("max_iterations", 1, 20, "Maximum iterations must be between 1 and 20"),
        ("convergence_threshold", 0.001, 0.1, "Convergence threshold must be between 0.001 and 0.1"),
        ("annotation_budget", 1, 1000, "Annotation budget must be between 1 and 1000"),
    ]

    for param, min_val, max_val, error_msg in validations:
        if param in apl_config:
            value = apl_config[param]
            if not (min_val <= value <= max_val):
                raise ActiveLearningError(f"{error_msg}, got {value}")

    # 验证伪标签配置
    pseudo_config = apl_config["pseudo_labeling"]
    confidence_threshold = pseudo_config["confidence_threshold"]
    uncertainty_threshold = pseudo_config["uncertainty_threshold"]

    if not (0.5 <= confidence_threshold <= 0.99):
        raise ActiveLearningError(f"Confidence threshold must be between 0.5 and 0.99, got {confidence_threshold}")

    if not (0.01 <= uncertainty_threshold <= 0.5):
        raise ActiveLearningError(f"Uncertainty threshold must be between 0.01 and 0.5, got {uncertainty_threshold}")

    # 验证策略权重
    strategies = apl_config["active_learning"]["strategies"]
    total_weight = sum(strategies.values())
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"Strategy weights sum to {total_weight:.3f}, normalizing to 1.0")
        for strategy in strategies:
            strategies[strategy] /= total_weight

    logger.info("✅ Configuration validation completed")
    return validated_config


def get_active_learning_info() -> Dict[str, Any]:
    """
    获取主动学习模块信息

    Returns:
        模块信息字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "components": {
            "uncertainty_estimation": ["mc_dropout", "deep_ensemble", "hybrid"],
            "pseudo_labeling": ["adaptive_threshold", "class_balance", "quality_control"],
            "active_learning": ["uncertainty", "diversity", "cluster_based", "query_by_committee"],
            "data_management": ["version_control", "annotation_interface", "combined_dataset"],
            "visualization": ["static_plots", "interactive_dashboard", "comparison_reports"],
            "training": ["active_pseudo", "active_kfold"],
        },
        "supported_tasks": ["active_train", "active_kfold"],
        "requirements": {
            "pytorch": ">=1.9.0",
            "pytorch_lightning": ">=1.5.0",
            "numpy": ">=1.20.0",
            "pandas": ">=1.3.0",
            "scikit-learn": ">=0.24.0",
            "matplotlib": ">=3.3.0",
            "seaborn": ">=0.11.0",
            "plotly": ">=5.0.0",
        },
    }


# 模块级别的便捷函数
def quick_start_active_learning(
    train_data_dir: str,
    test_data_dir: str,
    train_csv: str,
    test_csv: str,
    model_name: str = "swin_tiny_patch4_window7_224",
    experiment_name: str = None,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """
    快速启动主动学习的便捷函数

    Args:
        train_data_dir: 训练数据目录
        test_data_dir: 测试数据目录
        train_csv: 训练标签文件
        test_csv: 测试标签文件
        model_name: 模型名称
        experiment_name: 实验名称
        max_iterations: 最大迭代次数

    Returns:
        训练结果字典
    """
    logger.info("🚀 Quick start active learning...")

    # 创建默认配置
    config = {
        "experiment_name": experiment_name or f"quick_active_{int(__import__('time').time())}",
        "seed": 3407,
        "log_level": "INFO",
        "model": {
            "target": "lightning_landslide.src.models.LandslideClassificationModule",
            "params": {
                "base_model": {
                    "target": "lightning_landslide.src.models.optical_swin.OpticalSwinModel",
                    "params": {"model_name": model_name, "input_channels": 5, "num_classes": 1},
                }
            },
        },
        "data": {
            "target": "lightning_landslide.src.data.MultiModalDataModule",
            "params": {
                "train_data_dir": train_data_dir,
                "test_data_dir": test_data_dir,
                "train_csv": train_csv,
                "test_csv": test_csv,
                "batch_size": 32,
                "num_workers": 4,
            },
        },
        "trainer": {
            "target": "pytorch_lightning.Trainer",
            "params": {"max_epochs": 30, "accelerator": "auto", "devices": "auto"},
        },
        "active_pseudo_learning": {"max_iterations": max_iterations, "annotation_budget": 50},
        "outputs": {"base_output_dir": "outputs"},
    }

    # 验证配置
    config = validate_active_learning_config(config)

    # 创建训练器
    trainer = create_active_learning_pipeline(config)

    # 运行训练
    results = trainer.run()

    logger.info("✅ Quick start active learning completed")
    return results.to_dict() if hasattr(results, "to_dict") else results


# 错误处理装饰器
def handle_active_learning_errors(func):
    """装饰器：统一处理主动学习相关错误"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ActiveLearningError:
            # 重新抛出主动学习相关错误
            raise
        except Exception as e:
            # 包装其他异常
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise ActiveLearningError(f"Unexpected error in {func.__name__}: {e}") from e

    return wrapper


# 模块初始化时的检查
def _check_dependencies():
    """检查必要的依赖"""
    try:
        import torch
        import pytorch_lightning as pl
        import numpy as np
        import pandas as pd
        import sklearn
        import matplotlib
        import seaborn as sns

        logger.debug("✓ All required dependencies available")
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        raise ImportError(f"Missing required dependency for active learning module: {e}")


def _suppress_warnings():
    """抑制不必要的警告"""
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib")


# 模块初始化
try:
    _check_dependencies()
    _suppress_warnings()
    logger.info(f"🎯 MM-LandslideNet Active Learning Module v{__version__} loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize active learning module: {e}")
    raise
