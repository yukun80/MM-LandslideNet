# =============================================================================
# lightning_landslide/src/models/__init__.py - 模型模块导入配置
# =============================================================================

"""
模型模块统一导入接口

这个文件定义了整个models包的公共接口。通过这里的导入，
其他模块可以方便地访问所有可用的模型类。

设计原则：
1. 统一接口：所有模型都通过相同的方式导入
2. 易于扩展：添加新模型只需要在这里添加一行导入
3. 清晰命名：导出的名称应该清晰表达模型的用途

可用模型：
- LandslideClassificationModule: PyTorch Lightning训练模块
- OpticalSwinModel: 基于Swin Transformer的光学数据模型
- OpticalConvNextModel: 基于ConvNextv2的光学数据模型
- BaseModel: 所有模型的抽象基类
"""

# 核心训练模块
from .classification_module import LandslideClassificationModule

# 基础模型类
from .base import BaseModel

# 具体模型实现
from .optical_swin import OpticalSwinModel
from .optical_convnext import OpticalConvNextModel  # 新增：ConvNextv2模型

# 便捷创建函数
from .optical_swin import (
    create_swin_tiny,
    create_swin_small,
    create_swin_base,
)

from .optical_convnext import (  # 新增：ConvNextv2便捷函数
    create_convnext_tiny,
    create_convnext_small,
    create_convnext_base,
)

# 公共接口定义
__all__ = [
    # 核心类
    "LandslideClassificationModule",
    "BaseModel",
    # 模型实现
    "OpticalSwinModel",
    "OpticalConvNextModel",  # 新增
    # 便捷创建函数 - Swin Transformer系列
    "create_swin_tiny",
    "create_swin_small",
    "create_swin_base",
    # 便捷创建函数 - ConvNextv2系列
    "create_convnext_tiny",  # 新增
    "create_convnext_small",  # 新增
    "create_convnext_base",  # 新增
]

# 模型注册表 - 用于配置文件中的动态创建
MODEL_REGISTRY = {
    "optical_swin": OpticalSwinModel,
    "optical_convnext": OpticalConvNextModel,  # 新增
    "swin_transformer": OpticalSwinModel,  # 别名
    "convnextv2": OpticalConvNextModel,  # 别名
}

# 版本信息
__version__ = "1.1.0"  # 更新版本号以反映新增的ConvNextv2支持


def get_available_models():
    """
    获取所有可用的模型列表

    Returns:
        Dict[str, BaseModel]: 模型名称到模型类的映射
    """
    return MODEL_REGISTRY.copy()


def create_model_from_name(model_name: str, **kwargs) -> BaseModel:
    """
    根据模型名称创建模型实例

    这是一个便捷函数，可以根据字符串名称直接创建模型，
    而不需要使用完整的配置文件。主要用于快速实验和测试。

    Args:
        model_name: 模型名称，如 'optical_swin', 'optical_convnext'
        **kwargs: 传递给模型构造函数的参数

    Returns:
        创建的模型实例

    Example:
        >>> model = create_model_from_name('optical_convnext', input_channels=5)
        >>> print(model.get_feature_dim())
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model name: {model_name}. Available: {available}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def list_model_info():
    """
    打印所有可用模型的信息

    这个函数用于查看当前框架支持的所有模型，
    以及它们的基本特征。
    """
    print("\n" + "=" * 60)
    print("Lightning Landslide - Available Models")
    print("=" * 60)

    for name, model_class in MODEL_REGISTRY.items():
        print(f"\n📦 {name}:")
        print(f"   Class: {model_class.__name__}")
        print(f"   Module: {model_class.__module__}")

        # 尝试获取模型的文档字符串第一行作为描述
        doc = model_class.__doc__
        if doc:
            description = doc.strip().split("\n")[0]
            print(f"   Description: {description}")

    print("\n" + "=" * 60)
    print(f"Total models available: {len(MODEL_REGISTRY)}")
    print("=" * 60 + "\n")


# 导入时的自检
def _validate_imports():
    """验证所有导入是否正常"""
    try:
        # 验证核心类可以正常导入
        assert LandslideClassificationModule is not None
        assert BaseModel is not None
        assert OpticalSwinModel is not None
        assert OpticalConvNextModel is not None  # 新增验证

        # 验证便捷函数可以正常导入
        assert create_swin_tiny is not None
        assert create_convnext_tiny is not None  # 新增验证

        print("✓ All model imports validated successfully")

    except ImportError as e:
        print(f"✗ Model import validation failed: {e}")
        raise
    except AssertionError as e:
        print(f"✗ Model import assertion failed: {e}")
        raise


# 在调试模式下运行验证
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Testing models package...")
    _validate_imports()

    print("\nListing available models:")
    list_model_info()

    print("\nTesting model creation:")
    try:
        # 测试通过注册表创建模型
        swin_model = create_model_from_name("optical_swin", input_channels=5)
        convnext_model = create_model_from_name("optical_convnext", input_channels=5)

        print(f"✓ Swin model feature dim: {swin_model.get_feature_dim()}")
        print(f"✓ ConvNext model feature dim: {convnext_model.get_feature_dim()}")

        print("\n🎉 All tests passed! Models package is ready to use.")

    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        raise
