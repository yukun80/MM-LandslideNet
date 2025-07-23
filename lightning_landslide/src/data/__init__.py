# =============================================================================
# lightning_landslide/src/data/__init__.py - 数据模块导入文件
# =============================================================================

"""
数据模块统一导入接口

这个文件就像是数据模块的"目录"，它让我们可以方便地导入
所有需要的数据相关类。通过统一的导入接口，我们的代码
变得更加简洁和易于维护。

教学要点：
__init__.py文件在Python包管理中起着重要作用。它不仅让
目录成为Python包，还可以定义包的公共接口，控制外部
代码可以访问哪些组件。
"""

# 导入多模态数据模块（如果存在）
try:
    from .multimodal_dataset import MultiModalDataset

    print("✅ MultiModalDataset imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import MultiModalDataset: {e}")
    MultiModalDataset = None

# 导入虚拟数据模块（用于测试）
try:
    from .dummy_data_module import DummyDataModule, DummyLandslideDataset

    print("✅ DummyDataModule imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import DummyDataModule: {e}")
    DummyDataModule = None
    DummyLandslideDataset = None

# # 导入数据集类（如果存在）
# try:
#     from .landslide_dataset import LandslideDataset

#     print("✅ LandslideDataset imported successfully")
# except ImportError as e:
#     print(f"⚠️  Could not import LandslideDataset: {e}")
#     LandslideDataset = None

# # 导入数据处理工具
# try:
#     from .data_utils import *

#     print("✅ Data utilities imported successfully")
# except ImportError as e:
#     print(f"⚠️  Could not import data utilities: {e}")


# 定义公共接口
__all__ = [
    # 核心数据模块
    "MultiModalDataset",
    # 数据集类
    "DummyLandslideDataset",
    # 工具函数（通过data_utils导入）
]

# 移除None值，只导出真正可用的组件
__all__ = [name for name in __all__ if globals().get(name) is not None]

# 版本信息
__version__ = "2.0.0"
__author__ = "MM-LandslideNet Team"

print(f"📦 Data module initialized (v{__version__})")
print(f"📋 Available components: {', '.join(__all__)}")
