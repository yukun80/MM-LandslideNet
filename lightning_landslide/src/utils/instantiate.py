# =============================================================================
# lightning_landslide/src/utils/instantiate.py - 通用对象实例化工具
# =============================================================================

"""
配置驱动的对象实例化工具

这个模块提供了类似latent-diffusion的instantiate_from_config机制，
让我们可以通过配置文件动态创建任何Python对象。

核心思想：
"配置即代码" - 通过YAML配置文件就能完全定义对象的创建过程，
无需修改任何Python代码。这让我们的框架具备了极高的灵活性。

教学要点：
这个机制的威力在于它打破了硬编码的局限。想象一下，如果每次
添加新模型都要修改工厂类，代码会变得越来越复杂。而有了这个
工具，添加新模型只需要写配置文件即可。
"""

import importlib
import logging
from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_obj_from_str(string: str, reload: bool = False) -> Any:
    """
    从字符串路径导入Python对象

    这个函数是整个机制的核心。它可以将类似 'torch.nn.Linear'
    这样的字符串转换为实际的Python类或函数。

    Args:
        string: 对象的完整路径，如 'lightning_landslide.src.models.OpticalSwinModel'
        reload: 是否重新加载模块（开发时有用）

    Returns:
        导入的Python对象（类、函数等）

    Example:
        >>> cls = get_obj_from_str('torch.nn.Linear')
        >>> layer = cls(10, 5)  # 创建一个Linear层
    """
    try:
        # 分割模块路径和对象名
        # 例如：'torch.nn.Linear' -> module='torch.nn', cls='Linear'
        module, cls = string.rsplit(".", 1)

        if reload:
            # 重新加载模块（通常用于开发调试）
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)

        # 导入模块并获取对象
        return getattr(importlib.import_module(module, package=None), cls)

    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to import {string}: {e}")
        raise ImportError(f"Cannot import {string}") from e


def instantiate_from_config(config: Union[Dict, DictConfig], **kwargs) -> Any:
    """
    从配置实例化对象

    这是我们的"万能工厂"函数。它读取配置文件中的 'target' 字段，
    导入对应的类，然后用 'params' 字段中的参数创建实例。

    配置文件格式：
    ```yaml
    target: lightning_landslide.src.models.OpticalSwinModel  # 类的完整路径
    params:                                                   # 构造函数参数
      model_name: swin_tiny_patch4_window7_224
      num_classes: 1
      dropout_rate: 0.2
    ```

    Args:
        config: 配置字典或DictConfig对象
        **kwargs: 额外的参数（会覆盖config中的params）

    Returns:
        创建的对象实例

    Raises:
        KeyError: 如果配置中缺少 'target' 字段
        ImportError: 如果无法导入指定的类
    """
    # 处理特殊情况
    if not isinstance(config, (dict, DictConfig)):
        raise TypeError(f"Config must be dict or DictConfig, got {type(config)}")

    # 检查必需的 'target' 字段
    if "target" not in config:
        # 处理一些特殊的配置标记（借鉴latent-diffusion的设计）
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError(
            "Expected key `target` to instantiate object. "
            "Config format should be: {'target': 'path.to.Class', 'params': {...}}"
        )

    # 获取目标类
    target_cls = get_obj_from_str(config["target"])

    # 合并参数：config中的params + 传入的kwargs
    params = config.get("params", {})
    if isinstance(params, DictConfig):
        params = OmegaConf.to_container(params, resolve=True)

    # kwargs的优先级更高，会覆盖config中的同名参数
    final_params = {**params, **kwargs}

    logger.info(f"Instantiating {config['target']} with params: {list(final_params.keys())}")

    try:
        # 创建对象实例
        return target_cls(**final_params)
    except Exception as e:
        logger.error(f"Failed to instantiate {config['target']}: {e}")
        logger.error(f"Parameters: {final_params}")
        raise


def instantiate_with_config_group(
    configs: Dict[str, Union[Dict, DictConfig]], group_name: str, **kwargs
) -> Dict[str, Any]:
    """
    批量实例化配置组中的多个对象

    这个函数可以一次性创建多个相关的对象，比如同时创建
    模型、数据加载器、优化器等。在复杂的实验中很有用。

    Args:
        configs: 包含多个配置的字典
        group_name: 配置组名称（用于日志记录）
        **kwargs: 传递给所有对象的额外参数

    Returns:
        包含所有实例化对象的字典

    Example:
        >>> configs = {
        ...     'model': {'target': 'MyModel', 'params': {'hidden_dim': 256}},
        ...     'optimizer': {'target': 'torch.optim.Adam', 'params': {'lr': 1e-4}}
        ... }
        >>> objects = instantiate_with_config_group(configs, 'training')
        >>> model = objects['model']
        >>> optimizer = objects['optimizer']
    """
    logger.info(f"Instantiating {group_name} group with {len(configs)} components")

    instantiated = {}
    for name, config in configs.items():
        try:
            obj = instantiate_from_config(config, **kwargs)
            instantiated[name] = obj
            logger.debug(f"✓ Created {name}: {type(obj).__name__}")
        except Exception as e:
            logger.error(f"✗ Failed to create {name}: {e}")
            raise

    logger.info(f"Successfully instantiated {len(instantiated)} {group_name} components")
    return instantiated


def validate_config_structure(config: Union[Dict, DictConfig], required_sections: Optional[list] = None) -> bool:
    """
    验证配置文件的基本结构

    在实例化对象之前，先检查配置文件的格式是否正确。
    这可以帮助我们提前发现配置错误，避免浪费时间。

    Args:
        config: 要验证的配置
        required_sections: 必需的配置段列表

    Returns:
        True if valid, False otherwise
    """
    if required_sections is None:
        required_sections = ["model", "data", "training"]

    missing_sections = []
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)

    if missing_sections:
        logger.error(f"Missing required config sections: {missing_sections}")
        return False

    # 检查每个section是否有target字段（如果它是一个对象配置）
    for section_name, section_config in config.items():
        if isinstance(section_config, (dict, DictConfig)) and "target" in section_config:
            try:
                # 尝试导入target指定的类，验证其是否存在
                get_obj_from_str(section_config["target"])
                logger.debug(f"✓ Validated {section_name}.target: {section_config['target']}")
            except ImportError as e:
                logger.error(f"✗ Invalid {section_name}.target: {section_config['target']} - {e}")
                return False

    logger.info("✓ Configuration structure validation passed")
    return True


# 便利函数：为常见的PyTorch Lightning组件提供快捷方式
def create_model_from_config(config: Union[Dict, DictConfig], **kwargs) -> Any:
    """从配置创建模型的便利函数"""
    return instantiate_from_config(config, **kwargs)


def create_data_module_from_config(config: Union[Dict, DictConfig], **kwargs) -> Any:
    """从配置创建数据模块的便利函数"""
    return instantiate_from_config(config, **kwargs)


def create_trainer_from_config(config: Union[Dict, DictConfig], **kwargs) -> Any:
    """从配置创建训练器的便利函数"""
    return instantiate_from_config(config, **kwargs)


# 工厂注册机制：为复杂情况提供额外的灵活性
_FACTORY_REGISTRY = {}


def register_factory(name: str, factory_func: callable):
    """
    注册自定义工厂函数

    对于一些特别复杂的对象创建逻辑，可以注册专门的工厂函数。
    """
    _FACTORY_REGISTRY[name] = factory_func
    logger.info(f"Registered factory: {name}")


def create_from_factory(factory_name: str, config: Union[Dict, DictConfig], **kwargs) -> Any:
    """使用注册的工厂函数创建对象"""
    if factory_name not in _FACTORY_REGISTRY:
        raise KeyError(f"Unknown factory: {factory_name}. " f"Available: {list(_FACTORY_REGISTRY.keys())}")

    factory_func = _FACTORY_REGISTRY[factory_name]
    return factory_func(config, **kwargs)


if __name__ == "__main__":
    # 简单的使用示例和测试
    logging.basicConfig(level=logging.INFO)

    # 测试基本的实例化功能
    test_config = {"target": "torch.nn.Linear", "params": {"in_features": 10, "out_features": 5}}

    try:
        linear_layer = instantiate_from_config(test_config)
        print(f"✓ Successfully created: {linear_layer}")
        print(f"  Layer shape: {linear_layer.in_features} -> {linear_layer.out_features}")
    except Exception as e:
        print(f"✗ Test failed: {e}")
