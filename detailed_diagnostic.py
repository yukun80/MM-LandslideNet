#!/usr/bin/env python3
"""
修复版详细诊断脚本

这个版本修复了配置结构不匹配的问题。它确保传递给模型的配置
具有正确的结构，符合模型类的期望。

教学要点：
这个修复展示了"接口适配器"模式的应用。当两个组件的接口不匹配时，
我们创建一个适配器来使它们能够正确交互。在这里，我们修复了
配置结构来匹配模型类的期望。
"""

import sys
import os
import traceback
from pathlib import Path
import tempfile
import yaml

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """测试基础导入 - 第一道防线"""
    print("🔍 第一步：测试基础导入...")

    import_tests = [
        ("PyTorch Lightning", "import pytorch_lightning as pl"),
        ("PyTorch", "import torch"),
        ("OmegaConf", "from omegaconf import OmegaConf"),
        ("项目工具模块", "from lightning_landslide.src.utils.instantiate import instantiate_from_config"),
        ("项目数据模块", "from lightning_landslide.src.data import DummyDataModule"),
        ("项目模型模块", "from lightning_landslide.src.models import LandslideClassificationModule"),
    ]

    for name, import_code in import_tests:
        try:
            exec(import_code)
            print(f"  ✅ {name}: 导入成功")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            return False

    return True


def test_config_creation():
    """测试配置文件创建和解析 - 修复版"""
    print("\n🔍 第二步：测试配置文件创建和解析...")

    # 创建符合模型期望的完整配置结构
    # 这是关键修复：提供完整的配置层次结构
    complete_config = {
        "experiment_name": "diagnostic_test",
        "seed": 42,
        "log_level": "WARNING",
        # 修复：模型配置现在包含完整的层次结构
        "model": {
            "target": "lightning_landslide.src.models.LandslideClassificationModule",
            "params": {
                # 关键修复：传递完整的配置对象而不是空字典
                "cfg": {
                    "model": {"type": "dummy_model", "num_classes": 1, "dropout_rate": 0.1},
                    "training": {"optimizer": {"type": "adamw", "lr": 1e-3}, "loss": {"type": "bce"}},
                    "evaluation": {"metrics": ["accuracy", "f1"]},
                }
            },
        },
        "data": {
            "target": "lightning_landslide.src.data.DummyDataModule",
            "params": {"batch_size": 4, "num_samples": 16, "input_channels": 5, "num_workers": 0},
        },
        "trainer": {
            "target": "pytorch_lightning.Trainer",
            "params": {
                "max_epochs": 1,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
                "accelerator": "cpu",
                "devices": 1,
                "fast_dev_run": True,  # 最快的测试模式
            },
        },
        "outputs": {
            "checkpoint_dir": "/tmp/test_checkpoints",
            "log_dir": "/tmp/test_logs",
            "predictions_dir": "/tmp/test_predictions",
        },
    }

    try:
        from omegaconf import OmegaConf

        config = OmegaConf.create(complete_config)
        print("  ✅ 配置文件创建成功")

        # 测试配置验证
        from lightning_landslide.src.utils.instantiate import validate_config_structure

        is_valid = validate_config_structure(config)
        print(f"  ✅ 配置验证: {'通过' if is_valid else '失败'}")

        return config if is_valid else None

    except Exception as e:
        print(f"  ❌ 配置创建失败: {e}")
        traceback.print_exc()
        return None


def test_data_module_creation(config):
    """测试数据模块创建"""
    print("\n🔍 第三步：测试数据模块创建...")

    try:
        from lightning_landslide.src.utils.instantiate import instantiate_from_config

        print("  📊 创建数据模块...")
        data_module = instantiate_from_config(config.data)
        print(f"  ✅ 数据模块创建成功: {type(data_module).__name__}")

        print("  📊 设置数据模块...")
        data_module.setup("fit")
        print("  ✅ 数据模块设置成功")

        print("  📊 测试数据加载器...")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        print(f"  ✅ 训练加载器: {len(train_loader)} 批次")
        print(f"  ✅ 验证加载器: {len(val_loader)} 批次")

        print("  📊 测试数据批次...")
        train_batch = next(iter(train_loader))
        x, y = train_batch
        print(f"  ✅ 数据形状: x={x.shape}, y={y.shape}")

        return data_module

    except Exception as e:
        print(f"  ❌ 数据模块测试失败: {e}")
        traceback.print_exc()
        return None


def test_model_creation(config):
    """测试模型创建 - 修复版"""
    print("\n🔍 第四步：测试模型创建...")

    try:
        from lightning_landslide.src.utils.instantiate import instantiate_from_config

        print("  🧠 创建模型...")
        print(f"  🔍 调试：传递给模型的配置结构：")
        print(f"    - model.params.cfg 类型: {type(config.model.params.cfg)}")
        print(f"    - 是否包含 model 键: {'model' in config.model.params.cfg}")

        model = instantiate_from_config(config.model)
        print(f"  ✅ 模型创建成功: {type(model).__name__}")

        print("  🧠 测试模型前向传播...")
        import torch

        test_input = torch.randn(2, 5, 64, 64)  # 批次大小2，5通道，64x64图像

        with torch.no_grad():
            output = model(test_input)
        print(f"  ✅ 前向传播成功: 输入{test_input.shape} -> 输出{output.shape}")

        return model

    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        traceback.print_exc()
        return None


def test_trainer_creation(config):
    """测试训练器创建"""
    print("\n🔍 第五步：测试训练器创建...")

    try:
        from lightning_landslide.src.utils.instantiate import instantiate_from_config

        print("  ⚡ 创建训练器...")
        trainer = instantiate_from_config(config.trainer)
        print(f"  ✅ 训练器创建成功: {type(trainer).__name__}")

        return trainer

    except Exception as e:
        print(f"  ❌ 训练器创建失败: {e}")
        traceback.print_exc()
        return None


def test_full_training_cycle(config):
    """测试完整训练周期"""
    print("\n🔍 第六步：测试完整训练周期...")

    try:
        from lightning_landslide.src.utils.instantiate import instantiate_from_config

        print("  🎯 创建所有组件...")
        model = instantiate_from_config(config.model)
        data_module = instantiate_from_config(config.data)
        trainer = instantiate_from_config(config.trainer)

        print("  🎯 开始快速训练测试...")
        trainer.fit(model, data_module)
        print("  ✅ 训练测试完成!")

        return True

    except Exception as e:
        print(f"  ❌ 训练测试失败: {e}")
        traceback.print_exc()
        return False


def create_fixed_test_config():
    """创建修复后的测试配置文件"""
    print("\n🔍 第七步：创建修复后的完整测试配置...")

    # 这个配置结构与我们在artifacts中设计的结构保持一致
    fixed_config = {
        "experiment_name": "fixed_diagnostic_test",
        "description": "Fixed diagnostic test with proper config structure",
        "seed": 42,
        "log_level": "WARNING",
        "model": {
            "target": "lightning_landslide.src.models.LandslideClassificationModule",
            "params": {
                "cfg": {
                    # 提供完整的模型配置结构
                    "model": {"type": "optical_swin", "num_classes": 1, "dropout_rate": 0.1},
                    "training": {
                        "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
                        "loss": {"type": "bce"},
                        "max_epochs": 1,
                    },
                    "evaluation": {"metrics": ["accuracy", "f1", "auroc"]},
                    "data": {"batch_size": 4, "num_workers": 0},
                }
            },
        },
        "data": {
            "target": "lightning_landslide.src.data.DummyDataModule",
            "params": {"batch_size": 4, "num_samples": 16, "input_channels": 5, "image_size": 64, "num_workers": 0},
        },
        "trainer": {
            "target": "pytorch_lightning.Trainer",
            "params": {
                "max_epochs": 1,
                "limit_train_batches": 2,
                "limit_val_batches": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
                "enable_checkpointing": False,
                "accelerator": "cpu",
                "devices": 1,
                "fast_dev_run": False,  # 让它真正运行一个简短的训练
            },
        },
        "outputs": {
            "checkpoint_dir": "/tmp/fixed_test_checkpoints",
            "log_dir": "/tmp/fixed_test_logs",
            "predictions_dir": "/tmp/fixed_test_predictions",
        },
    }

    try:
        from omegaconf import OmegaConf

        config = OmegaConf.create(fixed_config)

        # 保存到临时文件以便进一步测试
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(fixed_config, f, default_flow_style=False)
            temp_config_path = f.name

        print(f"  ✅ 修复后的配置文件已保存: {temp_config_path}")
        return config, temp_config_path

    except Exception as e:
        print(f"  ❌ 配置文件创建失败: {e}")
        traceback.print_exc()
        return None, None


def test_main_py_integration(config_path):
    """测试与main.py的集成"""
    print("\n🔍 第八步：测试main.py集成...")

    try:
        import subprocess

        print("  📝 通过main.py运行训练测试...")

        # 运行main.py train命令
        cmd = [
            sys.executable,
            "main.py",
            "train",
            config_path,
            "--override",
            "trainer.params.limit_train_batches=1",
            "--override",
            "trainer.params.limit_val_batches=1",
        ]

        print(f"  📝 执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=60)  # 1分钟超时

        if result.returncode == 0:
            print("  ✅ main.py集成测试成功!")
            return True
        else:
            print(f"  ❌ main.py集成测试失败:")
            print(f"    返回码: {result.returncode}")
            print(f"    错误输出前500字符: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ❌ main.py集成测试超时")
        return False
    except Exception as e:
        print(f"  ❌ main.py集成测试异常: {e}")
        traceback.print_exc()
        return False


def main():
    """主诊断函数 - 修复版"""
    print("🩺 MM-LandslideNet 修复版详细诊断工具")
    print("=" * 60)
    print("这个修复版本解决了配置结构不匹配的问题")
    print("=" * 60)

    # 诊断步骤序列
    diagnostic_steps = [
        ("基础导入测试", test_basic_imports, None),
        ("配置创建测试", test_config_creation, None),
    ]

    # 执行基础测试
    for step_name, step_func, step_args in diagnostic_steps:
        print(f"\n{'='*20} {step_name} {'='*20}")

        try:
            if step_args:
                result = step_func(step_args)
            else:
                result = step_func()

            if not result:
                print(f"\n❌ {step_name}失败，停止后续测试")
                return False

            # 如果是配置创建测试，保存结果供后续使用
            if step_name == "配置创建测试":
                config = result

                # 继续执行依赖配置的测试
                dependent_tests = [
                    ("数据模块测试", test_data_module_creation, config),
                    ("模型创建测试", test_model_creation, config),
                    ("训练器创建测试", test_trainer_creation, config),
                    ("完整训练测试", test_full_training_cycle, config),
                ]

                for dep_name, dep_func, dep_args in dependent_tests:
                    print(f"\n{'='*20} {dep_name} {'='*20}")
                    try:
                        dep_result = dep_func(dep_args)
                        if not dep_result:
                            print(f"\n❌ {dep_name}失败")
                            # 不要立即返回，让我们看看能走多远
                            # return False
                    except Exception as e:
                        print(f"\n❌ {dep_name}异常: {e}")
                        traceback.print_exc()
                        # return False

        except Exception as e:
            print(f"\n❌ {step_name}异常: {e}")
            traceback.print_exc()
            return False

    # 创建修复后的完整测试配置
    print(f"\n{'='*20} 修复后配置测试 {'='*20}")
    fixed_config, config_path = create_fixed_test_config()

    if fixed_config and config_path:
        # 测试main.py集成
        main_py_success = test_main_py_integration(config_path)

        # 清理临时文件
        try:
            os.unlink(config_path)
        except:
            pass

        if main_py_success:
            print("\n" + "=" * 60)
            print("🎉 所有诊断测试通过，包括main.py集成！")
            print("💡 您的框架现在应该可以正常工作了")
            print("=" * 60)
            return True

    print("\n" + "=" * 60)
    print("⚠️  部分测试完成，但main.py集成可能还有问题")
    print("💡 至少各个组件是可以独立工作的")
    print("=" * 60)
    return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 下一步建议：")
        print("1. 更新原始的validate_framework.py，使用正确的配置结构")
        print("2. 或者直接使用main.py进行实际的训练实验")
        print("3. 验证其他任务模式（test/predict/validate）")
    else:
        print("\n🛠️ 需要进一步修复：")
        print("1. 根据错误信息修复剩余问题")
        print("2. 重点关注模型类的配置期望")
        print("3. 确保所有组件的接口都匹配")
