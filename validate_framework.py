#!/usr/bin/env python3
"""
完整框架功能测试脚本

这个脚本是我们重构工作的"最终验收测试"。它验证新框架的所有功能：
- 训练 (train)
- 测试 (test)
- 预测 (predict)
- 验证 (validate)

就像是一个产品出厂前的"全面质检"，确保每个功能都完美工作。

教学要点：
这个脚本展示了如何系统性地测试一个深度学习框架的完整功能。
它不仅测试了各个任务能否运行，还验证了它们的输出是否符合预期。
这种全面的测试方法是软件工程中的重要实践。
"""

import sys
import os
import subprocess
import tempfile
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import shutil

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("comprehensive_test.log")],
)
logger = logging.getLogger(__name__)


class ComprehensiveFrameworkTester:
    """
    完整框架测试器

    这个类就像是一个严格的"质量检查员"，它会系统性地测试
    我们框架的每一个功能，确保没有任何遗漏。

    测试流程：
    1. 准备测试环境和配置文件
    2. 运行训练任务（产生检查点）
    3. 基于训练结果运行测试任务
    4. 运行预测任务
    5. 运行验证任务
    6. 验证所有输出文件
    7. 清理测试环境
    """

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "test_framework_outputs"
        self.temp_configs = []
        self.test_results = {}
        self.checkpoint_path = None

    def setup_test_environment(self):
        """
        设置测试环境

        创建临时目录和配置文件，为测试做准备。
        这就像是在实验室中准备所有需要的器材和材料。
        """
        logger.info("🏗️  Setting up test environment...")

        # 创建测试输出目录
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)

        # 创建子目录
        for subdir in ["checkpoints", "logs", "predictions", "test_results", "configs"]:
            (self.test_dir / subdir).mkdir(parents=True)

        logger.info(f"✅ Test environment created at: {self.test_dir}")

    def create_test_configs(self) -> Dict[str, Path]:
        """
        创建所有任务的测试配置文件

        为每个任务（train/test/predict/validate）创建专门的配置文件。
        这些配置都经过优化，能够快速运行并产生可验证的结果。

        Returns:
            包含所有配置文件路径的字典
        """
        logger.info("📄 Creating test configuration files...")

        configs = {}

        # 基础配置：用于训练任务
        train_config = {
            "experiment_name": "framework_test_train",
            "description": "Training task test for comprehensive framework validation",
            "seed": 42,
            "log_level": "WARNING",  # 减少日志输出
            "model": {
                "target": "lightning_landslide.src.models.LandslideClassificationModule",
                "params": {
                    "base_model": {
                        "target": "lightning_landslide.src.models.optical_swin.OpticalSwinModel",
                        "params": {
                            "model_name": "swin_tiny_patch4_window7_224",
                            "input_channels": 5,
                            "num_classes": 1,
                            "dropout_rate": 0.1,
                            "pretrained": False,  # 避免下载预训练权重
                        },
                    },
                    "loss_config": {"type": "bce"},
                    "optimizer_config": {"type": "adamw", "adamw_params": {"lr": 1e-3, "weight_decay": 1e-4}},
                    "scheduler_config": {"type": "constant"},
                    "metrics_config": {"primary_metric": "f1", "metrics": ["accuracy", "f1", "auroc"]},
                },
            },
            "data": {
                "target": "lightning_landslide.src.data.DummyDataModule",
                "params": {
                    "batch_size": 8,
                    "num_samples": 64,  # 小数据集，快速训练
                    "input_channels": 5,
                    "image_size": 64,
                    "num_workers": 0,
                },
            },
            "trainer": {
                "target": "pytorch_lightning.Trainer",
                "params": {
                    "max_epochs": 2,  # 只训练2个epoch
                    "limit_train_batches": 4,  # 每个epoch只训练4个batch
                    "limit_val_batches": 2,  # 每次验证只用2个batch
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "logger": False,
                    "accelerator": "cpu",  # 强制使用CPU，避免GPU相关问题
                    "devices": 1,
                    "fast_dev_run": False,
                    "check_val_every_n_epoch": 1,
                },
            },
            "callbacks": {
                "model_checkpoint": {
                    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "params": {
                        "dirpath": str(self.test_dir / "checkpoints"),
                        "filename": "test_model",
                        "monitor": "val_f1",
                        "mode": "max",
                        "save_top_k": 1,
                        "save_last": True,
                    },
                }
            },
            "outputs": {
                "checkpoint_dir": str(self.test_dir / "checkpoints"),
                "log_dir": str(self.test_dir / "logs"),
                "predictions_dir": str(self.test_dir / "predictions"),
                "figures_dir": str(self.test_dir / "figures"),
            },
        }

        # 保存训练配置
        train_config_path = self.test_dir / "configs" / "train_config.yaml"
        with open(train_config_path, "w") as f:
            yaml.dump(train_config, f, default_flow_style=False)
        configs["train"] = train_config_path
        self.temp_configs.append(train_config_path)

        # 测试配置：基于训练配置，但用于测试
        test_config = train_config.copy()
        test_config.update(
            {
                "experiment_name": "framework_test_test",
                "description": "Test task for comprehensive framework validation",
                "checkpoint_path": str(self.test_dir / "checkpoints" / "test_model.ckpt"),
            }
        )
        # 修改trainer配置，专门用于测试
        test_config["trainer"]["params"].update(
            {"logger": False, "enable_checkpointing": False, "limit_test_batches": 2}
        )

        test_config_path = self.test_dir / "configs" / "test_config.yaml"
        with open(test_config_path, "w") as f:
            yaml.dump(test_config, f, default_flow_style=False)
        configs["test"] = test_config_path
        self.temp_configs.append(test_config_path)

        # 预测配置：用于推理任务
        predict_config = train_config.copy()
        predict_config.update(
            {
                "experiment_name": "framework_test_predict",
                "description": "Prediction task for comprehensive framework validation",
                "checkpoint_path": str(self.test_dir / "checkpoints" / "test_model.ckpt"),
            }
        )
        predict_config["trainer"]["params"].update(
            {"logger": False, "enable_checkpointing": False, "limit_predict_batches": 2}
        )

        predict_config_path = self.test_dir / "configs" / "predict_config.yaml"
        with open(predict_config_path, "w") as f:
            yaml.dump(predict_config, f, default_flow_style=False)
        configs["predict"] = predict_config_path
        self.temp_configs.append(predict_config_path)

        # 验证配置：用于验证任务
        validate_config = train_config.copy()
        validate_config.update(
            {
                "experiment_name": "framework_test_validate",
                "description": "Validation task for comprehensive framework validation",
                "checkpoint_path": str(self.test_dir / "checkpoints" / "test_model.ckpt"),
            }
        )
        validate_config["trainer"]["params"].update(
            {"logger": False, "enable_checkpointing": False, "limit_val_batches": 2}
        )

        validate_config_path = self.test_dir / "configs" / "validate_config.yaml"
        with open(validate_config_path, "w") as f:
            yaml.dump(validate_config, f, default_flow_style=False)
        configs["validate"] = validate_config_path
        self.temp_configs.append(validate_config_path)

        logger.info(f"✅ Created {len(configs)} test configuration files")
        return configs

    def run_task_test(self, task: str, config_path: Path, timeout: int = 300) -> Tuple[bool, str, str]:
        """
        运行单个任务的测试

        Args:
            task: 任务名称 (train/test/predict/validate)
            config_path: 配置文件路径
            timeout: 超时时间（秒）

        Returns:
            (success, stdout, stderr) 元组
        """
        logger.info(f"🧪 Testing {task} task...")

        # 构建命令
        cmd = [sys.executable, "main.py", task, str(config_path)]

        try:
            # 运行命令
            logger.info(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=timeout)

            success = result.returncode == 0

            if success:
                logger.info(f"✅ {task} task completed successfully")
            else:
                logger.error(f"❌ {task} task failed with return code {result.returncode}")
                logger.error(f"  Error output: {result.stderr[:500]}...")  # 只显示前500个字符

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {task} task timed out after {timeout} seconds")
            return False, "", f"Task timed out after {timeout} seconds"
        except Exception as e:
            logger.error(f"❌ {task} task failed with exception: {e}")
            return False, "", str(e)

    def verify_outputs(self, task: str) -> bool:
        """
        验证任务输出文件

        检查每个任务是否产生了预期的输出文件。
        这就像是检查工厂生产线是否产出了合格的产品。

        Args:
            task: 任务名称

        Returns:
            验证是否通过
        """
        logger.info(f"🔍 Verifying {task} task outputs...")

        verification_passed = True

        if task == "train":
            # 训练任务应该产生检查点文件
            checkpoint_dir = self.test_dir / "checkpoints"
            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

            if checkpoint_files:
                self.checkpoint_path = checkpoint_files[0]
                logger.info(f"  ✅ Found checkpoint: {self.checkpoint_path}")
            else:
                logger.error("  ❌ No checkpoint files found")
                verification_passed = False

        elif task == "test":
            # 测试任务应该产生结果文件
            results_dir = self.test_dir / "predictions"
            result_files = list(results_dir.glob("test_results_*.json"))

            if result_files:
                logger.info(f"  ✅ Found test results: {result_files[0]}")
                # 验证结果文件内容
                try:
                    with open(result_files[0], "r") as f:
                        results = json.load(f)
                    if "test_results" in results:
                        logger.info("  ✅ Test results contain expected fields")
                    else:
                        logger.error("  ❌ Test results missing expected fields")
                        verification_passed = False
                except Exception as e:
                    logger.error(f"  ❌ Failed to read test results: {e}")
                    verification_passed = False
            else:
                logger.error("  ❌ No test result files found")
                verification_passed = False

        elif task == "predict":
            # 预测任务应该产生预测文件
            predictions_dir = self.test_dir / "predictions"
            pred_files = list(predictions_dir.glob("predictions_*.json"))

            if pred_files:
                logger.info(f"  ✅ Found predictions: {pred_files[0]}")
                # 验证预测文件内容
                try:
                    with open(pred_files[0], "r") as f:
                        predictions = json.load(f)
                    if "predictions" in predictions and len(predictions["predictions"]) > 0:
                        logger.info(f"  ✅ Found {len(predictions['predictions'])} predictions")
                    else:
                        logger.error("  ❌ Predictions file is empty or malformed")
                        verification_passed = False
                except Exception as e:
                    logger.error(f"  ❌ Failed to read predictions: {e}")
                    verification_passed = False
            else:
                logger.error("  ❌ No prediction files found")
                verification_passed = False

        elif task == "validate":
            # 验证任务应该产生验证结果文件
            results_dir = self.test_dir / "predictions"
            val_files = list(results_dir.glob("validation_results_*.json"))

            if val_files:
                logger.info(f"  ✅ Found validation results: {val_files[0]}")
            else:
                logger.error("  ❌ No validation result files found")
                verification_passed = False

        return verification_passed

    def run_comprehensive_test(self) -> bool:
        """
        运行完整的框架测试

        这是测试的主函数，它按照正确的顺序执行所有测试，
        并生成详细的测试报告。

        Returns:
            所有测试是否都通过
        """
        logger.info("🚀 Starting comprehensive framework test")
        logger.info("=" * 70)

        try:
            # 1. 设置测试环境
            self.setup_test_environment()

            # 2. 创建测试配置
            configs = self.create_test_configs()

            # 3. 按顺序运行所有任务测试
            test_sequence = [
                ("train", configs["train"]),
                ("test", configs["test"]),
                ("predict", configs["predict"]),
                ("validate", configs["validate"]),
            ]

            all_tests_passed = True

            for task, config_path in test_sequence:
                logger.info(f"\n{'='*20} Testing {task.upper()} Task {'='*20}")

                # 运行任务
                success, stdout, stderr = self.run_task_test(task, config_path)

                # 记录结果
                self.test_results[task] = {
                    "success": success,
                    "stdout_length": len(stdout),
                    "stderr_length": len(stderr),
                    "has_errors": len(stderr) > 0,
                }

                if success:
                    # 验证输出
                    output_valid = self.verify_outputs(task)
                    self.test_results[task]["output_valid"] = output_valid

                    if not output_valid:
                        all_tests_passed = False
                else:
                    all_tests_passed = False
                    self.test_results[task]["output_valid"] = False

                # 如果训练失败，后续测试无法进行
                if task == "train" and not success:
                    logger.error("❌ Training failed, skipping remaining tests")
                    break

            # 4. 生成测试报告
            self.generate_test_report(all_tests_passed)

            return all_tests_passed

        except Exception as e:
            logger.error(f"💥 Comprehensive test failed with exception: {e}")
            return False
        finally:
            # 清理测试环境（可选）
            # self.cleanup()
            pass

    def generate_test_report(self, all_passed: bool):
        """
        生成详细的测试报告

        Args:
            all_passed: 是否所有测试都通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("📊 COMPREHENSIVE TEST REPORT")
        logger.info("=" * 70)

        # 总体状态
        overall_status = "✅ PASSED" if all_passed else "❌ FAILED"
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Test Environment: {self.test_dir}")
        logger.info(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 各任务详细结果
        logger.info("\n📋 Task Results:")
        for task, results in self.test_results.items():
            status = "✅ PASSED" if results["success"] and results.get("output_valid", True) else "❌ FAILED"
            logger.info(f"  {task.upper()}: {status}")

            if not results["success"]:
                logger.info(f"    - Execution failed")
            elif not results.get("output_valid", True):
                logger.info(f"    - Output validation failed")
            else:
                logger.info(f"    - All checks passed")

        # 输出文件统计
        logger.info(f"\n📁 Generated Files:")
        for file_path in self.test_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".ckpt", ".json", ".csv", ".yaml"]:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path.relative_to(self.test_dir)}: {size_mb:.2f} MB")

        # 结论和建议
        logger.info(f"\n🎯 Conclusion:")
        if all_passed:
            logger.info("🎉 All tests passed! Your framework is ready for production use.")
            logger.info("💡 You can now safely use all four task modes:")
            logger.info("   - python main.py train config.yaml")
            logger.info("   - python main.py test config.yaml")
            logger.info("   - python main.py predict config.yaml")
            logger.info("   - python main.py validate config.yaml")
        else:
            logger.info("⚠️  Some tests failed. Please review the errors above.")
            logger.info("🛠️  Fix the failing components before using the framework.")

        logger.info("=" * 70)

    def cleanup(self):
        """清理测试环境"""
        logger.info("🧹 Cleaning up test environment...")

        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                logger.info("✅ Test directory cleaned up")
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup test directory: {e}")


def main():
    """主函数：运行完整的框架测试"""
    print("🔍 MM-LandslideNet Comprehensive Framework Test")
    print("=" * 70)
    print("This test validates all four task modes of the new framework:")
    print("- Training (train)")
    print("- Testing (test)")
    print("- Prediction (predict)")
    print("- Validation (validate)")
    print("=" * 70)

    tester = ComprehensiveFrameworkTester()
    success = tester.run_comprehensive_test()

    if success:
        print("\n🎉 SUCCESS: All framework components are working correctly!")
        print("🚀 Your MM-LandslideNet framework is ready for real experiments!")
        return 0
    else:
        print("\n💥 FAILURE: Some framework components need attention.")
        print("🛠️  Please review the test output above and fix any issues.")
        return 1


if __name__ == "__main__":
    exit(main())
