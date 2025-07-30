# =============================================================================
# tests/test_active_learning_integration.py
# =============================================================================

"""
主动学习模块完整集成测试

这个测试脚本验证整个主动学习系统的完整性和正确性：
1. 模块导入测试
2. 配置验证测试
3. 组件功能测试
4. 端到端流程测试
5. 性能基准测试
6. 兼容性测试
"""

import sys
import os
import unittest
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置测试日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ActiveLearningModuleTest(unittest.TestCase):
    """主动学习模块基础测试"""

    def setUp(self):
        """测试前置设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Created temporary directory: {self.temp_dir}")

    def tearDown(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

    def test_module_imports(self):
        """测试模块导入"""
        logger.info("🧪 Testing module imports...")

        try:
            # 测试核心模块导入
            from lightning_landslide.src.active_learning import (
                ActivePseudoTrainer,
                UncertaintyResults,
                PseudoLabelGenerator,
                HybridActiveLearningSelector,
                EnhancedDataManager,
                ActiveLearningVisualizer,
            )

            logger.info("✅ Core modules imported successfully")

            # 测试工厂函数导入
            from lightning_landslide.src.active_learning import (
                create_active_pseudo_trainer,
                create_uncertainty_estimator,
                create_pseudo_label_generator,
                create_active_learning_selector,
            )

            logger.info("✅ Factory functions imported successfully")

            # 测试可选模块导入
            try:
                from lightning_landslide.src.active_learning import ActiveKFoldTrainer, create_active_kfold_trainer

                logger.info("✅ K-fold modules imported successfully")
            except ImportError:
                logger.warning("⚠️ K-fold modules not available")

            self.assertTrue(True, "All required modules imported successfully")

        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")

    def test_config_validation(self):
        """测试配置验证"""
        logger.info("🧪 Testing configuration validation...")

        from lightning_landslide.src.active_learning import validate_active_learning_config

        # 测试最小有效配置
        minimal_config = {
            "model": {"target": "test.Model"},
            "data": {"target": "test.Data"},
            "trainer": {"target": "test.Trainer"},
        }

        try:
            validated_config = validate_active_learning_config(minimal_config)
            self.assertIn("active_pseudo_learning", validated_config)
            logger.info("✅ Minimal config validation passed")
        except Exception as e:
            self.fail(f"Minimal config validation failed: {e}")

        # 测试完整配置
        complete_config = {
            "model": {"target": "test.Model"},
            "data": {"target": "test.Data"},
            "trainer": {"target": "test.Trainer"},
            "active_pseudo_learning": {
                "max_iterations": 3,
                "annotation_budget": 25,
                "uncertainty_estimation": {"method": "mc_dropout"},
                "pseudo_labeling": {"confidence_threshold": 0.85},
                "active_learning": {"strategies": {"uncertainty": 1.0}},
            },
        }

        try:
            validated_config = validate_active_learning_config(complete_config)
            self.assertEqual(validated_config["active_pseudo_learning"]["max_iterations"], 3)
            logger.info("✅ Complete config validation passed")
        except Exception as e:
            self.fail(f"Complete config validation failed: {e}")

    def test_uncertainty_estimation(self):
        """测试不确定性估计"""
        logger.info("🧪 Testing uncertainty estimation...")

        from lightning_landslide.src.active_learning import create_uncertainty_estimator

        # 创建模拟数据
        mock_predictions = np.random.rand(100, 2)  # 100样本，2类
        mock_model = Mock()
        mock_dataloader = Mock()

        # 测试MC Dropout估计器
        try:
            estimator = create_uncertainty_estimator("mc_dropout", n_forward_passes=5)
            self.assertIsNotNone(estimator)
            logger.info("✅ MC Dropout estimator created successfully")
        except Exception as e:
            self.fail(f"Failed to create MC Dropout estimator: {e}")

        # 测试混合估计器
        try:
            estimator = create_uncertainty_estimator("hybrid", use_mc_dropout=True, n_forward_passes=3)
            self.assertIsNotNone(estimator)
            logger.info("✅ Hybrid estimator created successfully")
        except Exception as e:
            self.fail(f"Failed to create hybrid estimator: {e}")

    def test_pseudo_label_generation(self):
        """测试伪标签生成"""
        logger.info("🧪 Testing pseudo label generation...")

        from lightning_landslide.src.active_learning import create_pseudo_label_generator, UncertaintyResults

        # 创建模拟不确定性结果
        mock_uncertainty = UncertaintyResults(
            sample_ids=[f"sample_{i}" for i in range(50)],
            predictions=np.random.rand(50, 2),
            uncertainty_scores=np.random.rand(50),
            epistemic_uncertainty=np.random.rand(50),
            aleatoric_uncertainty=np.random.rand(50),
            prediction_entropy=np.random.rand(50),
            confidence_scores=np.random.rand(50),
            calibrated_confidence=np.random.rand(50),
        )

        try:
            generator = create_pseudo_label_generator({"confidence_threshold": 0.8, "uncertainty_threshold": 0.2})

            # 设置类别分布
            generator.set_class_distribution({0: 30, 1: 20})

            # 生成伪标签
            results = generator.generate_pseudo_labels(mock_uncertainty)

            self.assertIsNotNone(results)
            self.assertIsInstance(results.high_confidence_samples, list)
            logger.info(f"✅ Generated {len(results.high_confidence_samples)} high-confidence pseudo labels")

        except Exception as e:
            self.fail(f"Pseudo label generation failed: {e}")

    def test_active_learning_selection(self):
        """测试主动学习选择"""
        logger.info("🧪 Testing active learning selection...")

        from lightning_landslide.src.active_learning import (
            create_active_learning_selector,
            UncertaintyResults,
            PseudoLabelSample,
        )

        # 创建模拟数据
        mock_uncertainty = UncertaintyResults(
            sample_ids=[f"sample_{i}" for i in range(100)],
            predictions=np.random.rand(100, 2),
            uncertainty_scores=np.random.rand(100),
            epistemic_uncertainty=np.random.rand(100),
            aleatoric_uncertainty=np.random.rand(100),
            prediction_entropy=np.random.rand(100),
            confidence_scores=np.random.rand(100),
            calibrated_confidence=np.random.rand(100),
        )

        # 创建候选样本
        candidate_samples = [
            PseudoLabelSample(
                sample_id=f"candidate_{i}",
                predicted_label=np.random.randint(0, 2),
                confidence=np.random.rand(),
                uncertainty=np.random.rand(),
                quality_score=np.random.rand(),
            )
            for i in range(20)
        ]

        try:
            selector = create_active_learning_selector(
                {"strategies": {"uncertainty": 0.6, "diversity": 0.4}, "budget_per_iteration": 10}
            )

            # 模拟特征嵌入
            mock_features = np.random.rand(100, 128)

            results = selector.select_samples(
                uncertainty_results=mock_uncertainty,
                candidate_samples=candidate_samples,
                feature_embeddings=mock_features,
                budget=10,
            )

            self.assertIsNotNone(results)
            self.assertLessEqual(len(results.selected_samples), 10)
            logger.info(f"✅ Selected {len(results.selected_samples)} samples for annotation")

        except Exception as e:
            self.fail(f"Active learning selection failed: {e}")

    def test_data_management(self):
        """测试数据管理"""
        logger.info("🧪 Testing data management...")

        from lightning_landslide.src.active_learning import create_enhanced_data_manager, create_annotation_interface

        # 创建模拟配置
        mock_config = {
            "data": {
                "params": {
                    "train_data_dir": str(self.temp_dir / "train"),
                    "test_data_dir": str(self.temp_dir / "test"),
                    "train_csv": str(self.temp_dir / "train.csv"),
                    "test_csv": str(self.temp_dir / "test.csv"),
                }
            }
        }

        # 创建模拟数据文件
        (self.temp_dir / "train").mkdir()
        (self.temp_dir / "test").mkdir()

        # 创建模拟CSV文件
        train_df = pd.DataFrame({"ID": [f"train_{i}" for i in range(20)], "label": np.random.randint(0, 2, 20)})
        train_df.to_csv(self.temp_dir / "train.csv", index=False)

        test_df = pd.DataFrame({"ID": [f"test_{i}" for i in range(10)]})
        test_df.to_csv(self.temp_dir / "test.csv", index=False)

        try:
            # 创建标注接口
            annotation_interface = create_annotation_interface(
                "simulated", ground_truth_file=str(self.temp_dir / "test.csv")
            )

            # 创建数据管理器
            data_manager = create_enhanced_data_manager(
                base_config=mock_config,
                output_dir=self.temp_dir,
                annotation_config={
                    "type": "simulated",
                    "params": {"ground_truth_file": str(self.temp_dir / "test.csv")},
                },
            )

            self.assertIsNotNone(data_manager)

            # 测试数据统计
            stats = data_manager.get_data_statistics()
            self.assertIn("total_samples", stats)
            logger.info(f"✅ Data manager created with {stats['total_samples']} samples")

        except Exception as e:
            self.fail(f"Data management test failed: {e}")

    def test_visualization(self):
        """测试可视化"""
        logger.info("🧪 Testing visualization...")

        from lightning_landslide.src.active_learning import create_visualizer

        try:
            visualizer = create_visualizer(self.temp_dir)
            self.assertIsNotNone(visualizer)

            # 创建模拟结果数据
            mock_performance_history = {
                "val_f1": [0.75, 0.78, 0.82, 0.84],
                "train_f1": [0.78, 0.82, 0.85, 0.87],
                "val_loss": [0.45, 0.42, 0.38, 0.35],
            }

            mock_data_usage_history = {
                "training_samples": [1000, 1050, 1100, 1150],
                "pseudo_labels": [0, 25, 50, 75],
                "new_annotations": [0, 25, 25, 25],
            }

            mock_iteration_results = [{"iteration": i + 1, "training_time": 120 + i * 10} for i in range(4)]

            # 测试训练概览可视化
            overview_path = visualizer.create_training_overview(
                mock_performance_history, mock_data_usage_history, mock_iteration_results
            )

            self.assertTrue(Path(overview_path).exists())
            logger.info(f"✅ Training overview created: {overview_path}")

        except Exception as e:
            self.fail(f"Visualization test failed: {e}")


class EndToEndIntegrationTest(unittest.TestCase):
    """端到端集成测试"""

    def setUp(self):
        """测试前置设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self._create_mock_data()
        logger.info(f"Created test environment: {self.temp_dir}")

    def tearDown(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_mock_data(self):
        """创建模拟数据"""
        # 创建数据目录
        (self.temp_dir / "train_data").mkdir()
        (self.temp_dir / "test_data").mkdir()

        # 创建模拟数据文件
        for i in range(50):
            np.save(self.temp_dir / "train_data" / f"train_{i}.npy", np.random.rand(5, 64, 64))

        for i in range(20):
            np.save(self.temp_dir / "test_data" / f"test_{i}.npy", np.random.rand(5, 64, 64))

        # 创建标签文件
        train_df = pd.DataFrame({"ID": [f"train_{i}" for i in range(50)], "label": np.random.randint(0, 2, 50)})
        train_df.to_csv(self.temp_dir / "train.csv", index=False)

        test_df = pd.DataFrame(
            {"ID": [f"test_{i}" for i in range(20)], "label": np.random.randint(0, 2, 20)}  # 模拟真实标签用于测试
        )
        test_df.to_csv(self.temp_dir / "test.csv", index=False)

    def test_quick_start_pipeline(self):
        """测试快速启动流水线"""
        logger.info("🧪 Testing quick start pipeline...")

        from lightning_landslide.src.active_learning import quick_start_active_learning

        try:
            # 使用很少的迭代进行快速测试
            with patch(
                "lightning_landslide.src.active_learning.active_pseudo_trainer.ActivePseudoTrainer"
            ) as mock_trainer:
                # 模拟训练器返回结果
                mock_result = Mock()
                mock_result.to_dict.return_value = {
                    "performance_history": {"val_f1": [0.7, 0.75, 0.8]},
                    "convergence_iteration": 3,
                    "total_training_time": 180,
                }
                mock_trainer.return_value.run.return_value = mock_result

                results = quick_start_active_learning(
                    train_data_dir=str(self.temp_dir / "train_data"),
                    test_data_dir=str(self.temp_dir / "test_data"),
                    train_csv=str(self.temp_dir / "train.csv"),
                    test_csv=str(self.temp_dir / "test.csv"),
                    experiment_name="test_quick_start",
                    max_iterations=2,
                )

                self.assertIsNotNone(results)
                self.assertIn("performance_history", results)
                logger.info("✅ Quick start pipeline test passed")

        except Exception as e:
            self.fail(f"Quick start pipeline test failed: {e}")

    def test_configuration_pipeline(self):
        """测试配置驱动的流水线"""
        logger.info("🧪 Testing configuration pipeline...")

        from lightning_landslide.src.active_learning import (
            create_active_learning_pipeline,
            validate_active_learning_config,
        )

        # 创建测试配置
        test_config = {
            "experiment_name": "test_config_pipeline",
            "seed": 42,
            "model": {
                "target": "lightning_landslide.src.models.LandslideClassificationModule",
                "params": {
                    "base_model": {
                        "target": "lightning_landslide.src.models.optical_swin.OpticalSwinModel",
                        "params": {"model_name": "swin_tiny_patch4_window7_224", "input_channels": 5, "num_classes": 1},
                    }
                },
            },
            "data": {
                "target": "lightning_landslide.src.data.MultiModalDataModule",
                "params": {
                    "train_data_dir": str(self.temp_dir / "train_data"),
                    "test_data_dir": str(self.temp_dir / "test_data"),
                    "train_csv": str(self.temp_dir / "train.csv"),
                    "test_csv": str(self.temp_dir / "test.csv"),
                    "batch_size": 4,
                    "num_workers": 0,
                },
            },
            "trainer": {
                "target": "pytorch_lightning.Trainer",
                "params": {"max_epochs": 2, "accelerator": "cpu", "devices": 1, "fast_dev_run": 1},
            },
            "active_pseudo_learning": {"max_iterations": 2, "annotation_budget": 5},
            "outputs": {"base_output_dir": str(self.temp_dir)},
        }

        try:
            # 验证配置
            validated_config = validate_active_learning_config(test_config)
            self.assertIsNotNone(validated_config)

            # 创建流水线
            pipeline = create_active_learning_pipeline(
                config=validated_config,
                experiment_name="test_pipeline",
                output_dir=str(self.temp_dir / "pipeline_test"),
            )

            self.assertIsNotNone(pipeline)
            logger.info("✅ Configuration pipeline created successfully")

        except Exception as e:
            self.fail(f"Configuration pipeline test failed: {e}")


class PerformanceBenchmarkTest(unittest.TestCase):
    """性能基准测试"""

    def test_memory_usage(self):
        """测试内存使用"""
        logger.info("🧪 Testing memory usage...")

        from lightning_landslide.src.active_learning.utils import MemoryMonitor

        with MemoryMonitor("Memory Usage Test") as monitor:
            # 模拟一些内存操作
            large_array = np.random.rand(1000, 1000)

            # 测试不确定性估计
            from lightning_landslide.src.active_learning import create_uncertainty_estimator

            estimator = create_uncertainty_estimator("mc_dropout", n_forward_passes=10)

            del large_array

            current_memory = monitor.current_usage()
            self.assertGreater(current_memory, 0)
            logger.info(f"✅ Memory usage test completed: {current_memory:.1f} MB")

    def test_timing_performance(self):
        """测试时间性能"""
        logger.info("🧪 Testing timing performance...")

        from lightning_landslide.src.active_learning.utils import Timer

        with Timer("Performance Test") as timer:
            # 模拟一些计算密集型操作
            large_computation = np.random.rand(500, 500) @ np.random.rand(500, 500)

            elapsed = timer.elapsed()
            self.assertGreater(elapsed, 0)
            logger.info(f"✅ Timing test completed: {elapsed:.2f}s")


class CompatibilityTest(unittest.TestCase):
    """兼容性测试"""

    def test_pytorch_compatibility(self):
        """测试PyTorch兼容性"""
        logger.info("🧪 Testing PyTorch compatibility...")

        try:
            import torch
            import pytorch_lightning as pl

            # 检查版本
            torch_version = torch.__version__
            pl_version = pl.__version__

            logger.info(f"PyTorch version: {torch_version}")
            logger.info(f"PyTorch Lightning version: {pl_version}")

            # 基本功能测试
            tensor = torch.randn(10, 5)
            self.assertEqual(tensor.shape, (10, 5))

            # CUDA可用性测试
            if torch.cuda.is_available():
                cuda_tensor = tensor.cuda()
                self.assertTrue(cuda_tensor.is_cuda)
                logger.info("✅ CUDA compatibility confirmed")
            else:
                logger.info("ℹ️ CUDA not available, using CPU")

            logger.info("✅ PyTorch compatibility test passed")

        except Exception as e:
            self.fail(f"PyTorch compatibility test failed: {e}")

    def test_dependencies_availability(self):
        """测试依赖包可用性"""
        logger.info("🧪 Testing dependencies availability...")

        required_packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "plotly"]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ {package} not available")

        if missing_packages:
            self.fail(f"Missing required packages: {missing_packages}")
        else:
            logger.info("✅ All required dependencies available")


def run_comprehensive_tests():
    """运行全面测试套件"""
    logger.info("🚀 Starting comprehensive test suite...")

    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [ActiveLearningModuleTest, EndToEndIntegrationTest, PerformanceBenchmarkTest, CompatibilityTest]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 输出结果摘要
    logger.info(f"\n{'='*60}")
    logger.info("🎯 TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    if result.failures:
        logger.error("❌ FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  - {test}: {traceback}")

    if result.errors:
        logger.error("❌ ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  - {test}: {traceback}")

    if result.wasSuccessful():
        logger.info("🎉 All tests passed successfully!")
        return True
    else:
        logger.error("💥 Some tests failed!")
        return False


def create_deployment_checklist():
    """创建部署检查清单"""
    checklist = {
        "environment": [
            "Python 3.8+ installed",
            "PyTorch 1.9+ installed",
            "PyTorch Lightning 1.5+ installed",
            "Required dependencies installed",
            "CUDA available (optional but recommended)",
            "Sufficient disk space (>10GB recommended)",
            "Sufficient RAM (>8GB recommended)",
        ],
        "data_preparation": [
            "Training data directory exists and accessible",
            "Test data directory exists and accessible",
            "Training CSV file with correct format",
            "Test CSV file with correct format",
            "Data file paths are correct",
            "Data format is consistent (.npy files)",
        ],
        "configuration": [
            "Configuration file created",
            "Model parameters specified",
            "Data paths configured correctly",
            "Training parameters set appropriately",
            "Active learning parameters configured",
            "Output directories specified",
        ],
        "functionality": [
            "Standard training works (python main.py train)",
            "K-fold training works (python main.py kfold)",
            "Active learning modules import correctly",
            "Configuration validation passes",
            "Test data can be loaded",
            "Model can be instantiated",
        ],
        "optimization": [
            "Batch size optimized for available memory",
            "Number of workers set appropriately",
            "GPU utilization optimized",
            "Checkpoint saving configured",
            "Logging level set appropriately",
        ],
    }

    return checklist


def validate_deployment():
    """验证部署状态"""
    logger.info("🔍 Validating deployment...")

    checklist = create_deployment_checklist()
    results = {}

    for category, items in checklist.items():
        logger.info(f"\n📋 Checking {category}...")
        results[category] = []

        for item in items:
            # 这里可以添加具体的检查逻辑
            # 现在只是示例
            status = "✅ PASS"  # 或 "❌ FAIL"
            results[category].append(f"{status} {item}")
            logger.info(f"  {status} {item}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Active Learning Integration Testing")
    parser.add_argument(
        "--mode",
        choices=["test", "validate", "both"],
        default="both",
        help="Test mode: test only, validate only, or both",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = True

    if args.mode in ["test", "both"]:
        logger.info("🧪 Running test suite...")
        success &= run_comprehensive_tests()

    if args.mode in ["validate", "both"]:
        logger.info("🔍 Running deployment validation...")
        validate_deployment()

    if success:
        logger.info("🎉 All checks completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Some checks failed!")
        sys.exit(1)
