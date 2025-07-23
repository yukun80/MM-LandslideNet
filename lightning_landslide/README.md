# MM-LandslideNet 新框架使用指南

## 🎯 概述

构建一个配置驱动的深度学习框架。这个框架的设计理念是"配置即代码"通过简单的YAML文件定义和运行复杂的深度学习实验。

### 🚀 核心优势

- **统一入口**：所有任务都通过 `python main.py` 执行
- **配置驱动**：添加新模型无需修改代码，只需编写配置文件
- **任务完整**：支持训练、测试、预测、验证四种任务模式
- **实验追踪**：每个实验的配置都会自动保存，确保可重现性
- **模块化设计**：组件可自由组合，就像搭积木一样灵活

---

## 📚 快速开始

### 1. 训练模型

```bash
# 使用光学基线配置训练模型
python main.py train configs/experiment/optical_baseline.yaml
```

这个命令会：

- 加载配置文件中定义的模型、数据和训练参数
- 自动创建输出目录
- 开始训练并保存最佳模型检查点
- 记录训练日志和指标

### 2. 测试已训练模型

```bash
# 测试训练好的模型
python main.py test configs/tasks/test_optical_baseline.yaml
```

这个命令会：

- 加载指定的检查点文件
- 在测试集上评估模型性能
- 生成详细的测试报告和可视化结果

### 3. 生成预测结果

```bash
# 对新数据进行预测
python main.py predict configs/tasks/predict_optical_baseline.yaml
```

这个命令会：

- 加载训练好的模型
- 对测试数据生成预测
- 保存竞赛提交格式的文件

### 4. 验证模型

```bash
# 在验证集上快速验证模型
python main.py validate configs/tasks/validate_optical_baseline.yaml
```

---

## 🛠️ 高级用法

### 命令行参数覆盖

您可以通过命令行覆盖配置文件中的任何参数：

```bash
# 修改训练轮数和批次大小
python main.py train configs/experiment/optical_baseline.yaml \
  --override trainer.params.max_epochs=50 \
  --override data.params.batch_size=128

# 使用不同的检查点进行测试
python main.py test configs/tasks/test_optical_baseline.yaml \
  --checkpoint outputs/checkpoints/best_model.ckpt

# 快速调试模式
python main.py train configs/experiment/optical_baseline.yaml \
  --override trainer.params.fast_dev_run=1
```

### 多个参数覆盖

```bash
python main.py train configs/experiment/optical_baseline.yaml \
  --override training.max_epochs=20 \
  --override data.batch_size=32 \
  --override model.params.dropout_rate=0.3 \
  --override compute.mixed_precision=false
```

---

## 📁 配置文件系统

### 配置文件结构

```
configs/
├── base/
│   └── base_config.yaml          # 基础配置模板
├── data/
│   └── optical_multimodal.yaml   # 数据配置
├── models/
│   └── optical_swin.yaml         # 模型配置
├── experiment/
│   └── optical_baseline.yaml     # 完整实验配置
└── tasks/
    ├── test_optical_baseline.yaml
    ├── predict_optical_baseline.yaml
    └── validate_optical_baseline.yaml
```

### 创建新实验

要创建新实验，只需复制并修改配置文件：

```bash
# 复制基线配置
cp configs/experiment/optical_baseline.yaml configs/experiment/my_experiment.yaml

# 编辑配置文件
vim configs/experiment/my_experiment.yaml

# 运行新实验
python main.py train configs/experiment/my_experiment.yaml
```

---

## 🔧 配置文件详解

### 基本实验配置结构

```yaml
# 实验元信息
experiment_name: "my_awesome_experiment"
description: "Testing new augmentation strategies"
version: "1.0.0"
tags: ["optical", "augmentation", "baseline"]

# 全局设置
seed: 42
log_level: "INFO"

# 模型配置
model:
  target: lightning_landslide.src.models.LandslideClassificationModule
  params:
    base_model:
      target: lightning_landslide.src.models.optical_swin.OpticalSwinModel
      params:
        model_name: "swin_tiny_patch4_window7_224"
        input_channels: 5
        num_classes: 1

# 数据配置
data:
  target: lightning_landslide.src.data.MultiModalDataModule
  params:
    batch_size: 64
    train_data_dir: "dataset/train_data"
    # ... 其他参数

# 训练器配置
trainer:
  target: pytorch_lightning.Trainer
  params:
    max_epochs: 100
    accelerator: "auto"
    devices: "auto"
```

### 添加新模型

要添加新的模型架构，只需：

1. 实现模型类（继承BaseModel）
2. 创建模型配置文件
3. 在实验配置中引用

```yaml
# 新模型配置示例
model:
  target: lightning_landslide.src.models.MyNewModel
  params:
    architecture: "efficientnet_b4"
    input_channels: 13  # 使用全部通道
    use_attention: true
    custom_param: "my_value"
```

---

## 📊 输出文件说明

### 训练输出

训练完成后，您会在输出目录中找到：

```
outputs/
├── checkpoints/
│   ├── best-epoch=XX-val_f1=X.XXXX.ckpt  # 最佳模型
│   └── last.ckpt                          # 最后一个epoch的模型
├── logs/
│   ├── tensorboard/                       # TensorBoard日志
│   └── config.yaml                        # 保存的配置文件
└── predictions/
    └── training_predictions.csv           # 训练过程中的预测
```

### 测试输出

```
outputs/test_results/
├── test_results_EXPERIMENT_NAME_TIMESTAMP.json  # 详细测试结果
├── confusion_matrix.png                         # 混淆矩阵
├── roc_curve.png                                # ROC曲线
└── classification_report.txt                    # 分类报告
```

### 预测输出

```
outputs/predictions/
├── predictions_EXPERIMENT_NAME_TIMESTAMP.json   # 完整预测信息
├── predictions_EXPERIMENT_NAME_TIMESTAMP.csv    # CSV格式预测
└── submission_EXPERIMENT_NAME_TIMESTAMP.csv     # 竞赛提交格式
```

---

## 🎛️ 任务特定配置

### 训练任务配置重点

- 设置合适的 `max_epochs` 和早停参数
- 配置数据增强策略
- 选择损失函数处理类别不平衡
- 设置学习率调度器

### 测试任务配置重点

- 指定正确的 `checkpoint_path`
- 启用测试时增强 (TTA)
- 配置详细的指标计算
- 设置结果可视化选项

### 预测任务配置重点

- 优化批次大小提高推理速度
- 配置输出格式（JSON/CSV/提交格式）
- 设置预测后处理参数
- 启用质量控制检查

---

## 🐛 故障排除

### 常见问题及解决方案

#### 1. 找不到模型类

```
ImportError: cannot import name 'MyModel' from 'src.models'
```

**解决方案：**

- 检查模型类是否在 `src/models/__init__.py` 中正确导入
- 确认配置文件中的 `target` 路径正确

#### 2. 配置文件格式错误

```
yaml.scanner.ScannerError: while scanning for the next token
```

**解决方案：**

- 检查YAML文件的缩进是否正确（使用空格，不要使用制表符）
- 验证所有的引号和括号都正确配对

#### 3. 检查点文件未找到

```
FileNotFoundError: Checkpoint not found: path/to/checkpoint.ckpt
```

**解决方案：**

- 确认检查点文件路径正确
- 检查训练是否成功完成并保存了检查点

#### 4. GPU内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案：**

- 减小配置文件中的 `batch_size`
- 启用混合精度训练：`precision: "16-mixed"`
- 使用梯度累积：`accumulate_grad_batches: 2`

---

## 🚀 最佳实践

### 1. 实验命名规范

使用描述性的实验名称：

```yaml
experiment_name: "optical_swin_tiny_focal_loss_aug_v2"
description: "Swin Tiny with focal loss and enhanced augmentation strategy v2"
```

### 2. 版本控制

为重要实验创建Git标签：

```bash
git tag -a "exp-optical-baseline-v1.0" -m "Optical baseline experiment v1.0"
git push origin "exp-optical-baseline-v1.0"
```

### 3. 配置文件组织

- 保持配置文件简洁明了
- 使用注释解释重要参数
- 将相似的实验放在同一目录下

### 4. 超参数调优

使用命令行覆盖进行快速超参数探索：

```bash
for lr in 1e-4 5e-5 2e-4; do
  python main.py train configs/experiment/optical_baseline.yaml \
    --override optimizer.params.lr=$lr \
    --override experiment_name="lr_search_$lr"
done
```

### 5. 结果分析

- 定期备份重要的检查点文件
- 使用TensorBoard比较不同实验的结果
- 保存配置文件的副本用于重要实验

---

## 📈 性能优化建议

### 训练加速

1. **使用混合精度**：

   ```yaml
   trainer:
     params:
       precision: "16-mixed"
   ```
2. **优化数据加载**：

   ```yaml
   data:
     params:
       num_workers: 8
       pin_memory: true
   ```
3. **梯度累积**（当GPU内存有限时）：

   ```yaml
   trainer:
     params:
       accumulate_grad_batches: 4
   ```

### 推理加速

1. **增大批次大小**：

   ```yaml
   data:
     params:
       batch_size: 256  # 推理时可以更大
   ```
2. **禁用不必要的组件**：

   ```yaml
   trainer:
     params:
       logger: false
       enable_checkpointing: false
   ```

---

### 记住关键命令

```bash
# 训练
python main.py train configs/experiment/optical_baseline.yaml

# 测试
python main.py test configs/tasks/test_optical_baseline.yaml

# 预测
python main.py predict configs/tasks/predict_optical_baseline.yaml

# 验证
python main.py validate configs/tasks/validate_optical_baseline.yaml
```

### 下一步

1. 运行 `python comprehensive_test.py` 验证框架功能
2. 使用您的真实数据配置文件
3. 开始您的滑坡检测实验！
