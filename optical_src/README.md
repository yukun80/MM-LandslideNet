# MM-LandslideNet 光学基线算法实现

## 📖 项目概述

`optical_src` 包实现了滑坡检测任务的**光学数据基线模型**，这是整个竞赛策略中的**第二阶段**核心组件。

### 🎯 核心目标

- **专注光学数据**：仅使用Sentinel-2光学遥感数据进行滑坡检测
- **基准模型**：为后续复杂的多模态模型提供性能基准
- **算法验证**：验证光学数据在滑坡检测任务中的有效性
- **技术铺垫**：为Phase 3多模态融合奠定技术基础

### 🏗️ 架构定位

```
Phase 1: 数据质量分析与预处理
    ↓
Phase 2: 光学基线模型 (optical_src) ← 当前阶段
    ↓  
Phase 3: 多模态融合模型
    ↓
Phase 4: 模型集成与优化
```

---

## 🔬 算法实现逻辑

### 🧠 模型架构

#### **BaselineOpticalModel** - 单分支分类器

```python
输入 (5通道) → Swin Transformer主干 → 分类头 → 二元分类输出
    64×64×5        特征提取(768维)      概率预测     滑坡/非滑坡
```

#### **主干网络**: Swin Transformer

- **模型选择**: `swin_tiny_patch4_window7_224`
- **特征维度**: 768维全局特征表示
- **预训练权重**: ImageNet预训练，提供强大的视觉特征提取能力

#### **输入层改造策略**

```python
原始: 3通道 (RGB) → 修改后: 5通道 (R,G,B,NIR,NDVI)

权重初始化策略:
- 前3通道 (RGB): 直接复制预训练权重
- 第4通道 (NIR): RGB权重平均初始化  
- 第5通道 (NDVI): Red+NIR组合权重初始化
```

#### **分类头设计**

```python
nn.Sequential(
    nn.LayerNorm(768),     # 特征归一化
    nn.Dropout(0.2),       # 防止过拟合
    nn.Linear(768, 1)      # 二元分类输出
)
```

### 📊 输入数据组成

#### **5通道输入张量** `(Batch, 5, 64, 64)`

| 通道           | 数据源     | 描述           | 波段范围        |
| -------------- | ---------- | -------------- | --------------- |
| **R**    | Sentinel-2 | 红光波段       | 650-680 nm      |
| **G**    | Sentinel-2 | 绿光波段       | 540-570 nm      |
| **B**    | Sentinel-2 | 蓝光波段       | 450-490 nm      |
| **NIR**  | Sentinel-2 | 近红外波段     | 780-900 nm      |
| **NDVI** | 计算得出   | 归一化植被指数 | (NIR-R)/(NIR+R) |

#### **NDVI计算公式**

```python
NDVI = (NIR - Red) / (NIR + Red + ε)
其中 ε = 1e-8 (避免除零错误)
```

---

## 🗃️ 数据集处理

### 📁 数据加载流程

#### **原始数据格式**

- **文件类型**: `.npy` 文件
- **数据维度**: `(64, 64, 12)` - 12通道遥感数据
- **存储路径**: `dataset/train_data/` 和 `dataset/test_data/`

#### **通道提取与处理**

```python
# 从12通道数据中提取光学通道
原始数据 (64×64×12) → 光学数据 (64×64×4) → 5通道输入 (64×64×5)
                        前4通道: R,G,B,NIR    添加NDVI通道
```

### 🧹 数据质量控制

#### **低质量图像过滤**

- **数据源**: `dataset/data_check/exclude_ids.json`
- **来源**: Phase 1数据质量分析阶段识别
- **过滤标准**:
  - 云覆盖过多
  - 数据缺失严重
  - 噪声水平过高
- **影响**: 从6,432个样本过滤后的样本

#### **数据划分**

```
过滤后可用训练样本: 6,432个
验证样本: 1,287个 (20.0%)
可用训练样本: 5,145个 (80.0%)
```

### 📏 数据归一化

#### **通道级归一化**

- **统计文件**: `dataset/data_check/channel_stats.json`
- **归一化方法**: Z-score标准化
- **公式**: `(pixel - mean) / (std + ε)`

#### **各通道统计参数**

```python
光学通道 (R,G,B,NIR):
- 使用预计算的均值和标准差
- 确保不同传感器数据的一致性

NDVI通道:
- 范围限制: [-1, 1]
- 使用np.clip()确保数值有效性
```

### 🔄 数据增强策略

#### **几何变换技术**

```python
训练时增强:
- 水平翻转: 50% 概率
- 垂直翻转: 50% 概率  
- 随机90度旋转: 50% 概率

验证时增强:
- 仅应用ToTensorV2转换 (无几何变换)
```

#### **增强实现**

```python
# 使用Albumentations库
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5), 
    A.RandomRotate90(p=0.5),
    ToTensorV2()  # HWC → CHW + 归一化
])
```

### 📊 数据集划分

#### **分层划分策略**

```python
训练集: 80% (4,116个样本)
验证集: 20% (1,029个样本)

分层方法: 按类别比例分割
随机种子: 42 (确保可重现性)
```

#### **类别分布维护**

- 保持训练集和验证集中的正负样本比例一致
- 避免验证偏差
- 确保评估结果的可靠性

### ⚖️ 类别不平衡处理

#### **加权随机采样器**

```python
# 计算类别权重
class_weights = 1.0 / class_counts
sample_weights = class_weights[sample_labels]

# 应用WeightedRandomSampler
WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True  # 允许重复采样
)
```

#### **不平衡统计**

```
负样本 (非滑坡): ~90%
正样本 (滑坡): ~10%
权重比例: 约1:9
```

---

## 🎯 训练策略

### 💥 损失函数

#### **BCEWithLogitsLoss + 正样本权重**

```python
# 自动计算正样本权重
pos_weight = neg_count / pos_count  # ≈ 9.0

# 损失函数配置
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

#### **优势特点**

- **数值稳定**: 内置sigmoid避免数值溢出
- **类别平衡**: pos_weight自动调节正负样本重要性
- **梯度优化**: 更好的梯度反向传播特性

### 🚀 优化器配置

#### **AdamW优化器**

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,           # 学习率
    weight_decay=1e-4  # L2正则化
)
```

#### **参数说明**

- **自适应学习率**: Adam算法的改进版本
- **权重衰减**: 有效防止过拟合
- **动量机制**: 加速收敛过程

### 📈 学习率调度

#### **余弦退火调度器**

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,        # 周期长度
    eta_min=1e-6     # 最小学习率
)
```

#### **调度特点**

- **平滑衰减**: 避免学习率突变
- **周期性**: 有助于跳出局部最优
- **最终收敛**: 确保训练后期稳定

### ⏹️ 早停机制

#### **监控指标与触发条件**

```python
监控指标: 验证集F1-score
耐心期: 10个epoch
最小改进: 1e-4

触发条件: F1-score连续10个epoch无显著提升
```

#### **早停优势**

- **防止过拟合**: 在最佳性能点停止训练
- **节省资源**: 避免无效的长时间训练
- **模型选择**: 自动保存最佳性能模型

### 🔁 可重现性保证

#### **随机种子设置**

```python
# 设置所有随机数生成器种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# CUDA确定性行为
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### ⚡ 混合精度训练

#### **自动混合精度 (AMP)**

```python
# 配置启用
MIXED_PRECISION = True

# 优势特点
- 训练速度提升: ~1.5-2x
- 显存使用减少: ~40%
- 精度基本无损失
```

### 📊 日志与监控

#### **多层次日志系统**

```python
1. 控制台输出: 实时训练进度
2. 文件日志: logs/optical_baseline/training.log
3. TensorBoard: 图形化指标可视化
```

#### **监控指标**

- **训练指标**: Loss, Accuracy, F1-score, Precision, Recall
- **验证指标**: 同训练指标 + AUC
- **系统指标**: 学习率变化, 训练时间, GPU使用率

#### **TensorBoard可视化**

```bash
# 启动TensorBoard
tensorboard --logdir=logs/optical_baseline

# 访问地址
http://localhost:6006
```

---

## ⚙️ 配置管理

### 🏗️ 模块化配置架构

#### **继承结构**

```python
configs/config.py (基础配置)
        ↓ 继承
optical_src/config.py (光学基线专用配置)
```

#### **配置特点**

- **参数隔离**: 光学基线专用参数独立管理
- **重写机制**: 可覆盖基础配置中的通用参数
- **扩展性**: 易于添加新的光学模型变体

### 📋 关键配置参数

#### **模型配置**

```python
MODEL_NAME = "swin_tiny_patch4_window7_224"
NUM_CLASSES = 1                    # 二元分类
DROPOUT_RATE = 0.2                 # Dropout比例
INPUT_CHANNELS = 5                 # 输入通道数
TARGET_SIZE = 224                  # Swin Transformer输入尺寸
```

#### **训练配置**

```python
NUM_EPOCHS = 50                    # 最大训练轮数
BATCH_SIZE = 32                    # 批次大小
LEARNING_RATE = 1e-4               # 初始学习率
WEIGHT_DECAY = 1e-4                # 权重衰减
```

#### **数据配置**

```python
HORIZONTAL_FLIP_PROB = 0.5         # 水平翻转概率
VERTICAL_FLIP_PROB = 0.5           # 垂直翻转概率
ROTATION_PROB = 0.5                # 旋转概率
```

#### **早停配置**

```python
EARLY_STOPPING = True              # 启用早停
PATIENCE = 10                      # 耐心期
MIN_DELTA = 1e-4                   # 最小改进阈值
```

### 🎛️ 模型变体支持

#### **预定义变体**

```python
MODEL_VARIANTS = {
    "swin_tiny": {
        "model_name": "swin_tiny_patch4_window7_224",
        "feature_dim": 768,
        "pretrained": True
    },
    "swin_small": {
        "model_name": "swin_small_patch4_window7_224", 
        "feature_dim": 768,
        "pretrained": True
    },
    "swin_base": {
        "model_name": "swin_base_patch4_window7_224",
        "feature_dim": 1024,
        "pretrained": True
    }
}
```

---

## 🚀 运行算法

### 📋 前置条件

#### **Phase 1 完成确认**

确保以下文件存在且完整：

- ✅ `dataset/data_check/exclude_ids.json` - 低质量图像列表
- ✅ `dataset/data_check/channel_stats.json` - 通道统计信息
- ✅ `dataset/Train.csv` - 训练标签文件
- ✅ `dataset/train_data/*.npy` - 训练图像数据

### 💻 执行命令

#### **方式1: 启动脚本 (推荐)**

```bash
# 从项目根目录执行
python run_optical_training.py
```

#### **方式2: 模块运行**

```bash
# 使用Python模块方式
python -m optical_src.train
```

#### **方式3: 直接运行**

```bash
# 直接运行训练文件
python optical_src/train.py
```

#### **后台运行 (Linux/macOS)**

```bash
# 后台运行并保存日志
nohup python run_optical_training.py > training.log 2>&1 &

# 查看实时日志
tail -f training.log
```

### 📈 预期输出

#### **启动阶段** (前30秒)

```
============================================================
MM-LandslideNet Optical Baseline Training Launcher
============================================================
Project root: /path/to/MM-LandslideNet
Starting training...
============================================================

2024-XX-XX XX:XX:XX - optical_baseline - INFO - Logging setup complete. Level: INFO
2024-XX-XX XX:XX:XX - optical_baseline.dataset - INFO - Loaded 1287 excluded image IDs
2024-XX-XX XX:XX:XX - optical_baseline.dataset - INFO - Loaded channel statistics for normalization
2024-XX-XX XX:XX:XX - optical_baseline.dataset - INFO - Filtered training data: 6432 -> 5145 samples
2024-XX-XX XX:XX:XX - optical_baseline.model - INFO - Created swin_tiny_patch4_window7_224 with 1 output classes
2024-XX-XX XX:XX:XX - optical_baseline.model - INFO - Modified input layer to accept 5 channels (R, G, B, NIR, NDVI)
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Using CUDA device: NVIDIA GeForce RTX 3080
```

#### **训练过程** (每个epoch)

```
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Starting training...
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Training for 50 epochs

Epoch 0:
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Epoch 0, Batch 0/160, Loss: 1.2345
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Epoch 0, Batch 50/160, Loss: 0.8901
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Epoch 0, Batch 100/160, Loss: 0.6789
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Epoch 0, Batch 150/160, Loss: 0.5432

Epoch 0 Results:
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO -   Train - Loss: 0.7234, F1: 0.3456, Acc: 0.8901
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO -   Val   - Loss: 0.6789, F1: 0.4567, Acc: 0.9012
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - New best model saved with F1: 0.4567
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Epoch 0 completed in 2m 34.5s
```

#### **目录结构生成**

训练过程中将自动创建以下目录结构：

```
MM-LandslideNet/
├── logs/optical_baseline/
│   ├── run_1234567890/          # TensorBoard日志
│   └── training.log             # 训练日志文件
├── outputs/
│   ├── checkpoints/optical_baseline/
│   │   ├── best_model.pth       # 最佳模型
│   │   ├── epoch_5.pth          # 定期保存
│   │   └── epoch_10.pth
│   ├── optical_baseline/        # 输出结果
│   ├── train_split.csv          # 训练集划分
│   └── val_split.csv            # 验证集划分
```

#### **训练完成输出**

```
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Training completed in 1h 23m 45.6s
2024-XX-XX XX:XX:XX - optical_baseline.train - INFO - Best validation F1: 0.7234 at epoch 28

🎉 训练完成总结:
- 最佳F1分数: 0.7234
- 最佳模型保存: outputs/checkpoints/optical_baseline/best_model.pth
- 训练日志: logs/optical_baseline/training.log
- TensorBoard: tensorboard --logdir=logs/optical_baseline
```

### 🔍 结果检查

#### **性能指标文件**

```bash
# 检查TensorBoard日志
tensorboard --logdir=logs/optical_baseline

# 查看训练日志
cat logs/optical_baseline/training.log

# 检查模型文件
ls -la outputs/checkpoints/optical_baseline/
```

#### **预期性能范围**

```
基线性能目标:
- 验证准确率: 85%+
- 验证F1分数: 0.60+
- 验证AUC: 0.80+

训练时间估算:
- RTX 3080: ~1.5小时 (50 epochs)
- RTX 4090: ~1小时 (50 epochs)
- GTX 1060: ~4小时 (50 epochs)
```

---

## 📝 使用示例

### 🔧 自定义配置训练

```python
from optical_src.config import OpticalBaselineConfig
from optical_src.train import Trainer

# 创建自定义配置
config = OpticalBaselineConfig()
config.NUM_EPOCHS = 100          # 增加训练轮数
config.BATCH_SIZE = 64           # 增大批次
config.LEARNING_RATE = 5e-5      # 降低学习率

# 启动训练
trainer = Trainer(config)
trainer.train()
```

### 🎯 模型推理示例

```python
from optical_src.model import BaselineOpticalModel
from optical_src.config import OpticalBaselineConfig
import torch

# 加载训练好的模型
config = OpticalBaselineConfig()
model = BaselineOpticalModel.from_config(config)

# 加载权重
checkpoint = torch.load('outputs/checkpoints/optical_baseline/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理示例
with torch.no_grad():
    # input_data: (batch_size, 5, 64, 64)
    outputs = model(input_data)
    probabilities = torch.sigmoid(outputs)
    predictions = (probabilities > 0.5).float()
```

---

## 🎯 性能基准

### 📊 基线性能目标

| 指标                 | 目标值 | 优秀值 |
| -------------------- | ------ | ------ |
| **验证准确率** | ≥85%  | ≥90%  |
| **验证F1分数** | ≥0.60 | ≥0.70 |
| **验证AUC**    | ≥0.80 | ≥0.85 |
| **训练时间**   | <2小时 | <1小时 |

### 🚀 优化建议

#### **进一步提升性能**

1. **模型升级**: 尝试 `swin_small` 或 `swin_base` 变体
2. **数据增强**: 添加颜色变换、噪声注入等技术
3. **损失函数**: 尝试Focal Loss或其他不平衡处理方法
4. **集成学习**: 训练多个模型进行投票预测

#### **训练加速**

1. **混合精度**: 确保启用AMP
2. **批次大小**: 根据GPU内存适当增大
3. **数据加载**: 增加 `num_workers`并行度
4. **模型选择**: 使用更小的Swin变体进行快速实验

---

## 📞 技术支持

### ❓ 常见问题

#### **Q1: 训练过程中GPU内存不足怎么办？**

```python
# 解决方案
1. 减小批次大小: config.BATCH_SIZE = 16
2. 启用梯度累积: 在train.py中实现
3. 使用更小的模型: variant="swin_tiny"
```

#### **Q2: 训练收敛速度太慢？**

```python
# 解决方案  
1. 增大学习率: config.LEARNING_RATE = 2e-4
2. 减小权重衰减: config.WEIGHT_DECAY = 1e-5
3. 调整调度器: config.SCHEDULER_T_MAX = 30
```

#### **Q3: 验证性能不理想？**

```python
# 分析方法
1. 检查数据质量: 重新审视exclude_ids.json
2. 调整类别权重: 手动设置pos_weight
3. 增强数据增强: 添加更多变换技术
```

### 📧 联系方式

- **项目文档**: 参考项目根目录README.md
- **代码问题**: 检查各模块的docstring说明
- **性能调优**: 参考configs/config.py基础配置

---

**🎉 祝您训练顺利！期待光学基线模型为后续多模态融合提供强有力的性能基准！**
