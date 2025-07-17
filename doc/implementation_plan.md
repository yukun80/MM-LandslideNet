# 模型重构实施计划：非对称双主干融合模型

**作者**: Gemini Pro
**日期**: 2025-07-16
**状态**: 待实施

## 1. 总体目标

将当前基于“三主干+后期融合”的 `MM-InternImage-TNF`模型，重构为一个更高效、更强大、更符合多模态遥感数据特性的**“非对称双主干、分层交叉融合”**新架构。此举旨在从根本上解决当前模型存在的“信息瓶颈”、参数冗余和训练不稳定等问题。

新架构的核心思想是：以一个强大的InternImage模型作为处理主要信息（光学数据）的**主干**，以一个轻量级的CNN作为处理辅助信息（SAR数据）的**辅干**，并在主干的多个中间阶段，通过交叉注意力机制智能地将SAR特征融入光学特征中。

## 2. 实施阶段与步骤

### 阶段一：数据管道重构 (`mm_intern_image_src/dataset.py`)

**目标**：修改 `Dataset`类，使其为新架构提供正确格式的数据：一个5通道的光学张量和一个8通道的SAR张量。

- **步骤 1.1：合并SAR通道**

  - **任务**: 修改 `_split_modalities`函数。移除 `sar_diff`的独立处理，将 `sar`（4通道）和 `sar_diff`（4通道）在通道维度上拼接（concatenate）成一个单一的8通道 `sar_combined`张量。
  - **验收标准**: 函数最终应返回 `optical` (H, W, 5) 和 `sar_combined` (H, W, 8) 两个Numpy数组。
- **步骤 1.2：调整归一化统计**

  - **任务**: 修改 `_extract_normalization_stats`函数。将 `sar`和 `sar_diff`的统计数据合并到一个新的键（例如 `sar_combined`）下。确保 `mean`和 `std`列表的顺序与上一步拼接的通道顺序一致。
  - **验收标准**: `self.stats`字典中应包含 `'sar_combined'`键，其 `mean`和 `std`均为8个元素。
- **步骤 1.3：更新归一化逻辑**

  - **任务**: 修改 `_normalize_modality`函数。确保它可以处理名为 `sar_combined`的新模态，并对其8个通道应用正确的Z-score归一化。
  - **验收标准**: 函数能正确处理 `modality='sar_combined'`的调用。
- **步骤 1.4：更新 `__getitem__`方法**

  - **任务**: 修改 `__getitem__`的返回字典。使其包含两个键：`"optical"`和 `"sar"`，分别对应5通道的光学张量和8通道的SAR张量。移除 `"sar_diff"`键。
  - **验收标准**: DataLoader产出的每个batch应为包含 `'optical'`和 `'sar'`键的字典。

---

### 阶段二：模型架构重构 (`mm_intern_image_src/models.py`)

**目标**：实现全新的 `AsymmetricFusionModel`，替换掉旧的 `MMInternImageTNF`。

- **步骤 2.1：定义轻量级SAR辅干 (`LightweightSARCNN`)**

  - **任务**: 创建一个新的 `nn.Module`子类，名为 `LightweightSARCNN`。
  - **输入**: 8通道的SAR数据。
  - **结构**: 构建一个类似FPN（特征金字塔网络）的结构。它应包含多个卷积块，并能在**不同尺度**上输出特征图，以匹配InternImage主干的4个Stage。
  - **输出**: 一个包含4个特征图的列表或元组，每个特征图的尺寸与InternImage对应Stage的输出尺寸相匹配。
- **步骤 2.2：定义交叉融合模块 (`CrossFusionBlock`)**

  - **任务**: 创建一个新的 `nn.Module`子类，名为 `CrossFusionBlock`。
  - **输入**: `optical_features` (来自光学主干，作为Query) 和 `sar_features` (来自SAR辅干，作为Key和Value)。
  - **结构**:
    1. 实现标准的多头交叉注意力（Multi-Head Cross-Attention）。
    2. 实现自适应门控：并行地，将 `optical_features`通过一个小的卷积层和Sigmoid函数，生成一个门控权重（gate）。
    3. 将交叉注意力的输出乘以这个门控权重，然后通过残差连接加回到原始的 `optical_features`上。
- **步骤 2.3：构建最终模型 (`AsymmetricFusionModel`)**

  - **任务**: 重命名或替换 `MMInternImageTNF`为一个新的 `AsymmetricFusionModel`类。
  - **结构**:
    1. **主干**: 创建**一个**InternImage实例作为 `optical_backbone`。修改其 `patch_embed`层以接受5通道输入，并实现智能权重初始化。
    2. **辅干**: 创建一个 `LightweightSARCNN`实例作为 `sar_backbone`。
    3. **融合模块**: 创建4个 `CrossFusionBlock`实例，每个对应一个融合阶段。
  - **`forward`方法逻辑**:
    1. SAR数据 -> `sar_backbone` -> 获得多尺度SAR特征图。
    2. **逐阶段**执行 `optical_backbone`的前向传播。
    3. 在每个Stage后，使用相应的 `CrossFusionBlock`将对应的SAR特征图融合进来。
    4. 最终输出送入分类头。

---

### 阶段三：配置与清理

**目标**：调整项目配置，移除旧代码，并进行最终验证。

- **步骤 3.1：更新配置文件 (`mm_intern_image_src/config.py`)**

  - **任务**: 移除与旧的“三主干”和 `TNFFusionBlock`相关的配置项。添加定义 `LightweightSARCNN`和 `CrossFusionBlock`结构的新配置项。更新 `MODEL_NAME`。
- **步骤 3.2：清理旧代码**

  - **任务**: 在 `models.py`中，安全地删除旧的 `MMInternImageTNF`、`InternImageBackbone`和 `TNFFusionBlock`类。确保 `train.py`和 `predict.py`中的模型创建调用已更新。
- **步骤 3.3：验证**

  - **任务**: 在 `models.py`的 `if __name__ == "__main__"`块中，编写一个测试脚本。创建一个伪造的数据批次（包含 `"optical"`和 `"sar"`张量），执行一次完整的前向传播，并使用 `assert`和 `print`检查所有中间步骤和最终输出的张量形状，确保整个模型的维度匹配正确无误。

## 3. 预期收益

- **性能提升**: 新架构更符合多模态学习范式，有望取得更高的F1分数。
- **稳定性增强**: 解决了信息瓶颈和数值不稳定风险，`Loss=NaN`问题将不复存在。
- **效率提升**: 参数量大幅减少，训练和推理速度将得到提升。
- **可维护性提高**: 代码结构更清晰、更简洁。
