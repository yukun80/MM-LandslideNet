# **实施计划：赢得滑坡检测挑战赛 (冠军优化版)**

**最终目标:** 竞赛第一名。
**核心理念:** 结合SOTA分层融合架构、动态模态加权、双重不确定性量化以及极限性能冲刺策略，构建一个在F1分数和创新性报告上都具备压倒性优势的解决方案。

---

### **Phase 1: 信息内容评估与数据清洗 (预计用时: 2天)**

*   **目标:** 识别并剔除信息量过低（如全噪声）的训练数据，然后基于清洗后的高质量数据集，计算统计特性并制定预处理流程。
*   **任务:**
    1.  **基于标准差的数据质量评估 (`scripts/data_quality_assessment.py`):**
        *   **创建新脚本:** 编写一个名为 `data_quality_assessment.py` 的新脚本。
        *   **定义质量标准:** 我们的核心假设是，缺乏有效信息的图像，其像素值标准差会显著偏低。此脚本将遍历所有训练样本，计算**每个样本在所有12个通道上的平均标准差**。
        *   **生成质量分数:** 将每个样本的 `image_id` 及其计算出的“平均标准差”保存到 `outputs/image_quality_scores.csv`。这为我们提供了一个量化的信息内容指标。
    2.  **阈值确定与坏数据排除 (`scripts/analyze_quality.py`):**
        *   **创建分析脚本:** 编写一个名为 `analyze_quality.py` 的新脚本（或使用Jupyter Notebook）。
        *   **分析分布:** 加载 `image_quality_scores.csv`，绘制所有样本“平均标准差”的直方图。通过观察这个分布，我们可以识别出代表“坏数据”的低分 outlier 群体，并据此确定一个合理的排除阈值。
        *   **生成排除列表:** 根据确定的阈值，筛选出所有低质量样本的 `image_id`，并将其保存到 `outputs/exclude_ids.json`。
    3.  **最终统计与预处理策略 (`scripts/data_analysis.py`):**
        *   **修改现有脚本:** 更新 `data_analysis.py`，使其首先加载 `outputs/exclude_ids.json`。
        *   **计算清洗后数据的统计:** 脚本将遍历**排除了坏数据之后**的高质量训练集，计算并输出每个通道的全局均值和标准差，更新到 `outputs/channel_stats.json`。
        *   **数据加载器规划:** 最终的 `Dataset` 类将使用 `exclude_ids.json` 作为黑名单，并使用 `channel_stats.json` 对高质量数据进行标准化处理。

---

### **Phase 2: 高级基线模型 (Attention-Fused Classifier) (预计用时: 2天)**

* **目标:** 快速验证一个**具备注意力融合机制的端到端分类流程**，并为SOTA模型提供更强的性能参照。
* **任务:**
  1. **数据加载器 (`src/data/dataset.py`):**
     * 创建PyTorch `Dataset` 类，能够加载 `.npy`文件，并应用预处理（如归一化、NDVI计算）。
     * 实现数据增强 (`albumentations`)：对训练数据应用随机水平/垂直翻转、90度旋转。
     * 实现**加权随机采样 (`WeightedRandomSampler`)**，根据Phase 1中计算的样本比例，对少数类（滑坡）进行过采样。
  2. **模型架构 (`src/models/advanced_baseline.py`):**
     * **骨干网络:** 双分支EfficientNet。
     * **任务对齐:** 放弃U-Net解码器。直接在两个EfficientNet编码器的最终特征图后接一个全局平均池化（Global Average Pooling），得到两个模态的特征向量。
     * **核心升级 - 注意力融合模块:**
       * 将两个模态的特征向量拼接。
       * 送入一个轻量级的**交叉注意力模块**（例如，一个标准Transformer编码器层），让两个向量进行信息交互。
       * 将注意力模块的输出送入最终的MLP分类头进行预测。
  3. **训练引擎 (`src/engine/train.py`):**
     * **损失函数:** 从 **Focal Loss + Dice Loss** 的组合开始。**新增实验:** 尝试在损失函数中直接为正负样本设置权重（`pos_weight`），作为加权采样的补充或替代方案。
     * **优化器:** AdamW。
     * **学习率调度:** Cosine Annealing with Warmup。
     * **验证:** 实现标准的验证流程，每个epoch结束后计算验证集上的F1分数，并保存F1分数最高的模型权重。

---

### **Phase 3: 冠军模型实现 (Hierarchical Cross-Fusion Transformer) 与旗舰级可信赖AI (预计用时: 5-6天)**

* **目标:** 构建一个具备**分层多尺度融合**能力的SOTA Transformer模型，并集成**动态模态加权**和**双重不确定性量化**功能。
* **核心架构:** **Hierarchical Cross-Fusion Transformer (HCF-Transformer)**
* **任务:**
  1. **模型架构 (`src/models/hcf_transformer.py`):**
     * **双Swin Transformer骨干:** Optical: `swin_base_patch4_window7_224`, SAR: `swin_tiny_patch4_window7_224`。
     * **核心升级 - 分层交叉融合 (Hierarchical Cross-Fusion):**
       * 在Swin Transformer的第2、3、4阶段的输出端，均插入一个轻量级的双向交叉注意力融合模块。
       * 下一阶段的输入，是上一阶段融合、增强后的特征。
     * **旗舰级创新 - 动态模态门控 (Dynamic Modality Gating):**
       * 从光学分支的输入中，提取一个简单的云量/质量指标。
       * 将此指标输入一个微型神经网络（门控网络），**动态地生成一个权重 `α` (范围0-1)**。
       * 在最终的分类头之前，进行加权融合：`fused_feature = α * optical_feature + (1 - α) * sar_feature`。
     * **双重不确定性量化:**
       * 在最终的MLP分类头中，额外预测一个方差 `σ^2`，用于量化**偶然不确定性**。
       * 保留Dropout并使用MC Dropout来量化**认知不确定性**。
  2. **可解释性 (XAI) (`src/utils/xai.py`):**
     * 可视化分层交叉注意力图。
     * 可视化动态门控权重 `α`。
     * 可视化Grad-CAM。

---

### **Phase 4: 极限冲刺、集成与冠军报告 (预计用时: 2-3天)**

* **目标:** 通过集成学习和后处理技术，将模型性能推向极限，并撰写一份无可辩驳的技术报告。
* **任务:**
  1. **集成学习 (Ensembling):**
     * 对5折交叉验证得到的5个HCF-Transformer模型进行**加权平均**。
  2. **后处理:**
     * **测试时增强 (TTA):** 水平/垂直翻转。
     * **（新增）伪标签 (Pseudo-Labeling):**
       * 用集成模型对测试集进行预测，挑选高置信度样本加入训练集进行微调。
  3. **生成提交文件:** 整合所有技术，生成最终的 `submission.csv`。
  4. **技术报告撰写:**
     * **叙事主线:** "我们构建了一个从数据特性出发，具备分层融合、动态感知和双重不确定性量化能力的下一代可信赖滑坡检测系统。"
     * **关键图表:** HCF-Transformer架构图、分层注意力图、动态权重图、双重不确定性对比图、Grad-CAM图。
