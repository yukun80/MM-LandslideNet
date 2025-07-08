# **实施计划：赢得滑坡检测挑战赛 (冠军优化版)**

**最终目标:** 竞赛第一名。
**核心理念:** 结合SOTA分层融合架构、动态模态加权、双重不确定性量化以及极限性能冲刺策略，构建一个在F1分数和创新性报告上都具备压倒性优势的解决方案。

---

### **Phase 1: 信息内容评估与数据清洗 (预计用时: 2天)**

* **目标:** 识别并剔除信息量过低（如全噪声）的训练数据，然后基于清洗后的高质量数据集，计算统计特性并制定预处理流程。
* **任务:**
  1. **基于标准差的数据质量评估 (`scripts/data_quality_assessment.py`):**
     * **创建新脚本:** 编写一个名为 `data_quality_assessment.py` 的新脚本。
     * **定义质量标准:** 我们的核心假设是，缺乏有效信息的图像，其像素值标准差会显著偏低。此脚本将遍历所有训练样本，计算**每个样本在所有12个通道上的平均标准差**。
     * **生成质量分数:** 将每个样本的 `image_id` 及其计算出的“平均标准差”保存到 `outputs/image_quality_scores.csv`。这为我们提供了一个量化的信息内容指标。
  2. **阈值确定与坏数据排除 (`scripts/analyze_quality.py`):**
     * **创建分析脚本:** 编写一个名为 `analyze_quality.py` 的新脚本（或使用Jupyter Notebook）。
     * **分析分布:** 加载 `image_quality_scores.csv`，绘制所有样本“平均标准差”的直方图。通过观察这个分布，我们可以识别出代表“坏数据”的低分 outlier 群体，并据此确定一个合理的排除阈值。
     * **生成排除列表:** 根据确定的阈值，筛选出所有低质量样本的 `image_id`，并将其保存到 `outputs/exclude_ids.json`。
  3. **最终统计与预处理策略 (`scripts/data_analysis.py`):**
     * **修改现有脚本:** 更新 `data_analysis.py`，使其首先加载 `outputs/exclude_ids.json`。
     * **计算清洗后数据的统计:** 脚本将遍历**排除了坏数据之后**的高质量训练集，计算并输出每个通道的全局均值和标准差，更新到 `outputs/channel_stats.json`。
     * **数据加载器规划:** 最终的 `Dataset` 类将使用 `exclude_ids.json` 作为黑名单，并使用 `channel_stats.json` 对高质量数据进行标准化处理。

---

### **Phase 2: 高效光学基线模型 (Swin Transformer Optical Baseline) (预计用时: 2天)**

*   **目标:** 基于Phase 1的发现，快速构建并训练一个仅使用高质量光学数据的基线模型。此举旨在验证数据清洗的有效性，并为后续更复杂的融合模型建立一个必须超越的、架构一致的坚实性能基准。
*   **任务:**
    1.  **数据加载器 (`src/data/dataset.py`):**
        *   **核心修改:** `Dataset`类将首先加载`outputs/exclude_ids.json`，并在初始化时完全排除这些低质量数据ID。
        *   **单模态输出:** `__getitem__`方法现在只专注于光学数据。它将加载`.npy`文件，计算NDVI，然后仅提取并返回一个**5通道的张量**（R, G, B, NIR, NDVI）。
        *   **不平衡处理:** 保留**加权随机采样 (`WeightedRandomSampler`)**，以处理清洗后数据集中仍然存在的类别不平衡问题。

    2.  **模型架构 (`src/models/baseline_optical_model.py`):**
        *   **创建新脚本:** `baseline_optical_model.py`。
        *   **单分支架构:** 构建一个简单的、单一分支的分类模型。
        *   **骨干网络:** 使用一个在ImageNet上预训练的`Swin Transformer`，如`timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)`。这确保了与Phase 3冠军模型的架构延续性。
        *   **输入层适配:** 修改Swin Transformer的第一个卷积层（`patch_embed.proj`）以接受5个输入通道。初始化新层的权重时，可以平均现有3通道的预训练权重，以保留学习成果。
        *   **分类头:** 在骨干网络之后接一个`nn.LayerNorm`和一个简单的`nn.Linear`层用于最终分类。

    3.  **训练引擎 (`src/engine/train.py`):**
        *   **简化流程:** 训练循环现在只处理来自数据加载器的单个张量输入，简化了代码逻辑。
        *   **保留核心组件:** 继续使用强大的损失函数（如Focal Loss）、优化器（AdamW）和学习率调度器（Cosine Annealing）来确保训练的稳定性和有效性。

---

### **Phase 3: Champion Model (Adaptive Tri-Modal Fusion Network) (预计用时: 5-6天)**

*   **核心理念:** 构建一个单一、端到端的自适应三模态融合冠军模型。该模型将采用三个并行的Swin Transformer骨干网络，分别处理光学、SAR强度和SAR差分数据。通过一个先进的交叉注意力融合核心，模型能够让光学特征主动地从SAR模态中查询和补充信息，从而根据每个样本的特性（如云层覆盖）动态地调整其融合策略，并量化其预测的不确定性。
*   **任务:**
    1.  **数据加载器 (`src/data/dataset.py`):**
        *   **关键实现:** 重构`Dataset`类，使其为每个样本提供三个独立的、经过归一化的数据张量，分别对应三个模态：
            *   `optical_data`: 5通道 (R, G, B, NIR, NDVI)
            *   `sar_intensity_data`: 4通道 (升/降轨的VV, VH)
            *   `sar_diff_data`: 4通道 (升/降轨的Diff VV, Diff VH)

    2.  **模型架构 (`src/models/champion_model.py`):**
        *   **三分支Swin Transformer编码器:**
            *   为每个模态实例化一个独立的`Swin Transformer`骨干网络（如 `swin_tiny_patch4_window7_224`）。
            *   修改每个骨干网络的输入层以接受相应数量的通道（5, 4, 4）。光学分支使用ImageNet预训练权重；SAR分支从随机权重开始训练。
            *   每个编码器输出一个高层特征向量（CLS token）：`feat_optical`, `feat_sar`, `feat_sar_diff`。
        *   **交叉注意力融合核心 (Cross-Attention Fusion Core):**
            *   **主干查询:** `feat_optical`作为主要的“查询”（Query）向量。
            *   **信息增强:** `feat_optical`将依次对`feat_sar`和`feat_sar_diff`执行交叉注意力操作。这使得光学特征能够根据需要，主动从SAR模态中“提取”和“融合”地形及变化信息，生成一个最终的、高度情境感知的融合特征向量 `fused_feature`。
        *   **不确定性预测头:**
            *   将最终的`fused_feature`送入一个MLP头。该MLP头将输出两个值：
                1.  分类的Logit（用于预测滑坡概率）。
                2.  一个方差 `σ^2`（用于量化模型的**数据不确定性**）。

    3.  **训练与推理:**
        *   **训练引擎:** 修改训练循环以适应三输入模型。损失函数需要结合分类损失（如Focal Loss）和不确定性损失（通过高斯负对数似然来利用预测的方差）。
        *   **推理引擎:** 在推理时，保留模型的Dropout层（设置为`model.train()`)并进行多次（如20次）前向传播（**MC Dropout**），以估算模型的**模型不确定性**。

    4.  **可解释性 (XAI) (`src/utils/xai.py`):**
        *   **交叉注意力可视化:** 关键的XAI洞见将来自于可视化交叉注意力模块的权重。我们可以清晰地展示，当光学图像清晰时，对SAR模态的注意力权重很低；而当光学图像被云遮挡时，模型会自动增加对SAR模态的注意力权重，以获取决策依据。
        *   **不确定性可视化:** 生成不确定性图，高亮显示模型对其预测“不自信”的区域，这直接对应了“可信赖AI”的要求。

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
