# **实施计划：MM-InternImage-TNF 冠军之路 (V2 - 模块化)**

**最终目标:** 竞赛第一名。
**核心理念:** 基于第一性原理，将SOTA骨干网络 (**InternImage-T**)、前沿多模态融合策略 (**TNF-style Fusion**) 和以数据为中心的预处理相结合，构建一个代码结构清晰、模块化、可维护、性能卓越的解决方案。

---

### **Phase 1 & 2: 基础工程 (已完成)**

*   **成果:** 我们已完成数据清洗、分析，并为所有模态独立计算了归一化统计参数。一个基础的训练和评估流程也已验证完毕。我们将在这些坚实的基础上进行构建。

---

### **Phase 3: SOTA模型模块化实施 (MM-InternImage-TNF)**

**设计总览与思路:** 本阶段的目标是将我们在研究阶段确立的先进算法思想，转化为一个结构清晰、代码优雅、易于调试和扩展的工程实现。我们选择**InternImage-T**作为骨干网络，因其动态感受野能更好地适应滑坡的不规则形态，同时其预训练权重能加速光学特征的学习。我们采用**三分支架构**，分别处理光学、SAR及SAR差值数据，以尊重不同模态的物理特性。最终，我们借鉴**TNF论文**中的核心思想，设计一个先进的融合模块来智能地整合三分支的特征。整个实现过程遵循**关注点分离 (Separation of Concerns)** 的软件工程原则，将代码拆分为配置、数据、模型、工具、训练和预测六个独立的模块，极大地提高了代码的可维护性和可扩展性。

**目标:** 将先进的算法思想，转化为一个结构清晰、代码优雅、易于调试和扩展的工程实现。我们将创建六个核心模块，各司其职。

#### **Step 3.1: `config.py` - 全局配置中心**

*   **文件路径:** `mm_intern_image_src/config.py`
*   **目的:** 集中管理所有超参数和路径，避免硬编码，方便快速实验。
*   **关键内容:**
    *   **路径:** `DATA_DIR`, `TRAIN_CSV_PATH`, `STATS_FILE_PATH`, `EXCLUDE_FILE_PATH`, `CHECKPOINT_DIR`.
    *   **硬件:** `DEVICE` (e.g., "cuda" if available else "cpu").
    *   **模型参数:** `MODEL_NAME` (e.g., 'MM-InternImage-TNF-T'), `NUM_CLASSES` (1 for binary).
    *   **训练超参数:** `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `VALIDATION_SPLIT`.
    *   **数据加载:** `NUM_WORKERS`.

#### **Step 3.2: `dataset.py` - 数据引擎**

*   **文件路径:** `mm_intern_image_src/dataset.py`
*   **目的:** 封装所有与数据加载、预处理和增强相关的复杂逻辑。
*   **核心实现:**
    *   **`MultiModalLandslideDataset(torch.utils.data.Dataset)` 类:**
        *   `__init__`: 接收 `df`, `data_dir`, `stats_dict`, `augmentations`。在此加载 `stats_dict`。
        *   `__getitem__`: 实现以下流程：
            1.  加载12通道 `.npy` 文件。
            2.  **模态分离:** 拆分为 `optical`, `sar`, `sar_diff` 三个NumPy数组。
            3.  **特征工程:** 为`optical`数据计算并堆叠NDVI通道，形成5通道数组。
            4.  **独立归一化:** 使用`stats_dict`中的均值和标准差，对三个模态的数组分别进行独立的Z-score归一化。
            5.  **数据增强:** 应用 `albumentations` 库（如果提供）。
            6.  **返回字典:** 返回 `{'optical': ..., 'sar': ..., 'sar_diff': ..., 'label': ...}`。

#### **Step 3.3: `utils.py` - 通用工具箱**

*   **文件路径:** `mm_intern_image_src/utils.py`
*   **目的:** 存放所有可重用的辅助函数和类，保持其他模块的整洁。
*   **核心功能:**
    *   **损失函数:**
        *   `FocalLoss(nn.Module)`: 实现Focal Loss，专注于难分样本。
        *   `DiceLoss(nn.Module)`: 实现Dice Loss，优化分割类指标，对类别不平衡有良好效果。
        *   `CombinedLoss(nn.Module)`: 将上述两种损失组合起来，`L = L_focal + L_dice`。
    *   **评估指标:**
        *   `calculate_metrics(preds, labels)`: 输入模型输出和真实标签，返回一个包含`f1_score`, `precision`, `recall`, `accuracy`的字典。
    *   **检查点管理:**
        *   `save_checkpoint(state, filepath)`: 保存模型、优化器等状态到文件。
        *   `load_checkpoint(filepath, model, optimizer)`: 加载状态。
    *   **环境设置:**
        *   `seed_everything(seed)`: 设置全局随机种子以保证实验可复现。

#### **Step 3.4: `models.py` - 模型架构定义**

*   **文件路径:** `mm_intern_image_src/models.py`
*   **目的:** 仅包含模型架构的定义，与训练、数据等逻辑完全解耦。
*   **核心实现:**
    *   **`TNFFusionBlock(nn.Module)` 类:**
        *   实现受TNF论文启发的融合块。接收三个分支的特征图作为输入，输出融合后的单一特征向量。
        *   内部逻辑包含：全局池化、自注意力、交叉注意力、门控机制。
    *   **`MMInternImageTNF(nn.Module)` 类:**
        *   `__init__`: 实例化三个`InternImage-T`骨干网络（光学分支加载预训练权重，SAR分支随机初始化），并实例化`TNFFusionBlock`和最终的分类头。
        *   `forward`: 定义数据流，将三个输入分别送入三个分支，将输出特征送入融合块，最后通过分类头得到预测结果。

#### **Step 3.5: `train.py` - 训练流程编排**

*   **文件路径:** `mm_intern_image_src/train.py`
*   **目的:** 编排整个训练和验证流程，连接数据、模型、损失和优化器。
*   **核心功能:**
    *   **`train_one_epoch(model, dataloader, optimizer, criterion, device)`:**
        *   实现单次epoch的训练循环，使用`tqdm`显示进度条。
    *   **`evaluate(model, dataloader, criterion, device)`:**
        *   实现验证逻辑，调用`utils.calculate_metrics`计算性能。
    *   **`run_training()` (主函数):**
        1.  **设置:** 调用`utils.seed_everything`，加载`config`。
        2.  **数据准备:** 加载CSV，执行过滤和划分，创建`MultiModalLandslideDataset`实例，并特别注意为`train_loader`创建`WeightedRandomSampler`来处理类别不平衡。
        3.  **模型初始化:** 实例化`MMInternImageTNF`，`CombinedLoss`，`AdamW`优化器，以及学习率调度器 (e.g., `CosineAnnealingLR`)。
        4.  **主循环:** 循环遍历epochs，调用`train_one_epoch`和`evaluate`。
        5.  **保存模型:** 在每个epoch后，根据验证集F1分数，调用`utils.save_checkpoint`保存最佳模型。

#### **Step 3.6: `predict.py` - 推理引擎**

*   **文件路径:** `mm_intern_image_src/predict.py`
*   **目的:** 提供一个独立的、易于使用的脚本，用于对新的、单个的`.npy`文件进行预测。
*   **核心功能:**
    *   **`preprocess_single_image(image_path, stats_dict)` 函数:**
        *   复制`Dataset`中的单样本处理逻辑（加载、分离、NDVI、归一化）。
    *   **`predict(model, image_tensor, device)` 函数:**
        *   接收模型和预处理后的张量，设置为`eval`模式，在`torch.no_grad()`下执行推理，返回预测概率。
    *   **主逻辑:**
        1.  使用`argparse`接收`--model_path`（模型检查点）和`--image_path`（待预测图片）作为命令行参数。
        2.  加载`channel_stats.json`。
        3.  调用`utils.load_checkpoint`加载训练好的模型权重。
        4.  调用`preprocess_single_image`处理输入图片。
        5.  调用`predict`获得结果并打印。

---

### **Phase 4 & 5: 迭代优化与冲刺 (待启动)**

*   在`MM-InternImage-TNF` V1版本稳定后，启动主动学习循环，并为最终的模型集成和报告撰写做准备。此模块化结构将极大地简化后续的实验和迭代过程。
