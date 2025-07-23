# MM-LandslideNet 光学基线模型推理说明

## 🚀 快速开始

### 运行推理程序
```bash
# 方式1：使用启动脚本（推荐）
python run_optical_inference.py

# 方式2：直接运行推理模块
python optical_src/inference.py
```

## 📁 输出文件

推理程序会在 `outputs/submissions/` 目录下生成两个文件：

### 1. 提交文件 `optical_baseline_submission_{timestamp}.csv`
```csv
ID,label
ID_ICB8K9,0
ID_2D4AOJ,0
ID_LSA46F,1
...
```
- **格式**：标准比赛提交格式
- **内容**：每个测试样本的最终预测标签（0或1）
- **用途**：直接提交给比赛平台

### 2. 概率文件 `optical_baseline_probabilities_{timestamp}.csv`
```csv
ID,probability
ID_ICB8K9,1.946e-05
ID_2D4AOJ,2.552e-05
ID_LSA46F,0.999973
...
```
- **格式**：包含原始预测概率
- **内容**：每个测试样本的预测概率值 [0,1]
- **用途**：分析模型置信度，调整阈值

## ⚙️ 配置参数

### 模型配置
- **模型**：Swin Transformer Tiny (swin_tiny_patch4_window7_224)
- **输入**：5通道数据 (R, G, B, NIR, NDVI)
- **图像尺寸**：64×64像素
- **权重文件**：`outputs/checkpoints/optical_baseline/best_model.pth`

### 推理设置
- **分类阈值**：0.5（概率>0.5预测为正样本）
- **批次大小**：根据训练配置自动设置
- **测试时增强（TTA）**：默认启用（提高性能）
- **GPU加速**：自动检测并使用可用GPU

## 📊 输出统计示例

```
Submission statistics:
  Total samples: 5,399
  Positive predictions: 245 (4.54%)
  Negative predictions: 5,154 (95.46%)
```

## 🔧 高级用法

### 自定义阈值
```python
from optical_src.inference import OpticalInference
from optical_src.config import OpticalBaselineConfig

# 创建推理实例
inference = OpticalInference()

# 生成概率预测
predictions = inference.predict(use_tta=True)

# 使用自定义阈值创建提交文件
submission_df = inference.create_submission(predictions, threshold=0.3)
output_path = inference.save_submission(submission_df, "custom_threshold_0.3.csv")
```

### 不使用测试时增强
```python
# 更快的推理（略低精度）
submission_path = inference.run_inference(
    use_tta=False,  # 禁用测试时增强
    threshold=0.5,
    save_probabilities=True
)
```

## 🔍 模型性能分析

### 预测分布特点
- **高置信正样本**：概率 > 0.9 的样本通常是明显的滑坡区域
- **高置信负样本**：概率 < 0.1 的样本通常是非滑坡区域
- **边界样本**：概率在 0.3-0.7 之间的样本可能需要人工复核

### 阈值调优建议
```
阈值 0.3: 更多正样本，提高召回率
阈值 0.5: 平衡精度和召回率（默认）
阈值 0.7: 更高精度，降低误报率
```

## 📝 注意事项

1. **GPU内存**：推理过程需要约3-4GB GPU内存
2. **处理时间**：5,399个测试样本约需3-5分钟
3. **文件大小**：
   - 提交文件：约70KB
   - 概率文件：约175KB

## ❓ 常见问题

### Q: 如何处理"weights_only"错误？
A: 推理程序已自动处理此问题，使用 `weights_only=False` 参数加载模型。

### Q: 推理失败怎么办？
A: 检查以下内容：
- 确保模型文件存在：`outputs/checkpoints/optical_baseline/best_model.pth`
- 确保测试数据存在：`dataset/test_data/` 和 `dataset/Test.csv`
- 查看日志文件：`logs/optical_baseline/inference.log`

### Q: 如何验证结果正确性？
A: 
- 检查提交文件行数是否为5,400（包含标题行）
- 确认所有ID都在测试集中
- 验证标签只包含0和1

## 📈 性能优化建议

1. **批次大小**：根据GPU内存适当调整
2. **多GPU推理**：可在代码中添加DataParallel支持
3. **混合精度**：可启用AMP加速推理
4. **模型量化**：考虑使用INT8量化降低内存占用

---

**生成时间**：${timestamp}  
**模型版本**：光学基线模型v1.0  
**推理版本**：inference.py v1.0 