#!/bin/bash

# cuDNN快速修复脚本
# 解决PyTorch + InternImage的cuDNN兼容性问题

set -e

echo "🔧 cuDNN环境快速修复脚本"
echo "=================================="

# 检查conda环境
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ 错误：请先激活conda环境 (conda activate dl310)"
    exit 1
fi

echo "✅ 当前conda环境: $CONDA_PREFIX"

# 定义cuDNN库路径
CUDNN_LIB_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"
echo "📁 cuDNN库路径: $CUDNN_LIB_PATH"

# 检查cuDNN库文件
if [ ! -d "$CUDNN_LIB_PATH" ]; then
    echo "❌ 错误：cuDNN库路径不存在，请重新安装PyTorch"
    exit 1
fi

echo "🔍 检查cuDNN库文件..."
REQUIRED_LIBS=(
    "libcudnn.so.8"
    "libcudnn_cnn_infer.so.8"
    "libcudnn_cnn_train.so.8"
    "libcudnn_ops_infer.so.8"
    "libcudnn_ops_train.so.8"
    "libcudnn_adv_infer.so.8"
    "libcudnn_adv_train.so.8"
)

for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "$CUDNN_LIB_PATH/$lib" ]; then
        echo "✅ $lib - 存在"
    else
        echo "❌ $lib - 缺失"
        MISSING_LIBS=true
    fi
done

if [ "$MISSING_LIBS" = true ]; then
    echo "❌ 缺失关键库文件，请重新安装PyTorch:"
    echo "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
    exit 1
fi

echo "🔗 创建符号链接..."
cd "$CUDNN_LIB_PATH"

# 创建符号链接
LIB_MAPPINGS=(
    "libcudnn.so.8:libcudnn.so"
    "libcudnn_cnn_infer.so.8:libcudnn_cnn_infer.so"
    "libcudnn_cnn_train.so.8:libcudnn_cnn_train.so"
    "libcudnn_ops_infer.so.8:libcudnn_ops_infer.so"
    "libcudnn_ops_train.so.8:libcudnn_ops_train.so"
    "libcudnn_adv_infer.so.8:libcudnn_adv_infer.so"
    "libcudnn_adv_train.so.8:libcudnn_adv_train.so"
)

for mapping in "${LIB_MAPPINGS[@]}"; do
    IFS=':' read -r target link_name <<< "$mapping"
    if [ -f "$target" ]; then
        if [ -L "$link_name" ] || [ -f "$link_name" ]; then
            rm -f "$link_name"
        fi
        ln -s "$target" "$link_name"
        echo "✅ 创建链接: $link_name -> $target"
    fi
done

echo "⚙️ 设置环境变量..."

# 创建conda环境激活脚本
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

# 创建激活脚本
cat > "$ACTIVATE_DIR/cudnn_env.sh" << EOF
#!/bin/bash
# cuDNN环境配置
export CUDNN_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CUDNN_LIB_PATH:\$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_PREFIX"

# 强制使用conda环境中的cuDNN
export CUDNN_PATH="$CUDNN_LIB_PATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 优先使用conda环境中的cuDNN库
export LD_LIBRARY_PATH="$CUDNN_LIB_PATH:\$LD_LIBRARY_PATH"
EOF

# 创建反激活脚本
cat > "$DEACTIVATE_DIR/cudnn_env.sh" << EOF
#!/bin/bash
# 清理cuDNN环境变量
unset CUDNN_HOME
unset CUDNN_PATH
unset PYTORCH_CUDA_ALLOC_CONF
EOF

chmod +x "$ACTIVATE_DIR/cudnn_env.sh"
chmod +x "$DEACTIVATE_DIR/cudnn_env.sh"

echo "✅ 环境变量脚本已创建"

echo "🧪 测试cuDNN基础功能..."
python3 -c "
import torch
import torch.backends.cudnn as cudnn
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'cuDNN版本: {cudnn.version()}')
print(f'cuDNN启用: {cudnn.enabled}')
" 2>/dev/null || echo "❗ 基础测试可能需要重新激活环境"

echo ""
echo "🎉 cuDNN环境修复完成！"
echo "==============================="
echo "请执行以下步骤完成配置："
echo ""
echo "1. 重新激活conda环境:"
echo "   conda deactivate"
echo "   conda activate dl310"
echo ""
echo "2. 验证环境变量:"
echo "   echo \$CUDNN_HOME"
echo "   echo \$LD_LIBRARY_PATH"
echo ""
echo "3. 测试cuDNN功能:"
echo "   python fix_cudnn_environment.py --test"
echo ""
echo "4. 如果问题仍然存在，尝试以下优化："
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256"
echo ""
echo "现在您可以尝试运行训练脚本！" 