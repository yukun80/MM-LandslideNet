"""
cuDNN环境配置修复脚本
解决PyTorch + InternImage的cuDNN兼容性问题
"""

import os
import sys
import subprocess
import pathlib
import shutil
from pathlib import Path


def get_conda_env_path():
    """获取当前conda环境路径"""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is None:
        print("❌ 错误：请确保已激活conda环境")
        sys.exit(1)
    return conda_prefix


def check_cudnn_libraries():
    """检查cuDNN库文件"""
    conda_prefix = get_conda_env_path()
    cudnn_lib_path = Path(conda_prefix) / "lib" / "python3.10" / "site-packages" / "nvidia" / "cudnn" / "lib"

    required_libs = [
        "libcudnn.so.8",
        "libcudnn_cnn_infer.so.8",
        "libcudnn_cnn_train.so.8",
        "libcudnn_ops_infer.so.8",
        "libcudnn_ops_train.so.8",
        "libcudnn_adv_infer.so.8",
        "libcudnn_adv_train.so.8",
    ]

    print(f"📁 检查cuDNN库文件路径: {cudnn_lib_path}")
    missing_libs = []

    for lib in required_libs:
        lib_path = cudnn_lib_path / lib
        if lib_path.exists():
            print(f"✅ {lib} - 存在")
        else:
            print(f"❌ {lib} - 缺失")
            missing_libs.append(lib)

    return cudnn_lib_path, missing_libs


def create_library_links(cudnn_lib_path):
    """创建cuDNN库文件的符号链接"""
    print("\n🔗 创建cuDNN库文件符号链接...")

    # 创建主要的符号链接
    lib_mappings = {
        "libcudnn.so.8": "libcudnn.so",
        "libcudnn_cnn_infer.so.8": "libcudnn_cnn_infer.so",
        "libcudnn_cnn_train.so.8": "libcudnn_cnn_train.so",
        "libcudnn_ops_infer.so.8": "libcudnn_ops_infer.so",
        "libcudnn_ops_train.so.8": "libcudnn_ops_train.so",
        "libcudnn_adv_infer.so.8": "libcudnn_adv_infer.so",
        "libcudnn_adv_train.so.8": "libcudnn_adv_train.so",
    }

    for target, link_name in lib_mappings.items():
        target_path = cudnn_lib_path / target
        link_path = cudnn_lib_path / link_name

        if target_path.exists():
            if link_path.exists():
                link_path.unlink()
            try:
                link_path.symlink_to(target_path.name)
                print(f"✅ 创建链接: {link_name} -> {target}")
            except Exception as e:
                print(f"❌ 创建链接失败 {link_name}: {e}")


def setup_environment_variables(cudnn_lib_path):
    """设置环境变量"""
    print("\n⚙️ 设置环境变量...")

    conda_prefix = get_conda_env_path()

    # 创建环境变量脚本
    env_script_path = Path(conda_prefix) / "etc" / "conda" / "activate.d" / "cudnn_env.sh"
    env_script_path.parent.mkdir(parents=True, exist_ok=True)

    env_script_content = f"""#!/bin/bash
# cuDNN环境配置
export CUDNN_HOME={conda_prefix}
export LD_LIBRARY_PATH={cudnn_lib_path}:$LD_LIBRARY_PATH
export CUDA_HOME={conda_prefix}

# 强制使用conda环境中的cuDNN
export CUDNN_PATH={cudnn_lib_path}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
"""

    with open(env_script_path, "w") as f:
        f.write(env_script_content)

    # 创建反激活脚本
    deactivate_script_path = Path(conda_prefix) / "etc" / "conda" / "deactivate.d" / "cudnn_env.sh"
    deactivate_script_path.parent.mkdir(parents=True, exist_ok=True)

    deactivate_script_content = """#!/bin/bash
# 清理cuDNN环境变量
unset CUDNN_HOME
unset CUDNN_PATH
unset PYTORCH_CUDA_ALLOC_CONF
"""

    with open(deactivate_script_path, "w") as f:
        f.write(deactivate_script_content)

    print(f"✅ 环境变量脚本已创建: {env_script_path}")
    print(f"✅ 反激活脚本已创建: {deactivate_script_path}")


def test_cudnn_functionality():
    """测试cuDNN功能"""
    print("\n🧪 测试cuDNN功能...")

    test_script = """
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {cudnn.version()}")
print(f"cuDNN启用: {cudnn.enabled}")
print(f"cuDNN确定性: {cudnn.deterministic}")

# 测试简单的卷积操作
if torch.cuda.is_available():
    try:
        device = torch.device('cuda')
        x = torch.randn(1, 3, 224, 224, device=device)
        conv = nn.Conv2d(3, 64, 3, padding=1).to(device)
        
        # 测试前向传播
        with torch.no_grad():
            y = conv(x)
            print(f"✅ cuDNN卷积测试成功: {y.shape}")
        
        # 测试反向传播
        y.sum().backward()
        print("✅ cuDNN反向传播测试成功")
        
         except Exception as e:
         print(f"❌ cuDNN测试失败: {e}")
         return False
 else:
     print("❌ CUDA不可用")
     return False
"""

    try:
        result = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ cuDNN功能测试通过")
            print(result.stdout)
            return True
        else:
            print("❌ cuDNN功能测试失败")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ cuDNN测试超时")
        return False
    except Exception as e:
        print(f"❌ cuDNN测试错误: {e}")
        return False


def main():
    """主函数"""
    print("🔧 cuDNN环境配置修复脚本")
    print("=" * 50)

    # 检查库文件
    cudnn_lib_path, missing_libs = check_cudnn_libraries()

    if missing_libs:
        print(f"\n❌ 缺失关键库文件，请重新安装PyTorch:")
        print("conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False

    # 创建符号链接
    create_library_links(cudnn_lib_path)

    # 设置环境变量
    setup_environment_variables(cudnn_lib_path)

    print("\n📋 修复完成，请执行以下步骤：")
    print("1. 重新激活conda环境:")
    print("   conda deactivate && conda activate dl310")
    print("2. 验证环境变量:")
    print("   echo $CUDNN_HOME")
    print("   echo $LD_LIBRARY_PATH")
    print("3. 测试cuDNN功能:")
    print("   python fix_cudnn_environment.py --test")

    return True


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = test_cudnn_functionality()
        sys.exit(0 if success else 1)
    else:
        success = main()
        sys.exit(0 if success else 1)
