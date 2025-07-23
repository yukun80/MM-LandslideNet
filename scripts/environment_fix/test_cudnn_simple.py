#!/usr/bin/env python3
"""
简单的cuDNN功能测试脚本
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys


def test_cudnn():
    """测试cuDNN功能"""
    print("🧪 测试cuDNN功能...")
    print("=" * 40)

    # 基本信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {cudnn.version()}")
    print(f"cuDNN启用: {cudnn.enabled}")
    print(f"cuDNN确定性: {cudnn.deterministic}")

    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False

    # 获取GPU信息
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name()}")

    try:
        device = torch.device("cuda")

        # 测试1: 简单卷积
        print("\n🔍 测试1: 简单卷积操作")
        x = torch.randn(1, 3, 224, 224, device=device)
        conv = nn.Conv2d(3, 64, 3, padding=1).to(device)

        with torch.no_grad():
            y = conv(x)
            print(f"✅ 卷积成功: {x.shape} -> {y.shape}")

        # 测试2: 反向传播
        print("\n🔍 测试2: 反向传播")
        x.requires_grad_(True)
        y = conv(x)
        loss = y.sum()
        loss.backward()
        print(f"✅ 反向传播成功: 梯度形状 {x.grad.shape}")

        # 测试3: 批量归一化
        print("\n🔍 测试3: 批量归一化")
        bn = nn.BatchNorm2d(64).to(device)
        y_bn = bn(y)
        print(f"✅ 批量归一化成功: {y.shape} -> {y_bn.shape}")

        # 测试4: 激活函数
        print("\n🔍 测试4: 激活函数")
        relu = nn.ReLU().to(device)
        y_relu = relu(y_bn)
        print(f"✅ 激活函数成功: {y_bn.shape} -> {y_relu.shape}")

        # 测试5: 池化操作
        print("\n🔍 测试5: 池化操作")
        pool = nn.MaxPool2d(2, 2).to(device)
        y_pool = pool(y_relu)
        print(f"✅ 池化成功: {y_relu.shape} -> {y_pool.shape}")

        # 测试6: 内存使用
        print("\n🔍 测试6: GPU内存使用")
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        memory_cached = torch.cuda.memory_reserved(device) / 1024**2
        print(f"✅ GPU内存 - 已分配: {memory_allocated:.1f}MB, 已缓存: {memory_cached:.1f}MB")

        return True

    except Exception as e:
        print(f"❌ cuDNN测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cudnn()

    if success:
        print("\n🎉 cuDNN功能测试全部通过！")
        print("环境配置成功，可以运行训练脚本")
    else:
        print("\n❌ cuDNN功能测试失败")
        print("请检查CUDA和cuDNN配置")
        sys.exit(1)
