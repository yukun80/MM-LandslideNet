"""
Fixed MM-LandslideNet-TNF models.py

主要修复：
1. 修复compute_loss方法的损失函数兼容性
2. 处理不同损失函数返回类型（tensor vs dict）
3. 确保targets维度正确处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

# Import InternImage (from existing implementation)
from mm_intern_image_src.intern_image_import import InternImage

"""
python -m mm_intern_image_src.models
"""

logger = logging.getLogger(__name__)


class OpticalBranch(nn.Module):
    """
    修复后的Optical Primary Branch for 5-channel optical data processing.

    核心修复：
    1. 在初始化时检测InternImage的实际输出维度
    2. 预先创建feature_adapter来处理维度不匹配
    3. 避免在forward时动态创建层
    4. 确保训练和推理时的架构完全一致
    """

    def __init__(
        self,
        in_channels: int = 5,
        feature_dim: int = 512,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
    ):
        super(OpticalBranch, self).__init__()

        self.in_channels = in_channels
        self.target_feature_dim = feature_dim  # 目标特征维度：512

        # Step 1: Create InternImage-T backbone
        self.backbone = self._create_intern_image_backbone(pretrained)

        # Step 2: 检测backbone的实际输出维度
        self.actual_backbone_dim = self._detect_backbone_output_dim()

        # Step 3: 如果维度不匹配，预先创建adapter
        if self.actual_backbone_dim != self.target_feature_dim:
            self.feature_adapter = nn.Linear(self.actual_backbone_dim, self.target_feature_dim)
            logger.info(f"✅ Feature adapter created: {self.actual_backbone_dim} -> {self.target_feature_dim}")
        else:
            self.feature_adapter = None
            logger.info(f"✅ No feature adapter needed: backbone outputs {self.actual_backbone_dim} features")

        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature projection and classification head
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(self.target_feature_dim),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Linear(self.target_feature_dim, 1)

        logger.info(f"OpticalBranch initialized with {self.target_feature_dim} features")

    def _detect_backbone_output_dim(self) -> int:
        """
        检测InternImage backbone的实际输出维度
        通过分析conv_head层的输出通道数来确定
        """
        # 方法1：通过conv_head层检测
        if hasattr(self.backbone, "conv_head") and hasattr(self.backbone.conv_head, "0"):
            conv_layer = self.backbone.conv_head[0]
            if hasattr(conv_layer, "out_channels"):
                actual_dim = conv_layer.out_channels
                logger.info(f"🔍 Detected backbone output dimension from conv_head: {actual_dim}")
                return actual_dim

        # 方法2：通过num_features属性检测
        if hasattr(self.backbone, "num_features"):
            actual_dim = self.backbone.num_features
            logger.info(f"🔍 Detected backbone output dimension from num_features: {actual_dim}")
            return actual_dim

        # 方法3：实际运行检测（在CPU上可能失败，但提供fallback）
        try:
            self.backbone.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, self.in_channels, 64, 64)
                dummy_features = self.backbone(dummy_input)

                # 处理不同的输出格式
                if dummy_features.dim() == 4:  # (B, C, H, W)
                    pooled_features = self.global_pool(dummy_features).flatten(1)
                elif dummy_features.dim() == 3:  # (B, H*W, C)
                    pooled_features = dummy_features.mean(dim=1)
                else:
                    pooled_features = dummy_features

                actual_dim = pooled_features.size(1)
                logger.info(f"🔍 Detected backbone output dimension via forward pass: {actual_dim}")
                return actual_dim
        except Exception as e:
            logger.warning(f"Forward pass detection failed: {e}")

        # 默认值：根据诊断结果，InternImage-T输出768维
        logger.warning(f"Could not detect backbone output dimension, using default 768")
        return 768

    def _create_intern_image_backbone(self, pretrained: bool) -> nn.Module:
        """Create InternImage-T backbone following existing implementation."""
        from mm_intern_image_src.intern_image_import import InternImage

        # InternImage-T配置：与现有代码完全一致
        backbone = InternImage(core_op="DCNv3", channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], num_classes=0)

        if pretrained:
            logger.info("✅ Using pretrained weights for optical backbone (conceptual).")

        # Modify input layer for 5 channels
        self._modify_input_layer(backbone, self.in_channels, pretrained)
        return backbone

    def _modify_input_layer(self, backbone: nn.Module, in_channels: int, pretrained: bool) -> None:
        """Modify InternImage input layer for multi-channel input."""
        original_conv = backbone.patch_embed.conv1

        if original_conv.in_channels == in_channels:
            return

        # Create new conv layer with correct input channels
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize weights appropriately
        with torch.no_grad():
            if pretrained and original_conv.in_channels == 3:
                # For pretrained weights, replicate RGB pattern
                new_conv.weight[:, :3, :, :] = original_conv.weight.data

                # Channel 3 (NIR): Use average of RGB
                new_conv.weight[:, 3, :, :] = original_conv.weight.data.mean(dim=1)

                # Channel 4 (NDVI): Initialize as combination of Red and NIR
                new_conv.weight[:, 4, :, :] = (
                    original_conv.weight.data[:, 0, :, :] * 0.5  # Red component
                    + new_conv.weight[:, 3, :, :] * 0.5  # NIR component
                ) * 0.5

                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0)

        backbone.patch_embed.conv1 = new_conv
        logger.info(f"✅ Optical backbone input layer modified for {in_channels} channels.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through optical branch.

        完全修复：不再有动态层创建，使用预先创建的adapter
        """
        # Extract features through InternImage backbone
        features = self.backbone(x)

        # Global pooling to get feature vector
        if features.dim() == 4:  # (B, C, H, W)
            features = self.global_pool(features).flatten(1)
        elif features.dim() == 3:  # (B, H*W, C)
            features = features.mean(dim=1)

        # 使用预先创建的feature adapter（如果存在）
        if self.feature_adapter is not None:
            features = self.feature_adapter(features)

        # Feature projection
        features = self.feature_proj(features)

        # Classification
        logits = self.classifier(features)

        return features, logits

    def get_feature_adapter_info(self) -> Dict[str, any]:
        """
        获取feature adapter的信息，用于调试和验证
        """
        if self.feature_adapter is not None:
            return {
                "has_adapter": True,
                "input_dim": self.actual_backbone_dim,
                "output_dim": self.target_feature_dim,
                "adapter_params": sum(p.numel() for p in self.feature_adapter.parameters()),
            }
        else:
            return {
                "has_adapter": False,
                "backbone_dim": self.actual_backbone_dim,
            }


class SARBranch(nn.Module):
    """
    SAR Collaborative Branch for 8-channel SAR data processing.

    Processes SAR original and difference channels using EfficientNet-B0
    for efficient feature extraction with focus on change detection.
    """

    def __init__(
        self, in_channels: int = 8, feature_dim: int = 512, pretrained: bool = True, dropout_rate: float = 0.1
    ):
        super(SARBranch, self).__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="avg",  # Use global average pooling
        )

        # Modify first layer for 8-channel SAR input
        self._modify_first_layer(pretrained)

        # Feature projection and classification head
        self.feature_proj = nn.Sequential(
            nn.BatchNorm1d(self.backbone.num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, self.feature_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(self.feature_dim, 1)

        logger.info(f"SARBranch initialized with {self.feature_dim} features")

    def _modify_first_layer(self, pretrained: bool) -> None:
        """Modify first convolutional layer for 8-channel SAR input."""
        original_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            self.in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize weights for SAR channels
        with torch.no_grad():
            if pretrained:
                # Replicate RGB pattern for SAR channels
                rgb_weights = original_conv.weight[:, :3]  # (out_ch, 3, k, k)

                # Initialize 8 SAR channels by replicating and modifying RGB weights
                for i in range(self.in_channels):
                    channel_idx = i % 3  # Cycle through RGB
                    scale_factor = 0.8 if i >= 4 else 1.0  # Difference channels get smaller weights
                    new_conv.weight[:, i] = rgb_weights[:, channel_idx] * scale_factor
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        self.backbone.conv_stem = new_conv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAR branch.

        Args:
            x: Input tensor of shape (B, 8, 64, 64)

        Returns:
            features: Global feature vector (B, feature_dim)
            logits: Classification logits (B, 1)
        """
        # Extract features through backbone
        features = self.backbone(x)  # EfficientNet global pooled features

        # Feature projection
        features = self.feature_proj(features)

        # Classification
        logits = self.classifier(features)

        return features, logits


class TNFFusionModule(nn.Module):
    """
    TNF-Inspired Gate-based Fusion Module.

    Implements efficient fusion mechanism inspired by Tri-branch Neural Fusion
    for combining optical and SAR features with adaptive weighting.
    """

    def __init__(
        self,
        optical_dim: int = 512,  # Updated to match InternImage-T output
        sar_dim: int = 512,
        fusion_dim: int = 512,  # Updated to match optical dim
        dropout_rate: float = 0.1,
    ):
        super(TNFFusionModule, self).__init__()

        self.optical_dim = optical_dim
        self.sar_dim = sar_dim
        self.fusion_dim = fusion_dim

        # Step 1: Dimension alignment
        self.sar_aligner = nn.Sequential(
            nn.Linear(sar_dim, fusion_dim), nn.ReLU(inplace=True), nn.Dropout(dropout_rate)
        )

        # Step 2: Modality quality assessment
        self.optical_quality = nn.Sequential(
            nn.Linear(optical_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.sar_quality = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Step 3: TNF Gate mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1),
        )

        # Step 4: Fusion feature processing
        self.fusion_proj = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout_rate))

        self.fusion_classifier = nn.Linear(fusion_dim, 1)

        logger.info(f"TNFFusionModule initialized with {fusion_dim} fusion features")

    def forward(
        self, optical_features: torch.Tensor, sar_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through TNF fusion module.

        Args:
            optical_features: Optical features (B, optical_dim)
            sar_features: SAR features (B, sar_dim)

        Returns:
            fusion_features: Fused features (B, fusion_dim)
            fusion_logits: Fusion classification logits (B, 1)
            fusion_weights: Adaptive fusion weights (B, 2)
        """
        batch_size = optical_features.size(0)

        # Step 1: Align SAR features to fusion dimension
        sar_aligned = self.sar_aligner(sar_features)  # (B, fusion_dim)

        # Step 2: Assess modality quality
        optical_conf = self.optical_quality(optical_features)  # (B, 1)
        sar_conf = self.sar_quality(sar_aligned)  # (B, 1)

        # Calculate quality-based weighting
        total_conf = optical_conf + sar_conf + 1e-8
        quality_weights = torch.cat([optical_conf / total_conf, sar_conf / total_conf], dim=1)  # (B, 2)

        # Step 3: TNF Gate-based fusion
        concat_features = torch.cat([optical_features, sar_aligned], dim=1)  # (B, 2*fusion_dim)
        gate_weights = self.gate_network(concat_features)  # (B, 2)

        # Combine quality assessment and gate weights
        adaptive_weights = 0.7 * quality_weights + 0.3 * gate_weights  # (B, 2)

        # Apply adaptive weighting
        fusion_features = adaptive_weights[:, 0:1] * optical_features + adaptive_weights[:, 1:2] * sar_aligned

        # Add residual connection (10% of original features)
        residual = 0.1 * (optical_features + sar_aligned)
        fusion_features = fusion_features + residual

        # Step 4: Process fusion features
        fusion_features = self.fusion_proj(fusion_features)
        fusion_logits = self.fusion_classifier(fusion_features)

        return fusion_features, fusion_logits, adaptive_weights


class MMTNFModel(nn.Module):
    """
    MM-LandslideNet-TNF: Main model implementing TNF-inspired dual-branch architecture.

    This model combines optical and SAR branches with TNF fusion for landslide detection,
    supporting both individual branch predictions and fused predictions.
    """

    def __init__(
        self,
        optical_channels: int = 5,
        sar_channels: int = 8,
        optical_feature_dim: int = 512,  # Updated to match InternImage-T
        sar_feature_dim: int = 512,
        fusion_dim: int = 512,  # Updated to match optical dim
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        branch_weights: Optional[Tuple[float, float, float]] = None,
    ):
        super(MMTNFModel, self).__init__()

        # Store configuration
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.optical_feature_dim = optical_feature_dim
        self.sar_feature_dim = sar_feature_dim
        self.fusion_dim = fusion_dim

        # Branch weights for loss computation (optical, sar, fusion)
        self.branch_weights = branch_weights or (0.3, 0.2, 0.5)

        # Initialize branches
        self.optical_branch = OpticalBranch(
            in_channels=optical_channels,
            feature_dim=optical_feature_dim,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )

        self.sar_branch = SARBranch(
            in_channels=sar_channels, feature_dim=sar_feature_dim, pretrained=pretrained, dropout_rate=dropout_rate
        )

        self.fusion_module = TNFFusionModule(
            optical_dim=optical_feature_dim, sar_dim=sar_feature_dim, fusion_dim=fusion_dim, dropout_rate=dropout_rate
        )

        # Model info
        self._log_model_info()

    def forward(
        self, optical_data: torch.Tensor, sar_data: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete TNF model.

        Args:
            optical_data: Optical input (B, 5, 64, 64)
            sar_data: SAR input (B, 8, 64, 64)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - optical_logits: Optical branch predictions (B, 1)
                - sar_logits: SAR branch predictions (B, 1)
                - fusion_logits: Fusion branch predictions (B, 1)
                - final_logits: Combined final predictions (B, 1)
                - fusion_weights: Adaptive fusion weights (B, 2) [optional]
                - features: All intermediate features [optional]
        """
        # Extract features from both branches
        optical_features, optical_logits = self.optical_branch(optical_data)
        sar_features, sar_logits = self.sar_branch(sar_data)

        # TNF fusion
        fusion_features, fusion_logits, fusion_weights = self.fusion_module(optical_features, sar_features)

        # Final ensemble prediction (simple averaging)
        final_logits = (optical_logits + sar_logits + fusion_logits) / 3.0

        # Prepare output
        outputs = {
            "optical_logits": optical_logits,
            "sar_logits": sar_logits,
            "fusion_logits": fusion_logits,
            "final_logits": final_logits,
            "fusion_weights": fusion_weights,
        }

        if return_features:
            outputs["features"] = {
                "optical_features": optical_features,
                "sar_features": sar_features,
                "fusion_features": fusion_features,
            }

        return outputs

    def compute_loss(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, loss_fn: nn.Module = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-branch loss with weighted combination.

        Args:
            outputs: Model outputs dictionary
            targets: Ground truth labels (B,) or (B, 1)
            loss_fn: Loss function to use

        Returns:
            Dictionary of losses
        """
        # Ensure targets have correct shape and are float
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)  # (B,) -> (B, 1)
        targets = targets.float()

        # Use default BCEWithLogitsLoss if none provided
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()

        # Helper function to compute loss and handle different return types
        def compute_single_loss(logits, targets, loss_fn):
            """Compute loss handling both tensor and dict returns."""
            loss_result = loss_fn(logits, targets)

            if isinstance(loss_result, dict):
                # CombinedLoss returns dict
                return loss_result["total_loss"]
            else:
                # Standard loss functions return tensor
                return loss_result

        # Individual branch losses
        optical_loss = compute_single_loss(outputs["optical_logits"], targets, loss_fn)
        sar_loss = compute_single_loss(outputs["sar_logits"], targets, loss_fn)
        fusion_loss = compute_single_loss(outputs["fusion_logits"], targets, loss_fn)

        # Weighted total loss
        total_loss = (
            self.branch_weights[0] * optical_loss
            + self.branch_weights[1] * sar_loss
            + self.branch_weights[2] * fusion_loss
        )

        return {
            "total_loss": total_loss,
            "optical_loss": optical_loss,
            "sar_loss": sar_loss,
            "fusion_loss": fusion_loss,
        }

    def predict(
        self, optical_data: torch.Tensor, sar_data: torch.Tensor, use_ensemble: bool = True, threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with optional ensemble and thresholding.

        Args:
            optical_data: Optical input (B, 5, 64, 64)
            sar_data: SAR input (B, 8, 64, 64)
            use_ensemble: Whether to use ensemble prediction
            threshold: Classification threshold

        Returns:
            Prediction dictionary with probabilities and classes
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(optical_data, sar_data)

            # Choose prediction source
            if use_ensemble:
                logits = outputs["final_logits"]
            else:
                logits = outputs["fusion_logits"]

            # Convert to probabilities and classes
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()

            return {
                "probabilities": probabilities,
                "predictions": predictions,
                "logits": logits,
                "all_outputs": outputs,
            }

    def _log_model_info(self) -> None:
        """Log model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"MM-TNF Model initialized:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Optical features: {self.optical_feature_dim}")
        logger.info(f"  - SAR features: {self.sar_feature_dim}")
        logger.info(f"  - Fusion features: {self.fusion_dim}")
        logger.info(f"  - Branch weights: {self.branch_weights}")


# Factory function for model creation
def create_tnf_model(config: Optional[Dict] = None, pretrained: bool = True, **kwargs) -> MMTNFModel:
    """
    Factory function to create MM-TNF model with configuration.

    Args:
        config: Optional configuration dictionary
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model arguments

    Returns:
        Initialized MM-TNF model
    """
    default_config = {
        "optical_channels": 5,
        "sar_channels": 8,
        "optical_feature_dim": 512,  # Updated to match InternImage-T
        "sar_feature_dim": 512,
        "fusion_dim": 512,  # Updated to match optical dim
        "dropout_rate": 0.1,
        "branch_weights": (0.3, 0.2, 0.5),
    }

    if config:
        default_config.update(config)
    default_config.update(kwargs)

    model = MMTNFModel(pretrained=pretrained, **default_config)

    logger.info("MM-TNF model created successfully")
    return model


# Alias for backward compatibility
create_optical_dominated_model = create_tnf_model


if __name__ == "__main__":
    import sys
    import traceback

    print("🔬 Testing MM-LandslideNet-TNF Model Architecture...")
    print("=" * 60)

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. InternImage requires GPU for DCNv3 operations.")
        print("This test will likely fail on CPU. Please run on a GPU-enabled system.")
        print("Proceeding anyway for structure verification...")

    try:
        # Test model creation
        print("\n🏗️ Creating MM-TNF model...")
        model = create_tnf_model(pretrained=False)  # Use pretrained=False for faster loading in tests
        model = model.to(device)
        model.eval()

        print("✅ Model created successfully!")

        # Print model structure summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

        print(f"\n📊 Model Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: {model_size_mb:.2f} MB")

        # Test forward pass
        print(f"\n🧪 Testing forward pass...")
        batch_size = 2
        optical_input = torch.randn(batch_size, 5, 64, 64).to(device)
        sar_input = torch.randn(batch_size, 8, 64, 64).to(device)

        with torch.no_grad():
            outputs = model(optical_input, sar_input)

        print(f"✅ Forward pass successful!")
        print(f"Output keys: {list(outputs.keys())}")

        # Test loss computation
        print(f"\n🧪 Testing loss computation...")
        targets = torch.randint(0, 2, (batch_size,)).float().to(device)

        # Test with standard loss
        loss_dict = model.compute_loss(outputs, targets)
        print(f"✅ Loss computation successful!")
        print(f"Loss keys: {list(loss_dict.keys())}")
        print(f"Total loss: {loss_dict['total_loss'].item():.4f}")

        print(f"\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)
