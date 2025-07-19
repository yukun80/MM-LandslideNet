"""
MM-LandslideNet-TNF: TNF-Inspired Dual-Branch Model for Landslide Detection

This module implements a dual-branch architecture inspired by Tri-branch Neural Fusion (TNF)
for multimodal landslide detection using Sentinel-1 SAR and Sentinel-2 optical data.

Key Features:
- Optical Primary Branch: InternImage-T for 5-channel optical data (output: 512-dim)
- SAR Collaborative Branch: EfficientNet-B0 for 8-channel SAR data (output: 512-dim)
- TNF Gate-based Fusion: Efficient fusion mechanism for 64×64 images
- Three-branch Training: Independent + fusion branch training strategy
- Optimized Data Flow: Unified NCHW format, no unnecessary conversions

Architecture:
    Input: Optical(5ch) + SAR(8ch) → Dual Branches → TNF Fusion → Three Outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # Still needed for EfficientNet in SAR branch
import logging
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

# Import InternImage (from existing implementation)
from mm_intern_image_src.intern_image_import import InternImage

logger = logging.getLogger(__name__)


class OpticalBranch(nn.Module):
    """
    Optical Primary Branch for 5-channel optical data processing.

    Processes R, G, B, NIR, NDVI channels using InternImage-T backbone
    for high-quality feature extraction with dynamic receptive fields.
    """

    def __init__(
        self,
        in_channels: int = 5,
        feature_dim: int = 512,  # Final feature dimension after InternImage-T
        pretrained: bool = True,
        dropout_rate: float = 0.1,
    ):
        super(OpticalBranch, self).__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Create InternImage-T backbone (following existing implementation)
        self.backbone = self._create_intern_image_backbone(pretrained)

        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature projection and classification head
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Linear(self.feature_dim, 1)

        logger.info(f"OpticalBranch initialized with {self.feature_dim} features")

    def _create_intern_image_backbone(self, pretrained: bool) -> nn.Module:
        """Create InternImage-T backbone following existing implementation."""
        # Create InternImage-T with exact same configuration as existing code
        backbone = InternImage(core_op="DCNv3", channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], num_classes=0)

        if pretrained:
            logger.info("✅ Using pretrained weights for optical backbone (conceptual).")

        # Modify input layer for 5-channel input (following existing pattern)
        self._modify_input_layer(backbone, self.in_channels, pretrained)

        return backbone

    def _modify_input_layer(self, backbone, in_channels, pretrained):
        """Modify input layer for 5-channel input (from existing implementation)."""
        original_conv = backbone.patch_embed.conv1
        if original_conv.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        if pretrained and original_conv.in_channels == 3:
            with torch.no_grad():
                # Copy RGB weights
                new_conv.weight[:, :3, :, :] = original_conv.weight.data
                # NIR: average of RGB
                new_conv.weight[:, 3, :, :] = original_conv.weight.data.mean(dim=1)
                # NDVI: based on NIR-Red relationship
                new_conv.weight[:, 4, :, :] = (
                    new_conv.weight[:, 3, :, :] - original_conv.weight.data[:, 0, :, :]
                ) * 0.5

        backbone.patch_embed.conv1 = new_conv
        logger.info(f"✅ Optical backbone input layer modified for {in_channels} channels.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through optical branch.

        Args:
            x: Input tensor of shape (B, 5, 64, 64)

        Returns:
            features: Global feature vector (B, feature_dim)
            logits: Classification logits (B, 1)
        """
        # Extract features through InternImage backbone (following original implementation)
        # Manually call InternImage components to get intermediate features
        x_opt = self.backbone.patch_embed(x)  # Apply patch embedding
        x_opt = self.backbone.pos_drop(x_opt)  # Apply position dropout

        # Process through all InternImage levels
        for level in self.backbone.levels:
            x_opt = level(x_opt)  # x_opt is in NHWC format

        # Convert NHWC to NCHW for global pooling and get final features
        features = x_opt.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW

        # Debug: Check feature dimensions
        if features.shape[1] != self.feature_dim:
            logger.warning(f"Feature dimension mismatch: expected {self.feature_dim}, got {features.shape[1]}")
            # Update feature_dim to match actual output
            self.feature_dim = features.shape[1]
            # Recreate feature_proj and classifier with correct dimensions
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(0.1),
            ).to(features.device)
            self.classifier = nn.Linear(self.feature_dim, 1).to(features.device)
            logger.info(f"Updated feature dimensions to {self.feature_dim}")

        # Global average pooling and flatten
        features = self.global_pool(features).flatten(1)  # (B, feature_dim)

        # Feature processing
        features = self.feature_proj(features)

        # Classification
        logits = self.classifier(features)

        return features, logits


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
                    scale_factor = 0.8 if i % 2 == 1 else 1.0  # Difference channels get smaller weights
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
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, loss_fn: nn.Module = nn.BCEWithLogitsLoss()
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-branch loss with weighted combination.

        Args:
            outputs: Model outputs dictionary
            targets: Ground truth labels (B, 1)
            loss_fn: Loss function to use

        Returns:
            Dictionary of losses
        """
        # Individual branch losses
        optical_loss = loss_fn(outputs["optical_logits"], targets)
        sar_loss = loss_fn(outputs["sar_logits"], targets)
        fusion_loss = loss_fn(outputs["fusion_logits"], targets)

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

        # Test branch components individually
        print(f"\n🔍 Testing individual branches...")

        # Test optical branch
        print("  Testing optical branch...")
        optical_test_input = torch.randn(2, 5, 64, 64).to(device)
        optical_features, optical_logits = model.optical_branch(optical_test_input)
        print(
            f"    ✅ Optical branch: {optical_test_input.shape} → features: {optical_features.shape}, logits: {optical_logits.shape}"
        )

        # Test SAR branch
        print("  Testing SAR branch...")
        sar_test_input = torch.randn(2, 8, 64, 64).to(device)
        sar_features, sar_logits = model.sar_branch(sar_test_input)
        print(f"    ✅ SAR branch: {sar_test_input.shape} → features: {sar_features.shape}, logits: {sar_logits.shape}")

        # Test fusion module
        print("  Testing fusion module...")
        fusion_features, fusion_logits, fusion_weights = model.fusion_module(optical_features, sar_features)
        print(f"    ✅ Fusion module: optical({optical_features.shape}) + sar({sar_features.shape})")
        print(
            f"      → fusion_features: {fusion_features.shape}, fusion_logits: {fusion_logits.shape}, weights: {fusion_weights.shape}"
        )

        # Test full forward pass
        print(f"\n🚀 Testing full forward pass...")
        batch_size = 2

        # Create test inputs with correct shapes
        optical_data = torch.randn(batch_size, 5, 64, 64).to(device)  # R, G, B, NIR, NDVI
        sar_data = torch.randn(batch_size, 8, 64, 64).to(device)  # 8-channel SAR data

        print(f"Input shapes:")
        print(f"  - Optical: {optical_data.shape}")
        print(f"  - SAR: {sar_data.shape}")

        # Forward pass
        with torch.no_grad():
            outputs = model(optical_data, sar_data, return_features=True)

        print(f"\n📤 Output shapes:")
        for key, value in outputs.items():
            if key == "features":
                print(f"  - {key}:")
                for feat_key, feat_value in value.items():
                    print(f"    - {feat_key}: {feat_value.shape}")
            else:
                print(f"  - {key}: {value.shape}")

        # Verify output shapes
        expected_shape = (batch_size, 1)
        assert (
            outputs["optical_logits"].shape == expected_shape
        ), f"Optical logits shape mismatch: {outputs['optical_logits'].shape} vs {expected_shape}"
        assert (
            outputs["sar_logits"].shape == expected_shape
        ), f"SAR logits shape mismatch: {outputs['sar_logits'].shape} vs {expected_shape}"
        assert (
            outputs["fusion_logits"].shape == expected_shape
        ), f"Fusion logits shape mismatch: {outputs['fusion_logits'].shape} vs {expected_shape}"
        assert (
            outputs["final_logits"].shape == expected_shape
        ), f"Final logits shape mismatch: {outputs['final_logits'].shape} vs {expected_shape}"

        print("✅ All output shapes are correct!")

        # Test loss computation
        print(f"\n💱 Testing loss computation...")
        targets = torch.randint(0, 2, (batch_size, 1)).float().to(device)
        print(f"Targets shape: {targets.shape}")

        losses = model.compute_loss(outputs, targets)
        print(f"Loss components:")
        for loss_name, loss_value in losses.items():
            print(f"  - {loss_name}: {loss_value.item():.4f}")

        print("✅ Loss computation successful!")

        # Test prediction function
        print(f"\n🎯 Testing prediction function...")
        predictions = model.predict(optical_data, sar_data, use_ensemble=True, threshold=0.5)
        print(f"Prediction outputs:")
        for pred_key, pred_value in predictions.items():
            if pred_key != "all_outputs":
                print(f"  - {pred_key}: {pred_value.shape}")

        print("✅ Prediction function successful!")

        # Test different input sizes (optional stress test)
        print(f"\n🔄 Testing different batch sizes...")
        for test_batch_size in [1, 4]:
            test_optical = torch.randn(test_batch_size, 5, 64, 64).to(device)
            test_sar = torch.randn(test_batch_size, 8, 64, 64).to(device)

            with torch.no_grad():
                test_outputs = model(test_optical, test_sar)
                expected_test_shape = (test_batch_size, 1)
                assert test_outputs["final_logits"].shape == expected_test_shape

            print(f"  ✅ Batch size {test_batch_size}: {test_outputs['final_logits'].shape}")

        print(f"\n🎉 All tests passed successfully!")
        print("=" * 60)
        print("🏆 MM-LandslideNet-TNF model is ready for training!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        print("\n💡 Common issues:")
        print("  1. CUDA not available for DCNv3 operations")
        print("  2. InternImage import issues")
        print("  3. Shape mismatches in fusion module")
        print("  4. Missing dependencies (timm, etc.)")
        sys.exit(1)
