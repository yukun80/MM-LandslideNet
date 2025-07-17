"""
Model Architecture for Asymmetric Dual-Backbone Fusion Model

This module implements the complete model architecture combining a primary InternImage
backbone for optical data with a lightweight CNN for SAR data, fused at multiple
stages using a cross-attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

# Import InternImage using helper module
try:
    from .intern_image_import import get_intern_image
    from .config import config
except ImportError:
    from intern_image_import import get_intern_image
    from config import config

"""
python -m mm_intern_image_src.models
"""


class LightweightSARCNN(nn.Module):
    """
    Lightweight CNN for SAR data feature extraction, architecturally synchronized
    with the InternImage backbone's downsampling strategy.
    """

    def __init__(self, in_channels: int = 8, base_channels: int = 32):
        super().__init__()
        # This stem mimics the InternImage StemLayer: two stride-2 convs
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )
        in_ch = base_channels * 2  # 64

        self.stages = nn.ModuleList()
        self.stage_out_channels = [in_ch]

        # Subsequent stages each downsample by a factor of 2
        for i in range(3):  # 3 more stages
            out_ch = in_ch * 2
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
            )
            self.stage_out_channels.append(out_ch)
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass, returning features from each stage."""
        # Input: (B, 8, 64, 64)
        x = self.stem(x)  # -> (B, 64, 16, 16)

        features = [x]  # First feature map is from the stem
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # Expected output shapes for a (64,64) input:
        # features[0]: (B, 64, 16, 16)
        # features[1]: (B, 128, 8, 8)
        # features[2]: (B, 256, 4, 4)
        # features[3]: (B, 512, 2, 2)
        return features


class CrossFusionBlock(nn.Module):
    """
    Cross-attention fusion block with an adaptive gate.
    Fuses SAR features into the optical feature stream.
    """

    def __init__(self, optical_dim: int, sar_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.optical_dim = optical_dim
        self.sar_dim = sar_dim

        assert optical_dim % num_heads == 0, "optical_dim must be divisible by num_heads"

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=optical_dim, num_heads=num_heads, kdim=sar_dim, vdim=sar_dim, dropout=dropout, batch_first=True
        )

        self.gate = nn.Sequential(nn.Linear(optical_dim, 1), nn.Sigmoid())

        self.norm = nn.LayerNorm(optical_dim)

    def forward(self, optical_features: torch.Tensor, sar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            optical_features: (B, H*W, C_opt) - Query
            sar_features: (B, H*W, C_sar) - Key/Value
        """
        attn_output, _ = self.cross_attention(query=optical_features, key=sar_features, value=sar_features)
        gate_weights = self.gate(optical_features.mean(dim=1))
        gated_attn_output = attn_output * gate_weights.unsqueeze(-1)
        fused_flat = self.norm(optical_features + gated_attn_output)
        return fused_flat


class AsymmetricFusionModel(nn.Module):
    """
    Asymmetric Dual-Backbone model for multi-modal landslide detection.
    """

    def __init__(self, num_classes: int = 1, pretrained_optical: bool = True):
        super().__init__()
        InternImage = get_intern_image()
        intern_config = config.INTERNIMAGE_CONFIG.copy()

        self.optical_backbone = InternImage(num_classes=0, **intern_config)
        self._adapt_optical_input(self.optical_backbone, 5, pretrained_optical)

        self.sar_backbone = LightweightSARCNN(in_channels=8)

        # Define dimensions for each stage explicitly
        self.optical_dims = [self.optical_backbone.channels * (2**i) for i in range(self.optical_backbone.num_levels)]
        self.sar_dims = self.sar_backbone.stage_out_channels

        self.fusion_blocks = nn.ModuleList()
        for i in range(len(self.optical_dims)):
            self.fusion_blocks.append(CrossFusionBlock(optical_dim=self.optical_dims[i], sar_dim=self.sar_dims[i]))

        # The input to the head is the output of conv_head, which has a scaled dimension
        cls_scale = 1.5  # Default scale from InternImage
        final_feature_dim = int(self.optical_dims[-1] * cls_scale)
        self.head = nn.Linear(final_feature_dim, num_classes)

        self._initialize_weights()

    def _adapt_optical_input(self, backbone: nn.Module, in_channels: int, pretrained: bool):
        """Modify the patch_embed layer for 5-channel input and init weights."""
        original_conv = backbone.patch_embed.conv1
        original_weight = original_conv.weight.data

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        if pretrained:
            print(f"Loading pretrained weights for {in_channels}-channel input...")
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_weight
                new_conv.weight[:, 3, :, :] = original_weight.mean(dim=1)
                new_conv.weight[:, 4, :, :] = original_weight.mean(dim=1) - original_weight[:, 0, :, :]

        backbone.patch_embed.conv1 = new_conv
        print(f"Adapted optical backbone to accept {in_channels} channels.")

    def _initialize_weights(self):
        for m in self.sar_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        optical, sar = batch["optical"], batch["sar"]

        sar_features = self.sar_backbone(sar)

        x = self.optical_backbone.patch_embed(optical)
        x = self.optical_backbone.pos_drop(x)

        # --- Corrected Fusion Loop ---
        # Fuse the initial features before the first level
        B, H, W, C = x.shape
        x_flat = x.flatten(1, 2)
        sar_flat = sar_features[0].flatten(2).transpose(1, 2)
        fused_x_flat = self.fusion_blocks[0](x_flat, sar_flat)
        x = fused_x_flat.view(B, H, W, C)

        for i, level in enumerate(self.optical_backbone.levels):
            x = level(x)  # Apply the main block and downsampling

            # After the last level, there is no more fusion
            if i < len(self.fusion_blocks) - 1:
                B, H, W, C = x.shape
                x_flat = x.flatten(1, 2)
                sar_flat = sar_features[i + 1].flatten(2).transpose(1, 2)
                fused_x_flat = self.fusion_blocks[i + 1](x_flat, sar_flat)
                x = fused_x_flat.view(B, H, W, C)

        # Final layers mirroring InternImage.forward_features
        x = self.optical_backbone.conv_head(x.permute(0, 3, 1, 2))
        x = self.optical_backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


def create_model(num_classes: int = 1, pretrained_optical: bool = True, device: str = None) -> AsymmetricFusionModel:
    """
    Factory function to create the AsymmetricFusionModel.
    """
    if device is None:
        device = config.DEVICE

    model = AsymmetricFusionModel(num_classes=num_classes, pretrained_optical=pretrained_optical)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AsymmetricFusionModel created with {total_params:,} trainable parameters.")

    return model


if __name__ == "__main__":
    print("Verifying the AsymmetricFusionModel...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device=device)
    model.eval()

    batch_size = 2
    dummy_optical = torch.randn(batch_size, 5, 64, 64).to(device)
    dummy_sar = torch.randn(batch_size, 8, 64, 64).to(device)
    dummy_batch = {"optical": dummy_optical, "sar": dummy_sar}

    print(f"Input optical shape: {dummy_optical.shape}")
    print(f"Input SAR shape: {dummy_sar.shape}")

    try:
        with torch.no_grad():
            output = model(dummy_batch)
            print(f"\nModel forward pass successful!")
            print(f"Output shape: {output.shape}")
            assert output.shape == (batch_size, 1), "Output shape is incorrect!"
            print("\nVerification successful!")

    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback

        traceback.print_exc()
