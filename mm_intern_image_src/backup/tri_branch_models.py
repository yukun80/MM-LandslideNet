import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import traceback
from typing import Dict, List

# --- InternImage and DCNv3 Imports ---
from .intern_image_import import InternImage

"""python -m mm_intern_image_src.models"""


# --- Lightweight SAR Backbones ---
class LightweightSARCNN(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 16, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.stages = nn.ModuleList()

        current_channels = in_channels
        output_channels = base_channels

        # The first stage does not downsample, subsequent stages do.
        for i in range(num_levels):
            stride = 2 if i > 0 else 1
            stage = nn.Sequential(
                nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.SiLU(inplace=True),
            )
            self.stages.append(stage)
            current_channels = output_channels
            if i < num_levels - 1:  # Don't double channels for the last stage
                output_channels *= 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        # The first stage of SAR should match the first stage of Optical (64x64)
        # The InternImage patch_embed downsamples by 4, so we apply a similar logic here.
        x = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)

        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class MediumSARChangeCNN(LightweightSARCNN):
    def __init__(self, in_channels: int = 4, base_channels: int = 32, num_levels=4):
        super().__init__(in_channels, base_channels, num_levels)


# --- Advanced Fusion Blocks ---
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=32):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.scale = head_dim**-0.5
        self.num_heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, query, context):
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: t.view(t.shape[0], -1, self.num_heads, t.shape[-1] // self.num_heads).transpose(1, 2), (q, k, v)
        )

        attention = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(query.shape[0], -1, (self.num_heads * (v.shape[-1])))
        return self.to_out(out)


class IntermediateFusionBlock(nn.Module):
    def __init__(self, optical_dim, sar_dim, change_dim):
        super().__init__()

        self.sar_fusion = CrossAttention(optical_dim, sar_dim)
        self.change_fusion = CrossAttention(optical_dim, change_dim)

        self.norm1 = nn.LayerNorm(optical_dim)
        self.norm2 = nn.LayerNorm(optical_dim)
        self.ffn = nn.Sequential(
            nn.Linear(optical_dim, optical_dim * 4), nn.GELU(), nn.Linear(optical_dim * 4, optical_dim)
        )
        self.norm3 = nn.LayerNorm(optical_dim)

    def forward(self, opt_feat, sar_feat, change_feat):
        B, C, H, W = opt_feat.shape

        # Align spatial dimensions before fusion
        sar_feat_aligned = F.interpolate(sar_feat, size=(H, W), mode="bilinear", align_corners=False)
        change_feat_aligned = F.interpolate(change_feat, size=(H, W), mode="bilinear", align_corners=False)

        opt_flat = opt_feat.flatten(2).transpose(1, 2)
        sar_flat = sar_feat_aligned.flatten(2).transpose(1, 2)
        change_flat = change_feat_aligned.flatten(2).transpose(1, 2)

        fused_sar = self.norm1(opt_flat + self.sar_fusion(opt_flat, sar_flat))
        fused_change = self.norm2(fused_sar + self.change_fusion(fused_sar, change_flat))
        fused_out = self.norm3(fused_change + self.ffn(fused_change))

        return fused_out.transpose(1, 2).view(B, C, H, W)


# --- Main Model ---
class OpticalDominatedCooperativeModel(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.optical_backbone = self._create_optical_backbone(pretrained)

        self.optical_dims = [64, 128, 256, 512]  # For InternImage-T

        self.sar_auxiliary = LightweightSARCNN(base_channels=16, num_levels=len(self.optical_dims))
        self.sar_change = MediumSARChangeCNN(base_channels=32, num_levels=len(self.optical_dims))

        self.sar_dims = [16 * (2**i) for i in range(len(self.optical_dims))]
        self.change_dims = [32 * (2**i) for i in range(len(self.optical_dims))]

        self.fusion_blocks = nn.ModuleList()
        for i in range(len(self.optical_dims)):
            self.fusion_blocks.append(
                IntermediateFusionBlock(self.optical_dims[i], self.sar_dims[i], self.change_dims[i])
            )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(self.optical_dims[-1]),
            nn.Linear(self.optical_dims[-1], 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _create_optical_backbone(self, pretrained: bool):
        backbone = InternImage(core_op="DCNv3", channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], num_classes=0)
        if pretrained:
            print("✅ Using pretrained weights for optical backbone (conceptual).")
        self._modify_input_layer(backbone, 5, pretrained)
        return backbone

    def _modify_input_layer(self, backbone, in_channels, pretrained):
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
                new_conv.weight[:, :3, :, :] = original_conv.weight.data
                new_conv.weight[:, 3, :, :] = original_conv.weight.data.mean(dim=1)
                new_conv.weight[:, 4, :, :] = (
                    new_conv.weight[:, 3, :, :] - original_conv.weight.data[:, 0, :, :]
                ) * 0.5

        backbone.patch_embed.conv1 = new_conv
        print(f"✅ Optical backbone input layer modified for {in_channels} channels.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        optical, sar, sar_change = batch["optical"], batch["sar"], batch["sar_change"]

        sar_feats = self.sar_auxiliary(sar)
        change_feats = self.sar_change(sar_change)

        # --- Main Multi-Level Fusion Loop ---
        x_opt = self.optical_backbone.patch_embed(optical)
        x_opt = self.optical_backbone.pos_drop(x_opt)

        for i, level in enumerate(self.optical_backbone.levels):
            # Permute to NCHW for fusion
            opt_feat_nchw = x_opt.permute(0, 3, 1, 2).contiguous()

            # Perform intermediate fusion
            fused_feat_nchw = self.fusion_blocks[i](opt_feat_nchw, sar_feats[i], change_feats[i])

            # Permute back to NHWC for the next InternImage level
            x_opt = fused_feat_nchw.permute(0, 2, 3, 1).contiguous()
            x_opt = level(x_opt)

        final_fused_features = x_opt.permute(0, 3, 1, 2).contiguous()
        logits = self.classifier(final_fused_features)
        return logits


def create_optical_dominated_model(num_classes: int = 1, pretrained: bool = True):
    return OpticalDominatedCooperativeModel(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    print("🔬 Verifying Fused OpticalDominatedCooperativeModel...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. This model requires a GPU for DCNv3 ops. Exiting.")
        sys.exit(0)

    model = create_optical_dominated_model(pretrained=False).to(device)
    model.eval()

    batch_size = 2
    dummy_batch = {
        "optical": torch.randn(batch_size, 5, 64, 64).to(device),
        "sar": torch.randn(batch_size, 4, 64, 64).to(device),
        "sar_change": torch.randn(batch_size, 4, 64, 64).to(device),
    }

    print(f"Input shapes: { {k: v.shape for k, v in dummy_batch.items()} }")

    try:
        with torch.no_grad():
            output = model(dummy_batch)
            print(f"\n✅ Model forward pass successful!")
            print(f"Output shape: {output.shape}")
            assert output.shape == (batch_size, 1), "Output shape is incorrect!"

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params:,}")
            print("\nVerification successful!")

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        traceback.print_exc()
