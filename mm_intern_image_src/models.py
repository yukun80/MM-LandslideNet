"""
Model Architecture for MM-InternImage-TNF

This module implements the complete model architecture combining three InternImage-T
backbones with a TNF-style fusion block for multi-modal landslide detection.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pathlib import Path

# Import InternImage using helper module
try:
    # Try relative import first (when imported as part of package)
    from .intern_image_import import InternImage
    from .config import config
except ImportError:
    # Fall back to absolute import (when run directly)
    from intern_image_import import InternImage
    from config import config


class TNFFusionBlock(nn.Module):
    """
    TNF-style Fusion Block inspired by the TNF paper.

    This block implements:
    1. Self-attention across modalities
    2. Cross-attention between modalities
    3. Gated fusion mechanism
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize TNF Fusion Block.

        Args:
            feature_dim: Dimension of input feature vectors
            hidden_dim: Hidden dimension for intermediate layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(TNFFusionBlock, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Input projection layers for each modality
        self.optical_proj = nn.Linear(feature_dim, hidden_dim)
        self.sar_proj = nn.Linear(feature_dim, hidden_dim)
        self.sar_diff_proj = nn.Linear(feature_dim, hidden_dim)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention layers
        self.cross_attention_optical = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Gate mechanism
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 3), nn.Softmax(dim=-1)
        )

        # Final fusion layers
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Residual connection
        self.residual_proj = nn.Linear(feature_dim * 3, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, optical_feat: torch.Tensor, sar_feat: torch.Tensor, sar_diff_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TNF Fusion Block.

        Args:
            optical_feat: Optical features [B, feature_dim]
            sar_feat: SAR features [B, feature_dim]
            sar_diff_feat: SAR difference features [B, feature_dim]

        Returns:
            Fused feature vector [B, hidden_dim]
        """
        batch_size = optical_feat.size(0)

        # Project features to hidden dimension
        optical_proj = self.optical_proj(optical_feat)  # [B, hidden_dim]
        sar_proj = self.sar_proj(sar_feat)  # [B, hidden_dim]
        sar_diff_proj = self.sar_diff_proj(sar_diff_feat)  # [B, hidden_dim]

        # Stack features for self-attention [B, 3, hidden_dim]
        modality_features = torch.stack([optical_proj, sar_proj, sar_diff_proj], dim=1)

        # Self-attention across modalities
        self_attn_out, _ = self.self_attention(modality_features, modality_features, modality_features)
        self_attn_out = self.norm1(self_attn_out + modality_features)  # Residual connection

        # Extract enhanced features
        optical_enhanced = self_attn_out[:, 0, :]  # [B, hidden_dim]
        sar_enhanced = self_attn_out[:, 1, :]  # [B, hidden_dim]
        sar_diff_enhanced = self_attn_out[:, 2, :]  # [B, hidden_dim]

        # Cross-attention: Use optical as query, SAR modalities as key/value
        sar_context = torch.stack([sar_enhanced, sar_diff_enhanced], dim=1)  # [B, 2, hidden_dim]
        optical_query = optical_enhanced.unsqueeze(1)  # [B, 1, hidden_dim]

        cross_attn_out, _ = self.cross_attention_optical(optical_query, sar_context, sar_context)
        cross_attn_out = self.norm2(cross_attn_out.squeeze(1) + optical_enhanced)

        # Gate mechanism for adaptive fusion
        gate_input = torch.cat([cross_attn_out, sar_enhanced, sar_diff_enhanced], dim=-1)
        gate_weights = self.gate_fc(gate_input)  # [B, 3]

        # Apply gate weights
        gated_optical = gate_weights[:, 0:1] * cross_attn_out
        gated_sar = gate_weights[:, 1:2] * sar_enhanced
        gated_sar_diff = gate_weights[:, 2:3] * sar_diff_enhanced

        # Concatenate gated features
        gated_features = torch.cat([gated_optical, gated_sar, gated_sar_diff], dim=-1)

        # Final fusion
        fused_features = self.fusion_fc(gated_features)
        fused_features = self.norm3(fused_features)

        # Residual connection from original concatenated features
        residual = self.residual_proj(torch.cat([optical_feat, sar_feat, sar_diff_feat], dim=-1))
        fused_features = fused_features + residual

        return self.dropout(fused_features)


class InternImageBackbone(nn.Module):
    """
    Wrapper for InternImage backbone to handle different input channels
    and extract features before classification head.
    """

    def __init__(self, in_channels: int, pretrained: bool = False, **intern_config):
        """
        Initialize InternImage backbone.

        Args:
            in_channels: Number of input channels
            pretrained: Whether to use pretrained weights (only for 3-channel)
            **intern_config: InternImage configuration parameters
        """
        super(InternImageBackbone, self).__init__()

        # Create InternImage model without classification head
        self.backbone = InternImage(num_classes=0, **intern_config)  # No classification head

        # Modify input layer if not 3 channels
        if in_channels != 3:
            # Get original patch embedding
            original_patch_embed = self.backbone.patch_embed

            # Create new patch embedding with correct input channels
            self.backbone.patch_embed = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
                original_patch_embed,
            )

        # Load pretrained weights if requested (only for optical branch with 5 channels)
        if pretrained and in_channels == 5:
            self._load_pretrained_weights()

        # Set feature dimension based on InternImage-T architecture
        # We know from testing that forward_features returns 768-dim features
        self.feature_dim = 768

    def _load_pretrained_weights(self):
        """Load pretrained InternImage-T weights."""
        try:
            # Get the path to the pretrained weights
            pretrained_path = Path(config.ROOT_DIR) / "configs" / "pre_checkpoints" / "internimage_t_1k_224.pth"
            
            if not pretrained_path.exists():
                print(f"Warning: Pretrained weights file not found at {pretrained_path}")
                return
                
            # Load the checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model' in checkpoint:
                pretrained_state_dict = checkpoint['model']
            else:
                pretrained_state_dict = checkpoint
                
            print(f"Loading pretrained weights from {pretrained_path}")
            print(f"Found {len(pretrained_state_dict)} parameters in pretrained model")
            
            # For our modified architecture, we need to load weights into the last part
            # of the patch_embed sequence (the original InternImage patch embedding)
            # Skip the first 3 layers (Conv2d, BatchNorm2d, ReLU) we added
            
            # Get the original patch embed (the 4th element in our sequential)
            original_patch_embed = self.backbone.patch_embed[3]
            
            # Load pretrained weights into original patch embed
            original_state_dict = original_patch_embed.state_dict()
            new_state_dict = {}
            loaded_params = 0
            
            for name, param in original_state_dict.items():
                if name in pretrained_state_dict:
                    new_state_dict[name] = pretrained_state_dict[name]
                    loaded_params += 1
                else:
                    new_state_dict[name] = param
                    
            # Load the weights
            original_patch_embed.load_state_dict(new_state_dict, strict=False)
            
            # Now load the rest of the backbone weights
            backbone_state_dict = self.backbone.state_dict()
            
            for name, param in backbone_state_dict.items():
                if name.startswith('patch_embed.') and not name.startswith('patch_embed.3.'):
                    # Skip our added layers
                    continue
                elif name.startswith('patch_embed.3.'):
                    # Already loaded above
                    continue
                else:
                    # Load other backbone weights
                    if name in pretrained_state_dict:
                        backbone_state_dict[name] = pretrained_state_dict[name]
                        loaded_params += 1
                        
            # Load the updated state dict
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f"Successfully loaded {loaded_params} parameters from pretrained weights")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with random initialization...")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature vector [B, feature_dim]
        """
        # Extract features using the backbone
        features = self.backbone.forward_features(x)
        return features


class MMInternImageTNF(nn.Module):
    """
    Multi-Modal InternImage with TNF Fusion for landslide detection.

    This model combines three InternImage-T backbones for processing
    optical, SAR, and SAR difference data, followed by a TNF-style
    fusion block and classification head.
    """

    def __init__(self, num_classes: int = 1, pretrained_optical: bool = True, **kwargs):
        """
        Initialize MM-InternImage-TNF model.

        Args:
            num_classes: Number of output classes (1 for binary classification)
            pretrained_optical: Whether to use pretrained weights for optical branch
        """
        super(MMInternImageTNF, self).__init__()

        # Get InternImage configuration
        intern_config = config.INTERNIMAGE_CONFIG.copy()

        # Create three InternImage backbones
        self.optical_backbone = InternImageBackbone(
            in_channels=config.OPTICAL_CHANNELS,  # 5 channels (R,G,B,NIR,NDVI)
            pretrained=pretrained_optical,
            **intern_config,
        )

        self.sar_backbone = InternImageBackbone(
            in_channels=config.SAR_CHANNELS,  # 4 channels
            pretrained=False,  # No pretrained weights for SAR
            **intern_config,
        )

        self.sar_diff_backbone = InternImageBackbone(
            in_channels=config.SAR_DIFF_CHANNELS,  # 4 channels
            pretrained=False,  # No pretrained weights for SAR diff
            **intern_config,
        )

        # Get feature dimensions
        self.feature_dim = self.optical_backbone.feature_dim

        # TNF Fusion Block
        fusion_config = config.FUSION_CONFIG.copy()
        fusion_config["feature_dim"] = self.feature_dim  # Use dynamic feature_dim from backbone
        self.fusion_block = TNFFusionBlock(**fusion_config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_CONFIG["hidden_dim"], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of MM-InternImage-TNF.

        Args:
            batch: Dictionary containing 'optical', 'sar', 'sar_diff' tensors

        Returns:
            Classification logits [B, num_classes]
        """
        # Extract modality data
        optical = batch["optical"]  # [B, 5, H, W]
        sar = batch["sar"]  # [B, 4, H, W]
        sar_diff = batch["sar_diff"]  # [B, 4, H, W]

        # Forward through individual backbones
        optical_feat = self.optical_backbone(optical)  # [B, feature_dim]
        sar_feat = self.sar_backbone(sar)  # [B, feature_dim]
        sar_diff_feat = self.sar_diff_backbone(sar_diff)  # [B, feature_dim]

        # Fusion through TNF block
        fused_feat = self.fusion_block(optical_feat, sar_feat, sar_diff_feat)

        # Classification
        logits = self.classifier(fused_feat)

        return logits

    def get_feature_maps(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for analysis.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary containing feature maps from each branch
        """
        with torch.no_grad():
            optical_feat = self.optical_backbone(batch["optical"])
            sar_feat = self.sar_backbone(batch["sar"])
            sar_diff_feat = self.sar_diff_backbone(batch["sar_diff"])
            fused_feat = self.fusion_block(optical_feat, sar_feat, sar_diff_feat)

        return {
            "optical_features": optical_feat,
            "sar_features": sar_feat,
            "sar_diff_features": sar_diff_feat,
            "fused_features": fused_feat,
        }


def create_model(num_classes: int = 1, pretrained_optical: bool = True, device: str = None) -> MMInternImageTNF:
    """
    Factory function to create MM-InternImage-TNF model.

    Args:
        num_classes: Number of output classes
        pretrained_optical: Whether to use pretrained weights for optical branch
        device: Device to move model to

    Returns:
        Initialized model
    """
    if device is None:
        device = config.DEVICE

    model = MMInternImageTNF(num_classes=num_classes, pretrained_optical=pretrained_optical)

    model = model.to(device)

    # Print model information
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import count_parameters
    except ImportError:
        # Fallback to manual count if import fails
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"MM-InternImage-TNF created with {total_params:,} trainable parameters")
        return model

    total_params = count_parameters(model)
    print(f"MM-InternImage-TNF created with {total_params:,} trainable parameters")

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()

    # Test forward pass
    batch_size = 2
    test_batch = {
        "optical": torch.randn(batch_size, 5, 64, 64),
        "sar": torch.randn(batch_size, 4, 64, 64),
        "sar_diff": torch.randn(batch_size, 4, 64, 64),
    }

    with torch.no_grad():
        output = model(test_batch)
        print(f"Model output shape: {output.shape}")

        # Test feature extraction
        features = model.get_feature_maps(test_batch)
        for name, feat in features.items():
            print(f"{name}: {feat.shape}")
