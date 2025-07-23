import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Import optical baseline config and utils
from optical_src.config import OpticalBaselineConfig
from optical_src.utils import setup_logging, count_parameters, get_model_size

# Setup logger for this module
logger = logging.getLogger("optical_baseline.model")


class BaselineOpticalModel(nn.Module):
    """
    Optical baseline model using Swin Transformer for landslide detection.
    Processes 5-channel input: R, G, B, NIR, NDVI
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        super(BaselineOpticalModel, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Create the backbone model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the default classification head
            global_pool="avg",  # Use average pooling
        )

        # Get the feature dimension from the backbone
        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone feature dimension: {self.feature_dim}")

        # Modify the input layer to accept 5 channels instead of 3
        self._modify_input_layer()

        # Create custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),  # type: ignore
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_classes),  # type: ignore
        )

        logger.info(f"Created {model_name} with {num_classes} output classes")
        self._print_model_info()

    def _modify_input_layer(self) -> None:
        """
        Modify the first convolutional layer to accept 5 input channels
        while preserving pretrained weights where possible.
        """
        # Get the original patch embedding layer
        original_conv = self.backbone.patch_embed.proj  # type: ignore
        original_weight = original_conv.weight.data  # type: ignore

        # Create new convolutional layer with 5 input channels
        new_conv = nn.Conv2d(
            in_channels=5,  # Changed from 3 to 5
            out_channels=original_conv.out_channels,  # type: ignore
            kernel_size=original_conv.kernel_size,  # type: ignore
            stride=original_conv.stride,  # type: ignore
            padding=original_conv.padding,  # type: ignore
            bias=original_conv.bias is not None,  # type: ignore
        )

        # Initialize weights for the new layer
        with torch.no_grad():
            # Copy existing weights for first 3 channels (RGB)
            new_conv.weight[:, :3, :, :] = original_weight  # type: ignore

            # Initialize NIR channel (channel 3) by averaging RGB weights
            new_conv.weight[:, 3, :, :] = original_weight.mean(dim=1)  # type: ignore

            # Initialize NDVI channel (channel 4) by using NIR-Red pattern
            # NDVI relates to NIR and Red, so we use a combination
            new_conv.weight[:, 4, :, :] = (  # type: ignore
                original_weight[:, 0, :, :] + original_weight[:, 2, :, :]  # type: ignore
            ) / 2  # Red + NIR-like

            # Copy bias if it exists
            if original_conv.bias is not None:  # type: ignore
                new_conv.bias = original_conv.bias  # type: ignore

        # Replace the original conv layer
        self.backbone.patch_embed.proj = new_conv  # type: ignore

        logger.info("Modified input layer to accept 5 channels (R, G, B, NIR, NDVI)")
        logger.info(f"New input layer: {new_conv}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 5, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Upsample input from 64x64 to 224x224 for Swin Transformer
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract features using the backbone
        features = self.backbone(x)  # Shape: (batch_size, feature_dim)

        # Apply classification head
        output = self.classifier(features)  # Shape: (batch_size, num_classes)

        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.

        Args:
            x: Input tensor of shape (batch_size, 5, height, width)

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            # Upsample input from 64x64 to 224x224 for Swin Transformer
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            features = self.backbone(x)
        return features

    @classmethod
    def from_config(cls, config: OpticalBaselineConfig, variant: str = "swin_tiny") -> "BaselineOpticalModel":
        """
        Create model from configuration.

        Args:
            config: OpticalBaselineConfig instance
            variant: Model variant to use

        Returns:
            BaselineOpticalModel instance
        """
        model_config = config.get_model_config(variant)

        return cls(
            model_name=model_config["model_name"],
            num_classes=model_config["num_classes"],
            pretrained=model_config["pretrained"],
            dropout_rate=model_config["dropout_rate"],
        )

    def _print_model_info(self) -> None:
        """Print model architecture information."""
        total_params, trainable_params = count_parameters(self)
        model_size = get_model_size(self)

        logger.info(f"Model Architecture: {self.model_name}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {model_size:.2f} MB")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Classification head: {self.classifier}")


class ModelFactory:
    """Factory class for creating different model variants."""

    @staticmethod
    def create_model(
        model_type: str = "swin_tiny", config: Optional[OpticalBaselineConfig] = None, **kwargs
    ) -> BaselineOpticalModel:
        """
        Create a model based on the specified type.

        Args:
            model_type: Type of model to create
            config: OpticalBaselineConfig instance (creates default if None)
            **kwargs: Additional arguments for model creation

        Returns:
            BaselineOpticalModel instance
        """
        if config is None:
            config = OpticalBaselineConfig()

        if model_type not in config.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model type: {model_type}. " f"Available types: {list(config.MODEL_VARIANTS.keys())}"
            )

        # Get model configuration and override with kwargs
        model_config = config.get_model_config(model_type)
        model_config.update(kwargs)  # Override with any provided kwargs

        return BaselineOpticalModel(
            model_name=model_config["model_name"],
            num_classes=model_config["num_classes"],
            pretrained=model_config["pretrained"],
            dropout_rate=model_config["dropout_rate"],
        )


def test_model() -> BaselineOpticalModel:
    """Test the model with dummy data."""
    from .utils import get_device

    device = get_device()

    # Create model using configuration
    config = OpticalBaselineConfig()
    model = BaselineOpticalModel.from_config(config)
    model = model.to(device)
    model.eval()

    # Create dummy input (batch_size=2, channels=5, height=64, width=64)
    dummy_input = torch.randn(2, config.INPUT_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)

    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        features = model.get_features(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    return model


if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="INFO")

    # Test the model
    model = test_model()
    print("\nModel test completed successfully!")

    # Test different model variants
    print("\nTesting different model variants:")
    config = OpticalBaselineConfig()
    for model_type in ["swin_tiny", "swin_small"]:
        try:
            model = ModelFactory.create_model(model_type, config=config)
            print(f"✓ {model_type}: Created successfully")
        except Exception as e:
            print(f"✗ {model_type}: Failed - {str(e)}")
