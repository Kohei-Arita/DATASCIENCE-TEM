"""
RSNA Aneurysm Detection - Model Architectures

PyTorch model implementations for medical image classification.
Supports various CNN architectures with medical imaging optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, List, Dict, Any
import timm
import logging

logger = logging.getLogger(__name__)


class AneurysmClassifier(nn.Module):
    """
    Deep learning classifier for intracranial aneurysm detection from medical images

    A flexible PyTorch model supporting multiple CNN architectures (ResNet, EfficientNet, 
    Vision Transformers, etc.) with medical imaging optimizations. Designed specifically 
    for binary classification of brain aneurysms from DICOM images.

    Key Features:
        - Multiple backbone architectures with ImageNet pretraining
        - Configurable classifier heads with dropout and batch normalization
        - Optional attention mechanisms (CBAM, SE, ECA)
        - Support for multi-scale input processing
        - Comprehensive parameter counting and model introspection
        - Medical imaging domain adaptations

    Architecture Overview:
        Input Image → Backbone CNN → [Optional Attention] → Classifier Head → Prediction
        
    Supported Backbones:
        - ResNet family (resnet18, resnet50, etc.)
        - EfficientNet family (efficientnet-b0 through b7)
        - Vision Transformers (vit_base_patch16_224, etc.)
        - ConvNeXt models
        - Swin Transformers
        - Any timm model (fallback)
        
    Example:
        >>> model = AneurysmClassifier(
        ...     architecture='resnet50',
        ...     num_classes=1,
        ...     pretrained=True,
        ...     use_attention=True,
        ...     attention_type='cbam'
        ... )
        >>> prediction = model(batch_images)  # Shape: (batch_size, 1)
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 512,
        num_layers: int = 2,
        use_attention: bool = False,
        attention_type: str = "cbam",
        activation: str = "relu",
        use_batch_norm: bool = True,
        **kwargs,
    ):
        """
        Args:
            architecture: Backbone architecture name
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            hidden_dim: Hidden dimension for classifier head
            num_layers: Number of layers in classifier head
            use_attention: Whether to use attention mechanism
            attention_type: Type of attention ('cbam', 'se', 'eca')
            activation: Activation function
            use_batch_norm: Use batch normalization in head
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Build backbone
        self.backbone, self.feature_dim = self._build_backbone(architecture, pretrained)

        # Build attention module
        if use_attention:
            self.attention = self._build_attention(attention_type, self.feature_dim)

        # Build classifier head
        self.classifier = self._build_classifier_head(
            self.feature_dim, num_classes, hidden_dim, num_layers, dropout, activation, use_batch_norm
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Model created: {architecture} with {self._count_parameters():,} parameters")

    def _build_backbone(self, architecture: str, pretrained: bool) -> tuple:
        """Build backbone network"""

        if architecture.startswith("resnet"):
            return self._build_resnet(architecture, pretrained)
        elif architecture.startswith("efficientnet"):
            return self._build_efficientnet(architecture, pretrained)
        elif architecture.startswith("vit"):
            return self._build_vision_transformer(architecture, pretrained)
        elif architecture.startswith("convnext"):
            return self._build_convnext(architecture, pretrained)
        elif architecture.startswith("swin"):
            return self._build_swin_transformer(architecture, pretrained)
        else:
            # Try timm models
            return self._build_timm_model(architecture, pretrained)

    def _build_resnet(self, architecture: str, pretrained: bool) -> tuple:
        """Build ResNet backbone"""
        model_fn = getattr(models, architecture)
        backbone = model_fn(pretrained=pretrained)

        # Remove classifier
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, feature_dim

    def _build_efficientnet(self, architecture: str, pretrained: bool) -> tuple:
        """Build EfficientNet backbone using timm"""
        backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # Remove global pooling
        )

        feature_dim = backbone.num_features
        return backbone, feature_dim

    def _build_vision_transformer(self, architecture: str, pretrained: bool) -> tuple:
        """Build Vision Transformer backbone"""
        backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0)

        feature_dim = backbone.num_features
        return backbone, feature_dim

    def _build_convnext(self, architecture: str, pretrained: bool) -> tuple:
        """Build ConvNeXt backbone"""
        backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0, global_pool="")

        feature_dim = backbone.num_features
        return backbone, feature_dim

    def _build_swin_transformer(self, architecture: str, pretrained: bool) -> tuple:
        """Build Swin Transformer backbone"""
        backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0)

        feature_dim = backbone.num_features
        return backbone, feature_dim

    def _build_timm_model(self, architecture: str, pretrained: bool) -> tuple:
        """Build model using timm (fallback)"""
        try:
            backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0, global_pool="")
            feature_dim = backbone.num_features
            return backbone, feature_dim
        except Exception as e:
            logger.error(f"Failed to create model {architecture}: {e}")
            # Fallback to ResNet50
            logger.warning("Falling back to ResNet50")
            return self._build_resnet("resnet50", pretrained)

    def _build_attention(self, attention_type: str, feature_dim: int) -> nn.Module:
        """Build attention module"""
        if attention_type == "se":
            return SEBlock(feature_dim)
        elif attention_type == "cbam":
            return CBAM(feature_dim)
        elif attention_type == "eca":
            return ECABlock(feature_dim)
        else:
            return nn.Identity()

    def _build_classifier_head(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
    ) -> nn.Module:
        """Build classifier head"""

        layers = []
        current_dim = input_dim

        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "swish":
            act_fn = nn.SiLU
        else:
            act_fn = nn.ReLU

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(current_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(act_fn())
            current_dim = hidden_dim

        # Final layer
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(current_dim, num_classes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize classifier head weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Extract features
        features = self.backbone(x)

        # Global average pooling if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)

        # Apply attention
        if self.use_attention:
            features = self.attention(features)

        # Classify
        output = self.classifier(features)

        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        features = self.backbone(x)

        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)

        if self.use_attention:
            features = self.attention(features)

        return features


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size()

        # Squeeze
        y = x

        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (for feature maps)"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        # Note: Spatial attention not applicable for 1D features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        return x


class ChannelAttention(nn.Module):
    """Channel Attention for 1D features"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        return x * torch.sigmoid(y)


class ECABlock(nn.Module):
    """Efficient Channel Attention"""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        k = t if t % 2 else t + 1

        # Use linear layer instead of conv1d for 1D features
        self.conv = nn.Linear(1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For 1D features, just apply sigmoid gating
        y = torch.sigmoid(x)
        return x * y


class MultiScaleModel(AneurysmClassifier):
    """Multi-scale model for handling different input resolutions"""

    def __init__(self, scales: List[int] = [224, 384, 512], **kwargs):
        super().__init__(**kwargs)
        self.scales = scales

        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * len(scales), self.feature_dim), nn.ReLU(), nn.Dropout(kwargs.get("dropout", 0.5))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with multi-scale inputs"""
        if x.size(-1) == self.scales[0]:  # Single scale
            return super().forward(x)

        # Multi-scale feature extraction
        features_list = []
        for scale in self.scales:
            x_scaled = F.interpolate(x, size=(scale, scale), mode="bilinear", align_corners=False)
            features = self.backbone(x_scaled)

            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.flatten(1)

            features_list.append(features)

        # Fuse multi-scale features
        fused_features = torch.cat(features_list, dim=1)
        fused_features = self.fusion(fused_features)

        # Apply attention and classify
        if self.use_attention:
            fused_features = self.attention(fused_features)

        output = self.classifier(fused_features)
        return output


class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble forward pass"""
        outputs = []

        for model in self.models:
            outputs.append(model(x))

        # Weighted average
        ensemble_output = sum(w * out for w, out in zip(self.weights, outputs))
        return ensemble_output

    def eval(self):
        """Set all models to eval mode"""
        for model in self.models:
            model.eval()
        return self


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create model from config"""

    model_config = config.get("model", {})
    architecture = model_config.get("architecture", "resnet50")

    # Multi-scale model
    if model_config.get("use_multiscale", False):
        scales = model_config.get("scales", [224, 384, 512])
        return MultiScaleModel(scales=scales, **model_config)

    # Standard model
    return AneurysmClassifier(**model_config)


def load_model_weights(model: nn.Module, checkpoint_path: str, strict: bool = True) -> nn.Module:
    """Load model weights from checkpoint"""

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model weights loaded from {checkpoint_path}")

    except Exception as e:
        logger.error(f"Failed to load model weights from {checkpoint_path}: {e}")
        raise

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {"total": total_params, "trainable": trainable_params, "non_trainable": total_params - trainable_params}


if __name__ == "__main__":
    # Test model creation
    config = {"model": {"architecture": "resnet50", "num_classes": 1, "pretrained": True, "dropout": 0.5, "hidden_dim": 512}}

    model = create_model(config)
    print(f"Model created: {model.architecture}")
    print(f"Parameters: {count_parameters(model)}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
