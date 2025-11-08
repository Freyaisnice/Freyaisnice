"""
Multi-task model for dog stool classification.
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskStoolClassifier(nn.Module):
    """
    Multi-task classification model for dog stool analysis.

    This model has a shared backbone (CNN feature extractor) and multiple
    task-specific classification heads.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: Dict[str, int] = None,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            backbone: Name of the backbone architecture
            num_classes: Dictionary mapping task names to number of classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
        """
        super(MultiTaskStoolClassifier, self).__init__()

        if num_classes is None:
            num_classes = {
                'stool_type': 7,
                'consistency': 5,
                'color': 8,
                'health': 4
            }

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Initialize backbone
        self.backbone, self.feature_dim = self._create_backbone(
            backbone, pretrained
        )

        # Create task-specific classification heads
        self.classifiers = nn.ModuleDict()
        for task_name, num_class in num_classes.items():
            self.classifiers[task_name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(512, num_class)
            )

    def _create_backbone(self, backbone: str, pretrained: bool):
        """Create and return the backbone model."""
        if backbone == "efficientnet_b0":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            feature_dim = model.classifier[1].in_features
            # Remove classification layer
            model.classifier = nn.Identity()
            return model, feature_dim

        elif backbone == "efficientnet_b1":
            model = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            )
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, feature_dim

        elif backbone == "resnet50":
            model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feature_dim = model.fc.in_features
            # Remove final fully connected layer
            model.fc = nn.Identity()
            return model, feature_dim

        elif backbone == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            )
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
            return model, feature_dim

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Dictionary mapping task names to logits
        """
        # Extract features
        features = self.backbone(x)

        # Apply task-specific classifiers
        outputs = {}
        for task_name, classifier in self.classifiers.items():
            outputs[task_name] = classifier(features)

        return outputs


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with task weighting.
    """

    def __init__(self, loss_weights: Dict[str, float] = None):
        """
        Args:
            loss_weights: Dictionary mapping task names to loss weights
        """
        super(MultiTaskLoss, self).__init__()

        if loss_weights is None:
            loss_weights = {
                'stool_type': 1.0,
                'consistency': 1.0,
                'color': 1.0,
                'health': 1.5
            }

        self.loss_weights = loss_weights
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dictionary of model predictions (logits)
            targets: Dictionary of ground truth labels

        Returns:
            Dictionary containing total loss and individual task losses
        """
        losses = {}
        total_loss = 0.0

        for task_name in predictions.keys():
            if task_name in targets:
                task_loss = self.criterion(predictions[task_name], targets[task_name])
                weight = self.loss_weights.get(task_name, 1.0)
                weighted_loss = weight * task_loss

                losses[f'{task_name}_loss'] = task_loss
                total_loss += weighted_loss

        losses['total_loss'] = total_loss
        return losses


def create_model(config: Dict) -> MultiTaskStoolClassifier:
    """
    Create model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        MultiTaskStoolClassifier instance
    """
    model_config = config.get('model', {})
    labels_config = config.get('labels', {})

    # Calculate number of classes for each task
    num_classes = {
        task_name: len(classes)
        for task_name, classes in labels_config.items()
    }

    model = MultiTaskStoolClassifier(
        backbone=model_config.get('backbone', 'efficientnet_b0'),
        num_classes=num_classes,
        pretrained=model_config.get('pretrained', True),
        dropout=model_config.get('dropout', 0.3)
    )

    return model


if __name__ == "__main__":
    # Test model
    model = MultiTaskStoolClassifier(
        backbone="efficientnet_b0",
        num_classes={
            'stool_type': 7,
            'consistency': 5,
            'color': 8,
            'health': 4
        }
    )

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)

    print("Model output shapes:")
    for task_name, output in outputs.items():
        print(f"{task_name}: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
