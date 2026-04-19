"""Model creation and loading utilities.

Supports three backbone architectures for plant disease classification
- ResNet-18: Simple baseline (~11M params)
- EfficientNet-B0: Best accuracy/parameter ratio (~5.3M params)
- ConvNeXt-Tiny: Highest accuracy (~28M params)

All backbones use IMAGENET1K_V1 pretrained weights by default.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes, backbone="convnext_tiny", pretrained=True):
    """Creates a classification model with the specified backbone.

    Replaces the original ImageNet classifier head with a new one
    mapping to the specified number of disease classes.

    Args:
        num_classes (int): Number of output classes (39 for the disease set).
        backbone (str): Architecture name. Choices:
            - 'convnext_tiny': ConvNeXt-Tiny, replaces classifier[2] with Linear.
            - 'efficientnet_b0': EfficientNet-B0, replaces classifier with Dropout + Linear.
            - 'resnet18': ResNet-18, replaces fc with Linear.
        pretrained (bool): If True => load IMAGENET1K_V1 weights. Default True.

    Returns:
        nn.Module: Model with new classifier head, ready for training.

    Raises:
        ValueError: If backbone name is not recognized.
    """
    if backbone == "convnext_tiny":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.convnext_tiny(weights=weights)
        # Replace final Linear layer (original - 1000 ImageNet classes)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif backbone == "efficientnet_b0":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # Replace classifier with Dropout (regularization) + Linear
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif backbone == "resnet18":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.resnet18(weights=weights)
        # Replace final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


def load_model(path, num_classes, backbone="convnext_tiny"):
    """Load a trained model from a saved checkpoint.

    Creates the model architecture (without pretrained weights),
    then loads the saved state dict from the checkpoint file.

    Args:
        path (str): Path to the .pth checkpoint file.
        num_classes (int): Number of output classes (must match saved model).
        backbone (str): Architecture used when the model was trained.

    Returns:
        nn.Module: Model loaded with trained weights, set to eval mode,
                   moved to available device (GPU if available, else CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes, backbone, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
