"""Model creation and loading the utilities."""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes, backbone='convnext_tiny', pretrained=True):
    """Creates a model with the specified backbone."""
    if backbone == 'convnext_tiny':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif backbone == 'efficientnet_b0':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif backbone == 'resnet18':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


def load_model(path, num_classes, backbone='convnext_tiny'):
    """Load a trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes, backbone, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model