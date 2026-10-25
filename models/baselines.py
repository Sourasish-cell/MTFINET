"""
Baseline Architectures for Ablation and Comparative Experiments
"""

import torch
import torch.nn as nn
from torchvision import models

class SingleTaskResNet(nn.Module):
    """Single-task ResNet-18 for either Metal classification OR Concentration classification."""
    def __init__(self, num_classes=8, pretrained=True):
        super(SingleTaskResNet, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class MultiTaskMobileNetV2(nn.Module):
    """Lightweight mobile edge baseline using MobileNetV2 backbone."""
    def __init__(self, num_metals=8, num_concentrations=7, pretrained=True):
        super(MultiTaskMobileNetV2, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.metal_head = nn.Linear(num_features, num_metals)
        self.conc_head = nn.Linear(num_features, num_concentrations)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.metal_head(features), self.conc_head(features)
