"""
Multi-Task Fluorescence Intelligence Network (MTFINet)
Architecture: Shared ResNet-18 Backbone with Dual Linear Classification Heads.
"""

import torch
import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    """
    MTFINet architecture based on a shared ResNet-18 feature extractor.
    
    Branch 1 (metal_head): Identifies 1 of 8 heavy metal labels (Categorical Cross-Entropy)
    Branch 2 (conc_head): Predicts 1 of 7 discrete concentration classes (Categorical Cross-Entropy)
    """
    def __init__(self, num_metals=8, num_concentrations=7, pretrained=True):
        super(MultiTaskResNet, self).__init__()
        
        # Load backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.base_model = models.resnet18(weights=weights)
        
        # Extract feature dimension before the original linear layer (512 for ResNet-18)
        num_features = self.base_model.fc.in_features
        
        # Replace original classification head with Identity to output feature vector f (512-D)
        self.base_model.fc = nn.Identity()
        
        # Dual task-specific linear heads
        self.metal_head = nn.Linear(num_features, num_metals)
        self.conc_head = nn.Linear(num_features, num_concentrations)
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input image batch of shape (B, 3, 224, 224)
        Returns:
            tuple: (metal_logits, conc_logits) of shapes (B, num_metals) and (B, num_concentrations)
        """
        # Shared spatial feature extraction through residual blocks and global pooling
        shared_features = self.base_model(x)  # Shape: (B, 512)
        
        # Task-specific logits
        metal_logits = self.metal_head(shared_features)
        conc_logits = self.conc_head(shared_features)
        
        return metal_logits, conc_logits

def get_mtfinet(num_metals=8, num_concentrations=7, pretrained=True):
    return MultiTaskResNet(num_metals=num_metals, num_concentrations=num_concentrations, pretrained=pretrained)
