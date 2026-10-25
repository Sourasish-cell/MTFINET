# MTFINet Model Checkpoints

This directory contains the trained PyTorch state dictionary for the Multi-Task Fluorescence Intelligence Network (MTFINet).

### Model File:
- `best_multitask_model_leakage_free.pth` (42.7 MB)
  - **Backbone:** ResNet-18 (`torchvision.models.ResNet18_Weights.DEFAULT`)
  - **Training Protocol:** Partitioned strictly by physical sample ID (GroupShuffleSplit, 0% leakage).
  - **Normalization:** ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - **Test Performance on 818 Held-Out Samples:**
    - Metal Classification Accuracy: **92.67%** (Macro F1: 93.61%)
    - Concentration-Class Accuracy: **97.43%** (TEC: 2.57%)
    - Mapped-Class Mean Absolute Error (QER): **2.00 µM**
    - Quadratic Weighted Kappa (QWK): **0.9850**

### How to Load in PyTorch:
```python
import torch
from models.mtfinet import MultiTaskResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskResNet(num_metals=8, num_concentrations=7, pretrained=False)
model.load_state_dict(torch.load('checkpoints/best_multitask_model_leakage_free.pth', map_location=device))
model.eval()
```
