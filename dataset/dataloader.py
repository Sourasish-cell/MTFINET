"""
Data loading and augmentation pipeline with zero partition leakage.
"""

import os
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

METALS = ['Copper', 'Mercury', 'Palladium', 'Zinc', 'Cadmium', 'Silver', 'Chromium', 'Lead']
CONCENTRATIONS = [0, 10, 50, 100, 150, 200, 250]

METAL_TO_IDX = {m: i for i, m in enumerate(METALS)}
CONC_TO_IDX = {c: i for i, c in enumerate(CONCENTRATIONS)}
CONC_MAP_UM = np.array(CONCENTRATIONS)

def get_transforms():
    """
    Returns train and evaluation transforms.
    Standard ImageNet normalization is applied to match torchvision pretrained backbones.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform

class FluoroManifestDataset(Dataset):
    """
    Dataset loader driven by manifest CSV containing sample IDs and split assignments.
    """
    def __init__(self, manifest_df, root_dir=None, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Resolve path
        path_str = row['relative_image_path'] if 'relative_image_path' in row else row['image_path']
        img_path = Path(path_str)
        if self.root_dir and not img_path.is_absolute():
            img_path = self.root_dir / img_path
            
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}. Please configure dataset root directory.")
            
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        metal_label = METAL_TO_IDX[row['metal']]
        conc_label = CONC_TO_IDX[int(row['concentration_uM'])]
        
        return img, metal_label, conc_label

def build_dataloaders(manifest_csv_path, root_dir=None, batch_size=32, num_workers=2):
    """
    Constructs DataLoaders for train, val, and test splits.
    """
    df = pd.read_csv(manifest_csv_path)
    train_tf, eval_tf = get_transforms()
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    train_set = FluoroManifestDataset(train_df, root_dir=root_dir, transform=train_tf)
    val_set = FluoroManifestDataset(val_df, root_dir=root_dir, transform=eval_tf)
    test_set = FluoroManifestDataset(test_df, root_dir=root_dir, transform=eval_tf)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
