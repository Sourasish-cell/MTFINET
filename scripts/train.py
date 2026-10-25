#!/usr/bin/env python3
"""
Training script for Multi-Task Fluorescence Intelligence Network (MTFINet)
Includes validation tracking, learning rate scheduling, and leakage-safe checkpoint selection.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Ensure root directory is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.mtfinet import MultiTaskResNet
from dataset.dataloader import build_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train MTFINet on Carbon-Dot Fluorescence Dataset")
    parser.add_argument("--manifest", type=str, default="dataset/dataset_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--root-dir", type=str, default=None, help="Root directory containing images")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--save-path", type=str, default="checkpoints/best_multitask_model.pth", help="Checkpoint save path")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"[*] Training MTFINet using device: {device}")
    
    train_loader, val_loader, test_loader = build_dataloaders(
        args.manifest, root_dir=args.root_dir, batch_size=args.batch_size, num_workers=0
    )
    
    model = MultiTaskResNet(num_metals=8, num_concentrations=7, pretrained=True).to(device)
    criterion_m = nn.CrossEntropyLoss()
    criterion_c = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Starting optimization for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, m_labels, c_labels in train_loader:
            imgs, m_labels, c_labels = imgs.to(device), m_labels.to(device), c_labels.to(device)
            optimizer.zero_grad()
            p_m, p_c = model(imgs)
            loss = criterion_m(p_m, m_labels) + criterion_c(p_c, c_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        v_m_preds, v_m_trues = [], []
        v_c_preds, v_c_trues = [], []
        with torch.no_grad():
            for imgs, m_labels, c_labels in val_loader:
                imgs, m_labels, c_labels = imgs.to(device), m_labels.to(device), c_labels.to(device)
                p_m, p_c = model(imgs)
                l = criterion_m(p_m, m_labels) + criterion_c(p_c, c_labels)
                val_loss += l.item() * imgs.size(0)
                v_m_preds.extend(torch.argmax(p_m, dim=1).cpu().numpy())
                v_c_preds.extend(torch.argmax(p_c, dim=1).cpu().numpy())
                v_m_trues.extend(m_labels.cpu().numpy())
                v_c_trues.extend(c_labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        v_acc_m = accuracy_score(v_m_trues, v_m_preds)
        v_acc_c = accuracy_score(v_c_trues, v_c_preds)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Metal Acc: {v_acc_m*100:.2f}% | Val Conc Acc: {v_acc_c*100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch+1}!")
            
    print(f"\n[*] Training completed. Best model saved to: {save_path}")

if __name__ == '__main__':
    main()
