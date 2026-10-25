#!/usr/bin/env python3
"""
Evaluation script for MTFINet on held-out test split.
Outputs Accuracy, TEC, QER (Mapped-Class MAE in µM), QWK, and per-class reports.
"""

import argparse
import sys
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.mtfinet import MultiTaskResNet
from dataset.dataloader import build_dataloaders, METALS, CONCENTRATIONS, CONC_MAP_UM

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MTFINet on Held-Out Test Set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_multitask_model_leakage_free.pth", help="Model weights path")
    parser.add_argument("--manifest", type=str, default="dataset/dataset_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--root-dir", type=str, default=None, help="Root directory containing images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save metric outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"[*] Evaluating MTFINet using device: {device}")
    
    _, _, test_loader = build_dataloaders(args.manifest, root_dir=args.root_dir, batch_size=args.batch_size, num_workers=0)
    
    model = MultiTaskResNet(num_metals=8, num_concentrations=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    t_m_preds, t_m_trues = [], []
    t_c_preds, t_c_trues = [], []
    
    with torch.no_grad():
        for imgs, m_labels, c_labels in test_loader:
            imgs = imgs.to(device)
            p_m, p_c = model(imgs)
            t_m_preds.extend(torch.argmax(p_m, dim=1).cpu().numpy())
            t_c_preds.extend(torch.argmax(p_c, dim=1).cpu().numpy())
            t_m_trues.extend(m_labels.numpy())
            t_c_trues.extend(c_labels.numpy())
            
    m_acc = accuracy_score(t_m_trues, t_m_preds)
    c_acc = accuracy_score(t_c_trues, t_c_preds)
    qer_mae = mean_absolute_error(CONC_MAP_UM[t_c_trues], CONC_MAP_UM[t_c_preds])
    qwk = cohen_kappa_score(t_c_trues, t_c_preds, weights='quadratic')
    
    print("\n" + "="*60)
    print("           MTFINet HELD-OUT TEST EVALUATION REPORT")
    print("="*60)
    print(f"Total Test Images:               {len(test_loader.dataset)}")
    print(f"Metal Identification Accuracy:   {m_acc*100:.2f}% (TEC: {(1-m_acc)*100:.2f}%)")
    print(f"Concentration-Class Accuracy:    {c_acc*100:.2f}% (TEC: {(1-c_acc)*100:.2f}%)")
    print(f"Mapped-Class MAE (QER):          {qer_mae:.2f} µM")
    print(f"Quadratic Weighted Kappa (QWK):  {qwk:.4f}")
    print("="*60)
    
    print("\n[+] Per-Metal Classification Metrics:")
    print(classification_report(t_m_trues, t_m_preds, target_names=METALS, digits=4))
    
    print("\n[+] Per-Concentration Classification Metrics:")
    print(classification_report(t_c_trues, t_c_preds, target_names=[f"{c} µM" for c in CONCENTRATIONS], digits=4))
    
    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    cm_m = confusion_matrix(t_m_trues, t_m_preds)
    sns.heatmap(cm_m, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=METALS, yticklabels=METALS)
    axes[0].set_title(f"Metal Identification ({m_acc*100:.2f}%)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Metal")
    axes[0].set_ylabel("True Metal")
    axes[0].tick_params(axis='x', rotation=45)
    
    cm_c = confusion_matrix(t_c_trues, t_c_preds)
    sns.heatmap(cm_c, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=[f"{c} µM" for c in CONCENTRATIONS], yticklabels=[f"{c} µM" for c in CONCENTRATIONS])
    axes[1].set_title(f"Concentration-Class Prediction ({c_acc*100:.2f}%)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Concentration")
    axes[1].set_ylabel("True Concentration")
    
    plt.tight_layout()
    cm_fig_path = out_dir / "figures" / "test_dual_confusion_matrix.png"
    cm_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_fig_path, dpi=300)
    print(f"[+] Saved confusion matrix visualization to: {cm_fig_path}")

if __name__ == '__main__':
    main()
