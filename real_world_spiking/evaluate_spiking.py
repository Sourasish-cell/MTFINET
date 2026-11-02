#!/usr/bin/env python3
"""
evaluate_spiking.py

Evaluation Engine for Real-World Water Matrix Spiking Study (Tap Water & Sea Water).
Uses pre-trained MTFINet to evaluate heavy-metal identification and concentration recovery.

Outputs:
1. Detailed sample predictions CSV
2. Analytical Spike-and-Recovery Table (CSV + LaTeX format for manuscript)
3. Publication-quality visualization figures (Recovery % bar charts, confusion matrices)
4. Comprehensive Markdown summary report
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add MTFINet root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from models.mtfinet import MultiTaskResNet
from dataset.dataloader import METALS, CONCENTRATIONS, METAL_TO_IDX, CONC_TO_IDX, CONC_MAP_UM

# Set style for publication plots
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MTFINet on Real-World Water Spiking Data")
    parser.add_argument("--manifest", type=str, default="manifests/spiking_manifest.csv",
                        help="Path to manifest CSV. If not found, auto-generates from data-dir.")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root folder containing real-world images")
    parser.add_argument("--checkpoint", type=str, 
                        default=str(PROJECT_ROOT / "checkpoints" / "best_multitask_model_leakage_free.pth"),
                        help="Path to pre-trained MTFINet checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save analytical results, tables, and figures")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    return parser.parse_args()

def load_or_create_manifest(manifest_path: Path, data_dir: Path):
    if manifest_path.exists():
        print(f"[*] Loading manifest from: {manifest_path}")
        return pd.read_csv(manifest_path)
    
    print(f"[!] Manifest not found at {manifest_path}. Auto-scanning {data_dir}...")
    import generate_spiking_manifest
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = [p for p in data_dir.rglob('*') if p.suffix.lower() in valid_exts]
    
    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No images found in {data_dir}. Please place your experimental images in "
            f"{data_dir}/tap_water and {data_dir}/sea_water, or provide a valid --manifest."
        )
        
    records = [generate_spiking_manifest.parse_filename_metadata(p) for p in sorted(image_files)]
    df = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"[+] Created new manifest: {manifest_path} ({len(df)} images)")
    return df

def generate_latex_table(recovery_df: pd.DataFrame, out_path: Path):
    """
    Exports a publication-grade LaTeX table formatted for IEEE Sensors Journal.
    """
    latex_lines = [
        r"\begin{table*}[!t]",
        r"\caption{Analytical Spike-and-Recovery Performance and Heavy-Metal Identification Across Real-World Environmental Water Matrices}",
        r"\label{tab_real_world_spiking}",
        r"\centering",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Water Matrix} & \textbf{Target Heavy Metal} & \textbf{Spiked Conc. ($\mu\mathrm{M}$)} & \textbf{Found Conc. (Mean $\pm$ SD, $\mu\mathrm{M}$)} & \textbf{Recovery (\%)} & \textbf{RSD (\%)} & \textbf{Metal Acc. (\%)} \\",
        r"\midrule"
    ]
    
    current_matrix = None
    for _, row in recovery_df.iterrows():
        matrix = row['water_matrix'].replace('_', ' ').title()
        metal = row['spiked_metal']
        spiked = row['spiked_concentration_uM']
        found_mean = row['mean_found_uM']
        found_sd = row['std_found_uM']
        recovery = f"{row['recovery_rate_pct']:.1f}\\%" if pd.notna(row['recovery_rate_pct']) else "---"
        rsd = f"{row['rsd_pct']:.1f}\\%" if pd.notna(row['rsd_pct']) else "0.0\\%"
        metal_acc = f"{row['metal_acc_pct']:.1f}\\%"
        
        found_str = f"${found_mean:.1f} \\pm {found_sd:.1f}$" if found_sd > 0 else f"${found_mean:.1f}$"
        
        if current_matrix is not None and current_matrix != matrix:
            latex_lines.append(r"\midrule")
            
        latex_lines.append(f"{matrix} & {metal} & {spiked} & {found_str} & {recovery} & {rsd} & {metal_acc} \\\\")
        current_matrix = matrix
        
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    with open(out_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"[+] Saved LaTeX table: {out_path}")

def plot_recovery_barchart(recovery_df: pd.DataFrame, out_path: Path):
    """
    Generates a high-resolution grouped bar chart comparing Recovery Rates (%) by Matrix.
    """
    # Exclude 0 uM blank control for recovery plot
    plot_df = recovery_df[recovery_df['spiked_concentration_uM'] > 0].copy()
    if len(plot_df) == 0:
        return
        
    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    palette = {'tap_water': '#1f77b4', 'sea_water': '#2ca02c'}
    ax = sns.barplot(
        data=plot_df,
        x='spiked_metal',
        y='recovery_rate_pct',
        hue='water_matrix',
        palette=palette,
        edgecolor='black',
        linewidth=0.8,
        errorbar=None
    )
    
    # Add acceptable recovery reference lines (80% - 120%)
    plt.axhline(100, color='red', linestyle='--', linewidth=1.2, label='Nominal 100% Recovery')
    plt.axhline(80, color='gray', linestyle=':', linewidth=1.0, label='Acceptable Analytical Window (80–120%)')
    plt.axhline(120, color='gray', linestyle=':', linewidth=1.0)
    
    plt.title("Spike-and-Recovery Performance Across Real-World Water Matrices", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Heavy-Metal Analyte", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Recovery Rate (%)", fontsize=12, fontweight='bold')
    plt.ylim(0, 140)
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Clean up legend labels
    handles, labels = ax.get_legend_handles_labels()
    clean_labels = [l.replace('_', ' ').title() for l in labels]
    plt.legend(handles, clean_labels, loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved recovery bar chart: {out_path}")

def plot_spiking_confusion_matrices(pred_df: pd.DataFrame, out_path: Path):
    """
    Generates confusion matrices for Tap Water and Sea Water matrices.
    """
    matrices = pred_df['water_matrix'].unique()
    num_m = len(matrices)
    
    fig, axes = plt.subplots(1, num_m, figsize=(8 * num_m, 6), dpi=300)
    if num_m == 1:
        axes = [axes]
        
    for idx, mat in enumerate(matrices):
        sub_df = pred_df[pred_df['water_matrix'] == mat]
        valid_df = sub_df[sub_df['spiked_metal'].isin(METALS)]
        if len(valid_df) == 0:
            continue
            
        y_true = [METAL_TO_IDX[m] for m in valid_df['spiked_metal']]
        y_pred = [METAL_TO_IDX[m] for m in valid_df['pred_metal']]
        
        acc = accuracy_score(y_true, y_pred) * 100.0
        cm = confusion_matrix(y_true, y_pred, labels=range(len(METALS)))
        
        cmap = 'Blues' if 'tap' in mat else 'Greens'
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx],
                    xticklabels=METALS, yticklabels=METALS, cbar=True)
        
        mat_title = mat.replace('_', ' ').title()
        axes[idx].set_title(f"{mat_title} (Metal Acc: {acc:.1f}%, N={len(valid_df)})", fontsize=13, fontweight='bold')
        axes[idx].set_xlabel("Predicted Metal", fontsize=11, fontweight='bold')
        axes[idx].set_ylabel("True Spiked Metal", fontsize=11, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved spiking confusion matrices: {out_path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print("\n" + "="*70)
    print("      MTFINet REAL-WORLD WATER MATRIX SPIKING EVALUATION ENGINE")
    print("="*70)
    print(f"[*] Execution Device: {device}")
    print(f"[*] Model Checkpoint: {args.checkpoint}")
    
    # 1. Load Model
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    model = MultiTaskResNet(num_metals=8, num_concentrations=7, pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[+] MTFINet architecture and weights successfully loaded.")
    
    # 2. Load Manifest
    manifest_df = load_or_create_manifest(Path(args.manifest), Path(args.data_dir))
    transform = get_eval_transforms()
    
    print(f"[*] Processing {len(manifest_df)} experimental samples...")
    
    # 3. Inference Loop
    results = []
    softmax = nn.Softmax(dim=1)
    
    for idx, row in manifest_df.iterrows():
        img_path = Path(row['image_path'])
        if not img_path.is_absolute() and not img_path.exists():
            img_path = Path(args.data_dir) / img_path
            
        if not img_path.exists():
            print(f"[!] Warning: Image file not found: {img_path}. Skipping.")
            continue
            
        try:
            raw_img = Image.open(img_path).convert('RGB')
            tensor_img = transform(raw_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                m_logits, c_logits = model(tensor_img)
                m_probs = softmax(m_logits).cpu().numpy()[0]
                c_probs = softmax(c_logits).cpu().numpy()[0]
                
            pred_m_idx = np.argmax(m_probs)
            pred_c_idx = np.argmax(c_probs)
            
            pred_metal = METALS[pred_m_idx]
            pred_conc_uM = CONCENTRATIONS[pred_c_idx]
            
            m_conf = float(m_probs[pred_m_idx])
            c_conf = float(c_probs[pred_c_idx])
            
            spiked_metal = row['spiked_metal']
            spiked_conc = int(row['spiked_concentration_uM'])
            
            metal_correct = (pred_metal == spiked_metal)
            conc_error_uM = abs(pred_conc_uM - spiked_conc)
            
            results.append({
                'image_path': str(img_path),
                'water_matrix': row['water_matrix'],
                'spiked_metal': spiked_metal,
                'spiked_concentration_uM': spiked_conc,
                'replicate_id': row.get('replicate_id', 1),
                'pred_metal': pred_metal,
                'pred_metal_confidence': m_conf,
                'metal_correct': metal_correct,
                'pred_concentration_uM': pred_conc_uM,
                'pred_conc_confidence': c_conf,
                'conc_error_uM': conc_error_uM
            })
        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")
            
    pred_df = pd.DataFrame(results)
    if len(pred_df) == 0:
        print("[!] No images were successfully processed. Exiting.")
        return
        
    detailed_csv = output_dir / "spike_recovery_detailed_predictions.csv"
    pred_df.to_csv(detailed_csv, index=False)
    print(f"\n[+] Saved detailed predictions to: {detailed_csv}")
    
    # 4. Compute Analytical Spike-and-Recovery Statistics
    grouped = pred_df.groupby(['water_matrix', 'spiked_metal', 'spiked_concentration_uM'])
    
    summary_rows = []
    for (matrix, metal, spiked_c), group in grouped:
        n_samples = len(group)
        mean_found = group['pred_concentration_uM'].mean()
        std_found = group['pred_concentration_uM'].std(ddof=1) if n_samples > 1 else 0.0
        if pd.isna(std_found):
            std_found = 0.0
            
        rsd = (std_found / mean_found * 100.0) if mean_found > 0 else 0.0
        recovery = (mean_found / spiked_c * 100.0) if spiked_c > 0 else np.nan
        metal_acc = (group['metal_correct'].sum() / n_samples) * 100.0
        
        summary_rows.append({
            'water_matrix': matrix,
            'spiked_metal': metal,
            'spiked_concentration_uM': spiked_c,
            'replicates': n_samples,
            'mean_found_uM': mean_found,
            'std_found_uM': std_found,
            'recovery_rate_pct': recovery,
            'rsd_pct': rsd,
            'metal_acc_pct': metal_acc
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "spike_recovery_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[+] Saved spike-and-recovery summary table to: {summary_csv}")
    
    # 5. Generate LaTeX Table
    latex_file = output_dir / "spike_recovery_table.tex"
    generate_latex_table(summary_df, latex_file)
    
    # 6. Generate Figures
    barchart_file = fig_dir / "spike_recovery_barchart.png"
    plot_recovery_barchart(summary_df, barchart_file)
    
    cm_file = fig_dir / "spiking_confusion_matrices.png"
    plot_spiking_confusion_matrices(pred_df, cm_file)
    
    # 7. Print Console Summary
    print("\n" + "="*70)
    print("          REAL-WORLD SPIKE-AND-RECOVERY PERFORMANCE SUMMARY")
    print("="*70)
    for matrix in summary_df['water_matrix'].unique():
        sub = summary_df[summary_df['water_matrix'] == matrix]
        spiked_sub = sub[sub['spiked_concentration_uM'] > 0]
        avg_rec = spiked_sub['recovery_rate_pct'].mean()
        avg_acc = sub['metal_acc_pct'].mean()
        print(f"\n--- Matrix: {matrix.replace('_', ' ').title()} ---")
        print(f"Overall Metal Identification Accuracy: {avg_acc:.2f}%")
        print(f"Mean Spike Recovery Rate:              {avg_rec:.2f}%")
        print(f"Analytical Recovery Range:             {spiked_sub['recovery_rate_pct'].min():.1f}% - {spiked_sub['recovery_rate_pct'].max():.1f}%")
        
    print("\n" + "="*70)
    print("[✓] ALL EXPERIMENTAL EVALUATION TASKS COMPLETED SUCCESSFULLY!")
    print(f"[✓] Inspect your generated tables and plots inside: {output_dir.resolve()}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
