#!/usr/bin/env python3
"""
crop_evaluation_images.py

Batch crops raw fluorescence images from /Users/royalsn/PHD/Work/Resources/Dataset_Evaluation
into standardized (224, 224, 3) cropped patches matching the training distribution
and sample images provided by the user.

Outputs:
1. Cropped images in /Users/royalsn/PHD/Work/Resources/Dataset_Evaluation_Cropped
2. Cropped images mirrored to /Users/royalsn/PHD/Work/MTFINet/real_world_spiking/data
3. Generated manifest: /Users/royalsn/PHD/Work/MTFINet/real_world_spiking/manifests/spiking_manifest.csv
4. Quality control verification report
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

RAW_DIR = Path("/Users/royalsn/PHD/Work/Resources/Dataset_Evaluation")
CROPPED_DIR = Path("/Users/royalsn/PHD/Work/Resources/Dataset_Evaluation_Cropped")
SPIKING_DATA_DIR = Path("/Users/royalsn/PHD/Work/MTFINet/real_world_spiking/data")
MANIFEST_PATH = Path("/Users/royalsn/PHD/Work/MTFINet/real_world_spiking/manifests/spiking_manifest.csv")

def find_fluorescent_circle(image):
    """
    Robust multi-stage fluorescent spot detector optimized for 4096x3072 UV images.
    Filters out background dark-box reflections, metallic clip reflections, and screws.
    """
    img_h, img_w = image.shape[:2]
    
    # --- Stage 1: (B - R) Color Difference with Vertical Masking ---
    # Carbon dots fluoresce in cyan-blue (B >> R), whereas metallic reflections,
    # screws, and the dark chamber have B ~= R.
    b, g, r = cv2.split(image)
    diff = cv2.subtract(b, r)
    diff[:600, :] = 0
    diff[3600:, :] = 0
    
    for p_val in [96.5, 96.0, 97.0, 95.5, 95.0, 97.5]:
        positive_diff = diff[diff > 0]
        if len(positive_diff) == 0:
            break
        th = np.percentile(positive_diff, p_val)
        mask = (diff >= th).astype(np.uint8) * 255
        kernel_close = np.ones((25, 25), np.uint8)
        kernel_open = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 100000 or area > 1400000:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / float(ch)
            if 0.65 <= aspect <= 1.45 and 500 <= max(cw, ch) <= 1200:
                perim = cv2.arcLength(c, True)
                circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
                cands.append((area, circ, (x, y, cw, ch)))
        if cands:
            cands.sort(key=lambda t: t[0], reverse=True)
            return cands[0][2]

    # --- Stage 2: Grayscale Adaptive Percentile Thresholding ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[:600, :] = 0
    gray[3500:, :] = 0
    for p_val in [96.5, 96.0, 97.0, 95.5, 95.0, 97.5]:
        positive_gray = gray[gray > 0]
        if len(positive_gray) == 0:
            break
        th = np.percentile(positive_gray, p_val)
        mask = (gray >= th).astype(np.uint8) * 255
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 100000 or area > 1400000:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / float(ch)
            if 0.65 <= aspect <= 1.45 and 500 <= max(cw, ch) <= 1200:
                cands.append((area, (x, y, cw, ch)))
        if cands:
            cands.sort(key=lambda t: t[0], reverse=True)
            return cands[0][1]

    # --- Stage 3: HSV Cyan-Blue Color Masking ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, bright_mask = cv2.threshold(hsv[:, :, 2], 70, 255, cv2.THRESH_BINARY)
    hue_mask = cv2.inRange(hsv[:, :, 0], 75, 145)
    mask = cv2.bitwise_and(bright_mask, hue_mask)
    mask[:600, :] = 0
    mask[3500:, :] = 0
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100000 or area > 1400000:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch)
        if 0.65 <= aspect <= 1.45 and 500 <= max(cw, ch) <= 1200:
            cands.append((area, (x, y, cw, ch)))
    if cands:
        cands.sort(key=lambda t: t[0], reverse=True)
        return cands[0][1]

    # --- Stage 4: Hough Circles Fallback Strictly Constrained ---
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=400, param1=50, param2=30,
        minRadius=300, maxRadius=550
    )
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            if 800 <= cy <= 3400 and 400 <= cx <= 2600:
                return cx - r, cy - r, 2 * r, 2 * r

    return None

def crop_square(image, x, y, cw, ch, padding_frac=0.12):
    img_h, img_w = image.shape[:2]
    side = max(cw, ch)
    pad = int(side * padding_frac)
    side += 2 * pad
    
    cx, cy = x + cw // 2, y + ch // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side
    
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h
        
    x1, y1 = max(0, x1), max(0, y1)
    s = min(x2 - x1, y2 - y1)
    return image[y1:y1 + s, x1:x1 + s]

def parse_metadata(rel_path: Path):
    parts = rel_path.parts
    matrix = 'tap_water' if 'tap' in parts[0].lower() else 'sea_water'
    metal = parts[1]
    
    conc_folder = parts[2]
    conc_match = re.search(r'(\d+)\s*(?:mu[mn]|um)', conc_folder, re.IGNORECASE)
    if not conc_match:
        raise ValueError(f"Could not parse concentration from {conc_folder}")
    conc = int(conc_match.group(1))
    
    # Replicate number from filename or order
    filename = rel_path.name
    rep_match = re.search(r'IMG\d+(_\d+)?', filename)
    rep_id = filename
    
    return matrix, metal, conc, rep_id

def process_single_image(img_path: Path):
    rel_path = img_path.relative_to(RAW_DIR)
    matrix, metal, conc, rep_id = parse_metadata(rel_path)
    
    im = cv2.imread(str(img_path))
    if im is None:
        return {'status': 'failed', 'path': str(img_path), 'reason': 'cannot read'}
        
    res = find_fluorescent_circle(im)
    if res is None:
        return {'status': 'failed', 'path': str(img_path), 'reason': 'no circle detected'}
        
    x, y, cw, ch = res
    crop = crop_square(im, x, y, cw, ch, padding_frac=0.12)
    if crop.size == 0:
        return {'status': 'failed', 'path': str(img_path), 'reason': 'empty crop'}
        
    final_crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Save 1: Resources/Dataset_Evaluation_Cropped
    out_path1 = CROPPED_DIR / rel_path
    out_path1.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path1), final_crop)
    
    # Save 2: MTFINet/real_world_spiking/data
    out_path2 = SPIKING_DATA_DIR / rel_path
    out_path2.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path2), final_crop)
    
    # Measure circle diameter in crop for Quality Control
    gray_crop = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
    _, thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cw_crop, ch_crop = 0, 0
    if cnts:
        _, _, cw_crop, ch_crop = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        
    return {
        'status': 'success',
        'image_path': str(out_path2),
        'cropped_resource_path': str(out_path1),
        'water_matrix': matrix,
        'spiked_metal': metal,
        'spiked_concentration_uM': conc,
        'replicate_id': rep_id,
        'circle_w': cw_crop,
        'circle_h': ch_crop
    }

def main():
    print("="*70)
    print("      MTFINet BATCH CROPPING PIPELINE FOR EVALUATION DATASET")
    print("="*70)
    print(f"[*] Raw Source Directory:      {RAW_DIR}")
    print(f"[*] Cropped Target Directory:  {CROPPED_DIR}")
    print(f"[*] Spiking Evaluation Data:   {SPIKING_DATA_DIR}")
    
    raw_files = sorted([p for p in RAW_DIR.rglob("*.jpg") if not p.name.startswith('.')])
    print(f"[*] Found {len(raw_files)} raw images to process.")
    
    num_workers = min(8, cpu_count())
    print(f"[*] Launching multiprocessing pool with {num_workers} parallel workers...")
    
    with Pool(num_workers) as pool:
        results = pool.map(process_single_image, raw_files)
        
    successes = [r for r in results if r['status'] == 'success']
    failures = [r for r in results if r['status'] == 'failed']
    
    print("\n" + "="*70)
    print("                 CROPPING EXECUTION SUMMARY")
    print("="*70)
    print(f"Total Processed: {len(results)}")
    print(f"Successful:      {len(successes)} ({len(successes)/len(results)*100:.1f}%)")
    print(f"Failed:          {len(failures)}")
    
    if failures:
        print("[!] Sample failures:")
        for f in failures[:5]:
            print("   ", f)
        raise RuntimeError("Cropping failed on some images. Please inspect.")
        
    # Quality Control Analysis
    df = pd.DataFrame(successes)
    print("\n[+] Quality Control Metrics (Circle dimensions in 224x224 crop):")
    print(f"    Mean Circle Width:  {df['circle_w'].mean():.1f} px (Ratio: {df['circle_w'].mean()/224:.2f})")
    print(f"    Mean Circle Height: {df['circle_h'].mean():.1f} px (Ratio: {df['circle_h'].mean()/224:.2f})")
    print(f"    Min Width: {df['circle_w'].min()} px, Max Width: {df['circle_w'].max()} px")
    print("    (Note: Benchmark sample images had width=178 and 175 px, ratio=0.78-0.80)")
    
    # Save Manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_df = df[['image_path', 'water_matrix', 'spiked_metal', 'spiked_concentration_uM', 'replicate_id']]
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    print(f"\n[+] Generated Spiking Evaluation Manifest: {MANIFEST_PATH}")
    print(f"    Total rows in manifest: {len(manifest_df)}")
    print("\nClass breakdown in manifest:")
    print(manifest_df.groupby(['water_matrix', 'spiked_metal']).size().unstack(fill_value=0))
    
    # Generate Visual Quality Control Grid comparing with user's sample images
    try:
        s1_path = Path("/Users/royalsn/.gemini/antigravity/brain/75579103-d38f-431d-8e7c-c36e9409a06c/.user_uploaded/media_1793366953471_c5439bda.jpg")
        s2_path = Path("/Users/royalsn/.gemini/antigravity/brain/75579103-d38f-431d-8e7c-c36e9409a06c/.user_uploaded/media_1793366970866_c27ac99b.jpg")
        
        grid_imgs = []
        titles = []
        if s1_path.exists():
            grid_imgs.append(cv2.resize(cv2.imread(str(s1_path)), (224, 224)))
            titles.append("Benchmark Sample 1")
        if s2_path.exists():
            grid_imgs.append(cv2.resize(cv2.imread(str(s2_path)), (224, 224)))
            titles.append("Benchmark Sample 2")
            
        sample_records = df.groupby(['water_matrix', 'spiked_metal']).first().reset_index()
        for _, r in sample_records.head(14).iterrows():
            im = cv2.imread(r['image_path'])
            if im is not None:
                grid_imgs.append(im)
                titles.append(f"{r['water_matrix'][:3]}:{r['spiked_metal'][:2]}_{r['spiked_concentration_uM']}uM")
                
        # Build 4x4 canvas
        cols, rows = 4, 4
        cell_size = 224
        header_h = 30
        canvas = np.zeros((rows * (cell_size + header_h), cols * cell_size, 3), dtype=np.uint8)
        
        for idx in range(min(len(grid_imgs), rows * cols)):
            r_idx = idx // cols
            c_idx = idx % cols
            y_offset = r_idx * (cell_size + header_h)
            x_offset = c_idx * cell_size
            
            # Header bar
            cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + cell_size, y_offset + header_h), (40, 40, 40), -1)
            cv2.putText(canvas, titles[idx], (x_offset + 5, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Image cell
            canvas[y_offset + header_h:y_offset + header_h + cell_size, x_offset:x_offset + cell_size] = grid_imgs[idx]
            
        grid_path = Path("/Users/royalsn/.gemini/antigravity/brain/75579103-d38f-431d-8e7c-c36e9409a06c/cropping_verification_grid_v2.jpg")
        cv2.imwrite(str(grid_path), canvas)
        print(f"[+] Saved visual verification grid: {grid_path}")
    except Exception as e:
        print(f"[!] Warning generating verification grid: {e}")
        
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
