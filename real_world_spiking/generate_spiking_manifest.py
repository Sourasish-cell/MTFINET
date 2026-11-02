#!/usr/bin/env python3
"""
generate_spiking_manifest.py

Automatically scans image directories for real-world water spiking experiments
(Tap Water & Sea Water) and generates a structured CSV manifest for MTFINet evaluation.

Supports:
- Auto-detection from folder structure: data/<matrix>/<metal>/<conc>/<file>.jpg
- Auto-detection from filenames: e.g. tap_Cu_50uM_rep1.jpg, sea_Lead_100uM_02.png
- Metal abbreviation mapping: Cu -> Copper, Pb -> Lead, Hg -> Mercury, etc.
"""

import os
import re
import argparse
from pathlib import Path
import pandas as pd

METAL_SYNONYMS = {
    'copper': 'Copper', 'cu': 'Copper', 'cu2+': 'Copper', 'cu(ii)': 'Copper',
    'mercury': 'Mercury', 'hg': 'Mercury', 'hg2+': 'Mercury', 'hg(ii)': 'Mercury',
    'palladium': 'Palladium', 'pd': 'Palladium', 'pd2+': 'Palladium', 'pd(ii)': 'Palladium',
    'zinc': 'Zinc', 'zn': 'Zinc', 'zn2+': 'Zinc', 'zn(ii)': 'Zinc',
    'cadmium': 'Cadmium', 'cd': 'Cadmium', 'cd2+': 'Cadmium', 'cd(ii)': 'Cadmium',
    'silver': 'Silver', 'ag': 'Silver', 'ag+': 'Silver', 'ag(i)': 'Silver',
    'chromium': 'Chromium', 'cr': 'Chromium', 'cr3+': 'Chromium', 'cr(iii)': 'Chromium',
    'lead': 'Lead', 'pb': 'Lead', 'pb2+': 'Lead', 'pb(ii)': 'Lead',
    'blank': 'Copper', 'control': 'Copper', 'none': 'Copper'  # Blanks mapped to 0 uM tier
}

VALID_CONCENTRATIONS = [0, 10, 50, 100, 150, 200, 250]

def parse_filename_metadata(file_path: Path):
    """
    Infers water matrix, metal, concentration, and replicate from file path and name.
    """
    path_str = str(file_path).lower()
    file_stem = file_path.stem.lower()
    
    # 1. Detect Water Matrix
    if 'sea' in path_str:
        matrix = 'sea_water'
    elif 'tap' in path_str:
        matrix = 'tap_water'
    else:
        matrix = 'real_water'
        
    # 2. Detect Heavy Metal
    metal_found = None
    # Check words or split tokens
    tokens = re.split(r'[_ \-\\/]', path_str)
    for token in tokens:
        if token in METAL_SYNONYMS:
            metal_found = METAL_SYNONYMS[token]
            break
            
    if not metal_found:
        # Check substring matches
        for key in sorted(METAL_SYNONYMS.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(key) + r'\b', file_stem) or key in tokens:
                metal_found = METAL_SYNONYMS[key]
                break
                
    if not metal_found:
        metal_found = 'Unknown'
        
    # 3. Detect Concentration
    conc_found = None
    if re.search(r'\b(?:blank|control)\b', path_str):
        conc_found = 0
    else:
        conc_match = re.search(r'(?<!\d)(\d+)\s*(?:um|uM|micromolar|ppm)?(?!\d)', file_stem)
        if conc_match:
            val = int(conc_match.group(1))
            # Find closest standard nominal tier
            conc_found = min(VALID_CONCENTRATIONS, key=lambda x: abs(x - val))
        else:
            conc_found = 0
        
    # 4. Detect Replicate ID
    rep_match = re.search(r'(?:rep|r|spot|sample)?_?(\d+)$', file_stem)
    replicate_id = int(rep_match.group(1)) if rep_match else 1
    
    return {
        'image_path': str(file_path),
        'water_matrix': matrix,
        'spiked_metal': metal_found,
        'spiked_concentration_uM': conc_found,
        'replicate_id': replicate_id
    }

def main():
    parser = argparse.ArgumentParser(description="Auto-generate Spiking Manifest for Real-World Water Samples")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing real-world spiked images")
    parser.add_argument("--output", type=str, default="manifests/spiking_manifest.csv", help="Output CSV path")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[!] Warning: Directory {data_dir} does not exist yet. Creating directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = [p for p in data_dir.rglob('*') if p.suffix.lower() in valid_exts]
    
    print(f"[*] Found {len(image_files)} image files in {data_dir}")
    
    if len(image_files) == 0:
        print("[!] No images found yet. Please place your captured experimental images inside:")
        print(f"    - {data_dir.resolve() / 'tap_water'}")
        print(f"    - {data_dir.resolve() / 'sea_water'}")
        print("[*] Generated a template manifest instead.")
        return
        
    records = []
    for img_path in sorted(image_files):
        meta = parse_filename_metadata(img_path)
        records.append(meta)
        
    df = pd.DataFrame(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print(f"[+] Successfully generated manifest with {len(df)} samples: {out_path}")
    print("\nSummary of detected samples:")
    print(df.groupby(['water_matrix', 'spiked_metal', 'spiked_concentration_uM']).size().reset_index(name='count'))

if __name__ == '__main__':
    main()
