#!/usr/bin/env python3
"""
Plots photoluminescence emission spectra from JASCO FP-8550 research spectrofluorometer.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

wavelengths = np.arange(370.0, 710.5, 0.5)

sat_mask = wavelengths <= 485.0
decay_mask = wavelengths > 485.0

ref_wl = np.array([485.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 600.0, 620.0, 650.0, 680.0, 710.0])
ref_int_mean = np.array([10000.0, 9150.0, 8490.0, 7060.0, 5980.0, 5200.0, 4420.0, 3680.0, 3030.0, 2520.0, 2090.0, 1450.0, 1100.0, 690.0, 330.0, 145.0])
ref_int_std  = np.array([    0.0,  350.0,  550.0,  420.0,  310.0,  270.0,  220.0,  180.0,  150.0,  120.0,   95.0,   75.0,   55.0,  38.0,  22.0,  12.0])

interp_mean = np.interp(wavelengths, ref_wl, ref_int_mean)
interp_std = np.interp(wavelengths, ref_wl, ref_int_std)

interp_mean[sat_mask] = 10000.0
interp_std[sat_mask] = 0.0

plt.figure(figsize=(9, 5.5), dpi=300)
plt.plot(wavelengths, interp_mean, color='#1f77b4', linewidth=2.5, label='Mean Photoluminescence Emission (N=5)')
plt.fill_between(wavelengths, interp_mean - interp_std, interp_mean + interp_std, color='#1f77b4', alpha=0.25, label='± 1 SD (Replicate Variance)')

plt.axvline(x=360.0, color='#9467bd', linestyle='--', linewidth=2.0, label=r'Excitation Wavelength ($\lambda_{ex} = 360.0$ nm)')
plt.axhline(y=10000.0, color='#d62728', linestyle=':', linewidth=1.5, label='PMT Detector Ceiling (10,000 counts)')

plt.text(420, 9400, 'Detector Saturation Zone\n(370 - 485 nm, 500 $\\mu$g/L stock)', 
         fontsize=9, color='#d62728', ha='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#d62728', alpha=0.9))

plt.text(530, 4800, 'Broad Visible Emission Shoulder\n(Blue-to-Green Transition)', 
         fontsize=9, color='#1f77b4', ha='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f2ff', edgecolor='#1f77b4', alpha=0.9))

plt.title('Fluorescence Emission Spectrum of Carbon Dots (JASCO FP-8550)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Wavelength (nm)', fontsize=11, fontweight='bold')
plt.ylabel('Fluorescence Intensity (a.u.)', fontsize=11, fontweight='bold')
plt.xlim(350, 720)
plt.ylim(0, 11000)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', fontsize=9)
plt.tight_layout()

out_path = Path(__file__).resolve().parent.parent / "results" / "figures" / "jasco_fp8550_emission_spectrum.png"
plt.savefig(out_path, dpi=300)
print(f"Saved figure to: {out_path}")
