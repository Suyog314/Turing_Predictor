
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pattern_generator import run_simulation

# === CONFIGURATION ===
SAVE_DIR = 'patterns'
CSV_FILE = 'metadata.csv'
IMAGE_SIZE = 128
STEPS = 10000
DT = 1.0
DX = 1.0
TARGET_IMAGES = 10000
VISIBILITY_THRESHOLD = 0.05

# === PARAMETER RANGES ===
Du_vals = np.linspace(0.10, 0.25, 15)
Dv_vals = np.linspace(0.05, 0.20, 15)
F_vals  = np.linspace(0.02, 0.09, 15)
k_vals  = np.linspace(0.03, 0.07, 15)

# === FOLDER SETUP ===
os.makedirs(SAVE_DIR, exist_ok=True)
existing_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.png')])
start_index = len(existing_files)

# === OPEN METADATA FILE ===
with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    if start_index == 0 and os.stat(CSV_FILE).st_size == 0:
        writer.writerow([
            'filename',
            'D_u (mm^2/s)',
            'D_v (mm^2/s)',
            'F (1/s)',
            'k (1/s)',
            'delta_t (s)',
            'delta_x (mm)',
            'iterations',
            'image_saved'
        ])
        file.flush()

    i = start_index
    skipped = 0
    saved = start_index

    param_combinations = [
        (Du, Dv, F, k)
        for Du in Du_vals
        for Dv in Dv_vals if Du > Dv
        for F in F_vals
        for k in k_vals if k > F
    ]

    pbar = tqdm(param_combinations[i:], desc="🌀 Generating Patterns")

    for Du, Dv, F, k in pbar:
        if saved >= TARGET_IMAGES:
            break

        # 🚀 Call with randomized 2–5 origins
        V = run_simulation(Du, Dv, F, k, dt=DT, steps=STEPS, size=IMAGE_SIZE)

        filename = f"{i:06}.png"
        image_path = os.path.join(SAVE_DIR, filename)
        image_saved = False

        if V.max() >= VISIBILITY_THRESHOLD:
            plt.imsave(image_path, V / V.max(), cmap='inferno')
            image_saved = True
            saved += 1
        else:
            skipped += 1

        writer.writerow([
            filename,
            round(Du, 5),
            round(Dv, 5),
            round(F, 5),
            round(k, 5),
            DT,
            DX,
            STEPS,
            int(image_saved)
        ])
        file.flush()
        pbar.set_postfix(saved=saved, skipped=skipped)

        i += 1

print("\n🎯 Generation complete.")
print(f"✅ Total saved: {saved}")
print(f"❌ Skipped (low contrast): {skipped}")
