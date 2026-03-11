"""
HSV Range Analysis Tool for Cone Detection

Samples HSV pixel values from ground-truth cone regions across all
test images and reports percentile statistics. Use the output to set data-driven
thresholds in color_segmentation.py instead of guessing.

Usage:
    python3 analyze_hsv.py

Output:
    - Per-channel (H, S, V) percentile table
    - Suggested lower/upper bounds at the 2nd and 98th percentile
    - Histogram saved to hsv_histogram.png for your report
"""

import cv2
import numpy as np
import csv
import ast

CSV_PATH = "./test_images_cone/test_images_cone.csv"

h_vals, s_vals, v_vals = [], [], []

with open(CSV_PATH) as f:
    for row in csv.reader(f):
        img_path = row[0]
        (x1, y1), (x2, y2) = ast.literal_eval(row[1])

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {img_path}")
            continue

        # crop to ground-truth bounding box
        cone_crop = img[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(cone_crop, cv2.COLOR_BGR2HSV)

        h_vals.extend(hsv_crop[:, :, 0].flatten().tolist())
        s_vals.extend(hsv_crop[:, :, 1].flatten().tolist())
        v_vals.extend(hsv_crop[:, :, 2].flatten().tolist())

h = np.array(h_vals)
s = np.array(s_vals)
v = np.array(v_vals)

print(f"\nTotal cone pixels sampled: {len(h):,}")
print(f"\n{'Channel':<10} {'Min':>6} {'P2':>6} {'P5':>6} {'P25':>6} {'P50':>6} {'P75':>6} {'P95':>6} {'P98':>6} {'Max':>6}")
print("-" * 76)
for name, arr in [("Hue", h), ("Saturation", s), ("Value", v)]:
    pcts = np.percentile(arr, [0, 2, 5, 25, 50, 75, 95, 98, 100])
    print(f"{name:<10} {pcts[0]:>6.0f} {pcts[1]:>6.0f} {pcts[2]:>6.0f} {pcts[3]:>6.0f} "
          f"{pcts[4]:>6.0f} {pcts[5]:>6.0f} {pcts[6]:>6.0f} {pcts[7]:>6.0f} {pcts[8]:>6.0f}")

h_lo, h_hi = int(np.percentile(h, 2)), int(np.percentile(h, 98))
s_lo, s_hi = int(np.percentile(s, 2)), int(np.percentile(s, 98))
v_lo, v_hi = int(np.percentile(v, 2)), int(np.percentile(v, 98))

print(f"\nSuggested HSV bounds (2nd–98th percentile of real cone pixels):")
print(f"  lower_orange = np.array([{h_lo}, {s_lo}, {v_lo}])")
print(f"  upper_orange = np.array([{h_hi}, {s_hi}, {v_hi}])")

# save histogram image for report
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("HSV Distribution of Ground-Truth Cone Pixels (all 20 test images)", fontsize=13)

    for ax, arr, name, color, lo, hi in [
        (axes[0], h, "Hue (H)", "orangered", h_lo, h_hi),
        (axes[1], s, "Saturation (S)", "darkorange", s_lo, s_hi),
        (axes[2], v, "Value (V)", "gold", v_lo, v_hi),
    ]:
        ax.hist(arr, bins=50, color=color, edgecolor="black", alpha=0.8)
        ax.axvline(lo, color="blue", linestyle="--", linewidth=1.5, label=f"P2={lo}")
        ax.axvline(hi, color="red", linestyle="--", linewidth=1.5, label=f"P98={hi}")
        ax.set_title(name)
        ax.set_xlabel("Value (0-255 for S,V; 0-180 for H)")
        ax.set_ylabel("Pixel count")
        ax.legend()

    plt.tight_layout()
    plt.savefig("hsv_histogram.png", dpi=150)
    print("\nHistogram saved to hsv_histogram.png (include this in your report)")
except ImportError:
    print("\n(matplotlib not available — skipping histogram. Install with: pip install matplotlib)")
