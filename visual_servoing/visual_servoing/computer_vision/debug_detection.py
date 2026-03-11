"""
Visual Debugger for Cone Color Segmentation

Shows side-by-side for each test image:
  1. Original image with ground truth (green) and detected (red) bounding boxes
  2. HSV mask after color thresholding
  3. Mask after morphological cleanup

Usage:
    python3 debug_detection.py           # show all 20 images one by one
    python3 debug_detection.py 7         # show only test7.jpg
    python3 debug_detection.py zeros     # show only images that scored 0

Press any key to advance to the next image.
"""

import cv2
import numpy as np
import csv
import ast
import sys
from color_segmentation import cd_color_segmentation

CSV_PATH = "./test_images_cone/test_images_cone.csv"

# ── Load ground truth ──
ground_truth = {}
with open(CSV_PATH) as f:
    for row in csv.reader(f):
        ground_truth[row[0]] = ast.literal_eval(row[1])


def iou(b1, b2):
    xi1 = max(b1[0][0], b2[0][0]); yi1 = max(b1[0][1], b2[0][1])
    xi2 = min(b1[1][0], b2[1][0]); yi2 = min(b1[1][1], b2[1][1])
    inter = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    a1 = (b1[1][0]-b1[0][0]+1) * (b1[1][1]-b1[0][1]+1)
    a2 = (b2[1][0]-b2[0][0]+1) * (b2[1][1]-b2[0][1]+1)
    return inter / float(a1 + a2 - inter)


def get_intermediate_masks(img):
    """Return the mask after color filter and after morphology, for debugging."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 80, 50])
    upper_orange = np.array([30, 255, 255])
    mask_color = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = np.ones((5, 5), np.uint8)
    mask_morph = cv2.erode(mask_color, kernel, iterations=1)
    mask_morph = cv2.dilate(mask_morph, kernel, iterations=2)
    return mask_color, mask_morph


def debug_image(img_path, bbox_true):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot load {img_path}")
        return

    bbox_est = cd_color_segmentation(img, None)
    score = iou(bbox_est, bbox_true)

    mask_color, mask_morph = get_intermediate_masks(img)

    # draw bounding boxes on a copy
    display = img.copy()
    cv2.rectangle(display, bbox_true[0], bbox_true[1], (0, 255, 0), 2)   # green = ground truth
    cv2.rectangle(display, bbox_est[0], bbox_est[1], (0, 0, 255), 2)      # red   = detected
    cv2.putText(display, f"IoU: {score:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(display, "GREEN=truth  RED=detected", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # convert masks to BGR so we can stack them
    mask_color_bgr = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)
    mask_morph_bgr = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)

    # label masks
    cv2.putText(mask_color_bgr, "After color filter", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(mask_morph_bgr, "After morphology", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # resize all to same height for side-by-side display
    h = display.shape[0]
    def resize_h(im, height):
        scale = height / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * scale), height))

    row = np.hstack([
        resize_h(display, h),
        resize_h(mask_color_bgr, h),
        resize_h(mask_morph_bgr, h),
    ])

    import os
    os.makedirs("debug_output", exist_ok=True)
    name = os.path.basename(img_path).replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
    out_path = f"debug_output/{name}_iou{score:.2f}.png"
    cv2.imwrite(out_path, row)
    print(f"  Saved → {out_path}")


# ── Main ──
items = list(ground_truth.items())

if len(sys.argv) == 2:
    arg = sys.argv[1]
    if arg == "zeros":
        # only show images that scored 0
        items = [(p, b) for p, b in items
                 if iou(cd_color_segmentation(cv2.imread(p), None), b) == 0.0]
        print(f"Showing {len(items)} images with IoU=0")
    else:
        # show a specific test number, e.g. "7" → test7.jpg
        target = f"./test_images_cone/test{arg}.jpg"
        items = [(p, b) for p, b in items if p == target]
        if not items:
            print(f"Image not found: {target}")
            sys.exit(1)

for img_path, bbox_true in items:
    debug_image(img_path, bbox_true)
