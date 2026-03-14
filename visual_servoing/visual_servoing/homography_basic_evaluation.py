#!/usr/bin/env python3
"""
Leave-one-out cross-validation for homography calibration points.

For each point, computes the homography from the remaining 5 points
and measures the prediction error on the held-out point.
"""

import numpy as np
import cv2

# Copied from homography_transformer.py so this script runs without ROS
PTS_IMAGE_PLANE = [[582, 183], [325, 191], [109, 214], [639, 246], [479, 198], [464, 335], [615, 217], [127, 195]]
PTS_GROUND_PLANE = [[87.01, -57.87], [75.98, 0], [50.0, 25.2], [35.83, -20.87], [64.17, -20.47], [24.41, -3.94], [51.57, -33.46], [67.32, 35.04]]
METERS_PER_INCH = 0.0254


def main():
    pts_image = np.float32(PTS_IMAGE_PLANE)
    pts_ground = np.float32(PTS_GROUND_PLANE) * METERS_PER_INCH

    errors = []
    rel_errors = []

    print(f"{'Point':<8} {'Actual (m)':<24} {'Predicted (m)':<24} {'Error (m)':<12} {'Error (in)':<12} {'Rel (%)'}")
    print("=" * 92)

    for i in range(len(pts_image)):
        # Exclude point i
        train_image = np.delete(pts_image, i, axis=0)[:, np.newaxis, :]
        train_ground = np.delete(pts_ground, i, axis=0)[:, np.newaxis, :]

        h, _ = cv2.findHomography(train_image, train_ground)

        # Predict the excluded point
        u, v = pts_image[i]
        pt = np.array([[u], [v], [1.0]])
        xy = h @ pt
        xy /= xy[2, 0]
        px, py = xy[0, 0], xy[1, 0]

        ax, ay = pts_ground[i]
        err = np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        dist = np.sqrt(ax ** 2 + ay ** 2)
        rel_err = (err / dist) * 100.0
        errors.append(err)
        rel_errors.append(rel_err)

        print(f"{i+1:<8} ({ax:.4f}, {ay:.4f}){'':<6} ({px:.4f}, {py:.4f}){'':<6} {err:.4f}{'':<6} {err / METERS_PER_INCH:.2f}{'':<6} {rel_err:.1f}")

    print("=" * 92)
    print(f"Mean error:  {np.mean(errors):.4f}m  ({np.mean(errors) / METERS_PER_INCH:.2f} in)  {np.mean(rel_errors):.1f}%")
    print(f"Std dev:     {np.std(errors):.4f}m  ({np.std(errors) / METERS_PER_INCH:.2f} in)  {np.std(rel_errors):.1f}%")
    print(f"Max error:   {np.max(errors):.4f}m  ({np.max(errors) / METERS_PER_INCH:.2f} in)  {np.max(rel_errors):.1f}%")


if __name__ == "__main__":
    main()
