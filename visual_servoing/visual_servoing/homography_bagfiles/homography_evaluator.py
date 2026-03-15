#!/usr/bin/env python3
"""
Homography evaluation script.

Usage:
    1. Place cone at a known position
    2. Record a short bag with the following command:
       ros2 bag record /zed/zed_node/rgb/image_rect_color -o bagfile
    3. Run this script directly on the bag file, where X and Y are the measured coordinates:
       python3 homography_validator.py bagfile --x_inches X --y_inches Y
    The script reads images from the bag, runs color segmentation + homography
    transform offline, and compares predictions to your measured location.
"""

import sys
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sqlite3
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
# Copied from homography_transformer.py so this script runs without ROS package install
PTS_IMAGE_PLANE = [[582, 183], [325, 191], [109, 214], [639, 246], [479, 198], [464, 335], [615, 217], [127, 195]]
PTS_GROUND_PLANE = [[87.01, -57.87], [75.98, 0], [50.0, 25.2], [35.83, -20.87], [64.17, -20.47], [24.41, -3.94], [51.57, -33.46], [67.32, 35.04]]
METERS_PER_INCH = 0.0254


def cd_color_segmentation(img, template):
    """Copied from color_segmentation.py so this script runs without ROS package install."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 200, 80])
    upper_orange = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ((0, 0), (0, 0))

    cone_candidates = [c for c in contours
                       if cv2.boundingRect(c)[2] <= 2.5 * cv2.boundingRect(c)[3]]
    if not cone_candidates:
        cone_candidates = contours

    largest = max(cone_candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return ((x, y), (x + w, y + h))

def build_homography():
    np_pts_ground = np.array(PTS_GROUND_PLANE) * METERS_PER_INCH
    np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])
    np_pts_image = np.float32(np.array(PTS_IMAGE_PLANE)[:, np.newaxis, :])

    h, _ = cv2.findHomography(np_pts_image, np_pts_ground)
    return h


def transform_uv_to_xy(h, u, v):
    homogeneous_point = np.array([[u], [v], [1]])
    xy = np.dot(h, homogeneous_point)
    scaling_factor = 1.0 / xy[2, 0]
    homogeneous_xy = xy * scaling_factor
    return homogeneous_xy[0, 0], homogeneous_xy[1, 0]


def imgmsg_to_cv2_bgr(msg):
    """Convert sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    dtype = np.uint8
    n_channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}.get(msg.encoding, 3)
    img = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.width, n_channels)
    if msg.encoding in ("rgb8", "rgba8"):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgra8":
        img = img[:, :, :3]
    return img


def read_bag_images(bag_path):
    db_files = [
        os.path.join(bag_path, f)
        for f in os.listdir(bag_path)
        if f.endswith(".db3")
    ]
    if not db_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_path}")

    images = []
    for db_file in sorted(db_files):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "SELECT data FROM messages "
            "JOIN topics ON messages.topic_id = topics.id "
            "WHERE topics.name = '/zed/zed_node/rgb/image_rect_color' "
            "ORDER BY messages.timestamp"
        )
        for (data,) in cur.fetchall():
            msg = deserialize_message(bytes(data), Image)
            images.append(imgmsg_to_cv2_bgr(msg))
        conn.close()

    return images


def main():
    parser = argparse.ArgumentParser(description="Evaluate homography accuracy from a bag file.")
    parser.add_argument("bag_path", help="Path to the ROS2 bag directory")
    parser.add_argument("--x_inches", type=float, default=None, help="Measured x position in inches")
    parser.add_argument("--y_inches", type=float, default=None, help="Measured y position in inches")
    args = parser.parse_args()

    has_measurement = args.x_inches is not None and args.y_inches is not None

    if not has_measurement:
        print("Actual measurement not found, can't compute error.")
        print("Will still show predictions from the bag.\n")

    # Build homography from calibration points
    h = build_homography()

    # Read images from bag
    print(f"Reading images from {args.bag_path}...")
    images = read_bag_images(args.bag_path)
    print(f"Found {len(images)} frames.\n")

    if not images:
        print("No images found in bag.")
        return

    # Run pipeline on each frame
    predictions = []
    prediction_frames = []
    for i, image in enumerate(images):
        bounding_box = cd_color_segmentation(image, None)
        if bounding_box is not None:
            (x1, y1), (x2, y2) = bounding_box
            u = float((x1 + x2) / 2)
            v = float(y2)
            x, y = transform_uv_to_xy(h, u, v)
            predictions.append((x, y))
            prediction_frames.append(i + 1)
            print(f"Frame {i+1}: pixel=({u:.0f}, {v:.0f}) -> x={x:.4f}m, y={y:.4f}m")
        else:
            print(f"Frame {i+1}: no cone detected")

    # Summary
    if not predictions:
        print("\nNo cone detected in any frame.")
        return

    xs = [p[0] for p in predictions]
    ys = [p[1] for p in predictions]
    avg_x = np.mean(xs)
    avg_y = np.mean(ys)

    print(f"\n{'='*50}")
    print(f"Frames with detection: {len(predictions)}/{len(images)}")
    print(f"Average prediction:  x={avg_x:.4f}m, y={avg_y:.4f}m")
    print(f"Std dev:             x={np.std(xs):.4f}m, y={np.std(ys):.4f}m")

    if has_measurement:
        measured_x = args.x_inches * METERS_PER_INCH
        measured_y = args.y_inches * METERS_PER_INCH
        x_err = abs(measured_x - avg_x)
        y_err = abs(measured_y - avg_y)
        error = np.sqrt((measured_x - avg_x)**2 + (measured_y - avg_y)**2)

        print(f"Measured location:   x={measured_x:.4f}m, y={measured_y:.4f}m")
        print(f"X error: {x_err:.4f}m")
        print(f"Y error: {y_err:.4f}m")
        print(f"Euclidean distance error: {error:.4f}m")

    print(f"{'='*50}")

    # Generate prediction-over-time charts
    bag_name = os.path.basename(os.path.normpath(args.bag_path))
    chart_path = f"{bag_name}_predictions.png"

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_x.plot(prediction_frames, xs, 'o-', label="Predicted x", markersize=3)
    if has_measurement:
        ax_x.axhline(y=measured_x, color='r', linestyle='--', label=f"Measured x ({measured_x:.4f}m)")
    ax_x.set_ylabel("x (meters)")
    ax_x.set_title("Predicted x over time")
    ax_x.legend()
    ax_x.grid(True, alpha=0.3)
    if avg_x >= 0:
        ax_x.set_ylim(bottom=0)
    else:
        ax_x.set_ylim(top=0)

    ax_y.plot(prediction_frames, ys, 'o-', label="Predicted y", markersize=3)
    if has_measurement:
        ax_y.axhline(y=measured_y, color='r', linestyle='--', label=f"Measured y ({measured_y:.4f}m)")
    ax_y.set_ylabel("y (meters)")
    ax_y.set_xlabel("Frame")
    ax_y.set_title("Predicted y over time")
    ax_y.legend()
    ax_y.grid(True, alpha=0.3)
    if avg_y >= 0:
        ax_y.set_ylim(bottom=0)
    else:
        ax_y.set_ylim(top=0)

    fig.suptitle(f"Homography predictions: {bag_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    print(f"\nChart saved to {chart_path}")


if __name__ == "__main__":
    main()
