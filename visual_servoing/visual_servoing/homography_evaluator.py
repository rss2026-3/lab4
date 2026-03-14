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
import argparse
import numpy as np
import cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag2_py
from visual_servoing.computer_vision.color_segmentation import cd_color_segmentation
from visual_servoing.homography_transformer import PTS_IMAGE_PLANE, PTS_GROUND_PLANE, METERS_PER_INCH

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


def read_bag_images(bag_path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    images = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == "/zed/zed_node/rgb/image_rect_color":
            msg = deserialize_message(data, Image)
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
            images.append(image)

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
    for i, image in enumerate(images):
        bounding_box = cd_color_segmentation(image, None)
        if bounding_box is not None:
            (x1, y1), (x2, y2) = bounding_box
            u = float((x1 + x2) / 2)
            v = float(y2)
            x, y = transform_uv_to_xy(h, u, v)
            predictions.append((x, y))
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


if __name__ == "__main__":
    main()
