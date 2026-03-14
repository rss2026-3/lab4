#!/usr/bin/env python3
"""
Click on the camera feed to print pixel coordinates.

Usage:
    source ~/racecar_ws/install/setup.bash
    python3 pixel_clicker.py

Run from the VNC/noVNC desktop session (same display as rviz).
Click on the image to print (u, v) coordinates. Press 'q' to quit.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class PixelClicker(Node):
    def __init__(self):
        super().__init__('pixel_clicker')
        self.bridge = CvBridge()
        self.latest_image = None
        self.sub = self.create_subscription(
            Image, '/zed/zed_node/rgb/image_rect_color', self.callback, 10)

    def callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel: ({x}, {y})")


def main():
    rclpy.init()
    node = PixelClicker()
    cv2.namedWindow('Click to get pixel coords')
    cv2.setMouseCallback('Click to get pixel coords', on_click)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.03)
        if node.latest_image is not None:
            cv2.imshow('Click to get pixel coords', node.latest_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
