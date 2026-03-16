#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from visual_servoing.computer_vision.color_segmentation import cd_color_segmentation
from visual_servoing.computer_vision.color_segmentation import cd_color_segmentation_line


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        super().__init__("cone_detector")

        # Line following mode: set via launch param, defaults to cone parking
        self.declare_parameter("line_following", False)
        self.line_following = self.get_parameter("line_following").value

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge()  # Converts between ROS images and OpenCV Images

        # Annular mask for line following (lazy-initialized on first frame)
        self.annular_mask = None
        self.inner_radius = 50   # px — masks out ground directly under car
        self.outer_radius = 700  # px — covers most of the visible ground

        mode = "line following" if self.line_following else "cone parking"
        self.get_logger().info(f"Cone Detector Initialized (mode: {mode})")

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        if self.line_following:
            h, w = image.shape[:2]

            # Build annular mask once (lazy init based on actual image size)
            if self.annular_mask is None:
                self.annular_mask = np.zeros((h, w), dtype=np.uint8)
                center = (w // 2, h)  # Bottom-center of image
                cv2.circle(self.annular_mask, center, self.outer_radius, 255, -1)
                cv2.circle(self.annular_mask, center, self.inner_radius, 0, -1)

            masked_image = cv2.bitwise_and(image, image, mask=self.annular_mask)
            bounding_box = cd_color_segmentation_line(masked_image)

            if bounding_box is not None:
                (x1, y1), (x2, y2) = bounding_box
                u = float((x1 + x2) / 2)
                v = float((y1 + y2) / 2)

                cone_msg = ConeLocationPixel()
                cone_msg.u = u
                cone_msg.v = v
                self.cone_pub.publish(cone_msg)
                self.get_logger().info(f"Line detected: pixel=({u:.0f}, {v:.0f})", throttle_duration_sec=1.0)
            else:
                self.get_logger().warn("No orange detected in annular mask", throttle_duration_sec=2.0)

            debug_msg = self.bridge.cv2_to_imgmsg(masked_image, "bgr8")
            self.debug_pub.publish(debug_msg)
            return

        # Computes bounding box with color segmentation
        bounding_box = cd_color_segmentation(image, None)

        if bounding_box is not None:
            (x1, y1), (x2, y2) = bounding_box
            # Bottom-center pixel (on the ground plane)
            u = float((x1 + x2) / 2)
            v = float(y2)

            cone_msg = ConeLocationPixel()
            cone_msg.u = u
            cone_msg.v = v
            self.cone_pub.publish(cone_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
