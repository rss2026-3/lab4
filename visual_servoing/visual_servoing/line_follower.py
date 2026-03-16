#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
import numpy as np


from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped


class LineFollower(Node):
   """
   A controller for following an orange line.
   Listens for a relative cone location (virtual target on the line)
   and publishes control commands at a fixed rate.
   """

   def __init__(self):
       super().__init__("line_follower")

       self.declare_parameter("drive_topic", "/vesc/high_level/input/nav_1")
       DRIVE_TOPIC = self.get_parameter("drive_topic").value

       self.declare_parameter("max_speed", 0.5)
       self.max_speed = self.get_parameter("max_speed").value

       self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
       self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

       self.create_subscription(
           ConeLocation, "/relative_cone", self.relative_cone_callback, 1)

       self.relative_x = 0.
       self.relative_y = 0.
       self.drive_cmd = AckermannDriveStamped()

       # Publish drive commands at 20Hz to outpace the safety controller's zero stream
       self.timer = self.create_timer(1.0 / 20.0, self.timer_callback)

       self.get_logger().info(f"Line Follower Initialized (max_speed: {self.max_speed})")

   def relative_cone_callback(self, msg):
       self.relative_x = msg.x_pos
       self.relative_y = msg.y_pos

       angle_to_target = np.arctan2(self.relative_y, self.relative_x)

       # Steering: proportional to angle toward target point on line
       Ksteering = 1
       self.drive_cmd.drive.steering_angle = float(np.clip(Ksteering * angle_to_target, -0.34, 0.34))

       # Constant forward speed, slow down on sharp turns
       turn_factor = 1.0 - 0.5 * min(abs(angle_to_target) / (np.pi / 3), 1.0)
       speed = self.max_speed * turn_factor
       self.drive_cmd.drive.speed = float(np.clip(speed, 0.0, self.max_speed))

       self.get_logger().info(
           f"target=({self.relative_x:.2f}, {self.relative_y:.2f})m "
           f"angle={np.degrees(angle_to_target):.1f}deg "
           f"speed={self.drive_cmd.drive.speed:.2f} "
           f"steer={np.degrees(self.drive_cmd.drive.steering_angle):.1f}deg",
           throttle_duration_sec=1.0)

   def timer_callback(self):
       self.drive_pub.publish(self.drive_cmd)
       self.error_publisher()

   def error_publisher(self):
       error_msg = ParkingError()
       error_msg.x_error = self.relative_x
       error_msg.y_error = self.relative_y
       error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)
       self.error_pub.publish(error_msg)


def main(args=None):
   rclpy.init(args=args)
   lf = LineFollower()
   rclpy.spin(lf)
   rclpy.shutdown()

if __name__ == '__main__':
   main()
