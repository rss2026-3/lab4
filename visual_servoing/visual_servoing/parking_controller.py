#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped
class ParkingController(Node):
   """
   A controller for parking in front of a cone.
   Listens for a relative cone location and publishes control commands.
   Can be used in the simulator and on the real robot.
   """
   def __init__(self):
       super().__init__("parking_controller")
       self.declare_parameter("drive_topic")
       DRIVE_TOPIC = self.get_parameter("drive_topic").value  # set in launch file; different for simulator vs racecar
       self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
       self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

       self.create_subscription(
           ConeLocation, "/relative_cone", self.relative_cone_callback, 1)
       self.parking_distance = .5  #can be changed 
       self.relative_x = 0
       self.relative_y = 0
       self.get_logger().info("Parking Controller Initialized")


   def relative_cone_callback(self, msg):
       self.relative_x = msg.x_pos
       self.relative_y = msg.y_pos
       drive_cmd = AckermannDriveStamped()
       #################################
       # YOUR CODE HERE
       # Use relative position and your control law to set drive_cmd
       drive_cmd.header.stamp = self.get_clock().now().to_msg() 
       distance = np.sqrt(self.relative_x**2 + self.relative_y**2)
       angle_to_cone = np.arctan2(self.relative_y, self.relative_x)
       distance_error = distance - self.parking_distance

       # Steering: proportional to angle toward cone
       Ksteering = 1
       K_speed = 0.5
       drive_cmd.drive.steering_angle = float(np.clip(Ksteering * angle_to_cone, -0.34, 0.34))
       if abs(angle_to_cone) > np.pi / 3:
           speed = 0.5 * np.sign(distance_error)   
       else:
           speed = K_speed * distance_error
           if abs(distance_error) < 0.05:
               speed = 0.0
           elif abs(speed) < 0.3:
               speed = 0.3 * np.sign(distance_error)
    
       drive_cmd.drive.speed = float(np.clip(speed, -1.0, 1.0))

       #################################

       self.drive_pub.publish(drive_cmd)
       self.error_publisher()

   def error_publisher(self):
       """
       Publish the error between the car and the cone. We will view this
       with rqt_plot to plot the success of the controller
       """
       error_msg = ParkingError()

       #################################

       # YOUR CODE HERE
       
       error_msg.x_error = self.relative_x - self.parking_distance
       error_msg.y_error = self.relative_y
       error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2) - self.parking_distance

       #################################

       self.error_pub.publish(error_msg)



def main(args=None):
   rclpy.init(args=args)
   pc = ParkingController()
   rclpy.spin(pc)
   rclpy.shutdown()

if __name__ == '__main__':
   main()



