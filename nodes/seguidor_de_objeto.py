#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Bool
from geometry_msgs.msg import Twist
import message_filters


KP = 0.02

class PIDController( Node ):

  def __init__( self):
    super().__init__( 'p_controller' )
    self.kp = KP
    self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
    self.blue_square_position_sub = message_filters.Subscriber(self, Int16, '/blue_square/position')
    self.blue_square_detected_sub = message_filters.Subscriber(self, Bool, '/blue_square/detected')

    self.get_logger().info("Iniciando ...")
    sync = message_filters.ApproximateTimeSynchronizer([self.blue_square_position_sub, self.blue_square_detected_sub], 1, 0.05, allow_headerless=True)
    sync.registerCallback(self.actuacion_cb)
            

  def actuacion_cb(self, pos_x_obj, detected):
    speed = Twist()
    if not detected.data:
      self.get_logger().info("Sin objeto detectado")
      speed.linear.x, speed.angular.z = (0.0, 0.0)
      self.cmd_vel_pub.publish(speed)
      return

    self.get_logger().info(f"Objeto detectado en x={pos_x_obj.data}")
    # Proportional
    p_actuation = self.kp*-pos_x_obj.data

    # Message sending
    speed.angular.z = p_actuation
    speed.linear.x = 0.2 if abs(pos_x_obj.data)<50 else 0.0
    self.cmd_vel_pub.publish(speed)


def main():
  rclpy.init()
  p_ctrl = PIDController()
  rclpy.spin( p_ctrl )

if __name__ == '__main__':
  main()



