#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Bool
from geometry_msgs.msg import Twist
import message_filters

ACCEL_SUBIDA = 0.8 # 0.25 segs 0 -> 0.2
ACCEL_BAJADA = 1.2 # 0.125 segs 0.2 -> 0.0
KP = 0.008

class PIDController( Node ):

  def __init__( self):
    super().__init__( 'p_controller' )
    self.kp = KP
    self.cmd_vel_pub = self.create_publisher(Twist, "/commands/velocity", 10)
    self.blue_square_position_sub = message_filters.Subscriber(self, Int16, '/blue_square/position')
    self.blue_square_detected_sub = message_filters.Subscriber(self, Bool, '/blue_square/detected')

    self.get_logger().info("Iniciando ...")
    sync = message_filters.ApproximateTimeSynchronizer([self.blue_square_position_sub, self.blue_square_detected_sub], 1, 0.05, allow_headerless=True)
    sync.registerCallback(self.actuacion_cb)

    self.vel = 0
    self.prev_time = None

  @property
  def dt(self):
    now = self.get_clock().now().nanoseconds / 10 ** 9
    if self.prev_time is not None:
      dt = now - self.prev_time
      self.prev_time = now
      return dt
    else:
      self.prev_time = now
      return 0.0
    
  def velocidad(self, vel_deseada):
    if vel_deseada > self.vel:
      self.vel = min(self.vel + ACCEL_SUBIDA*self.dt, vel_deseada)
    else:
      self.vel = max(self.vel - ACCEL_BAJADA*self.dt, vel_deseada)
    return self.vel

  def actuacion_cb(self, pos_x_obj, detected):
    speed = Twist()
    if not detected.data:
      self.get_logger().info("Sin objeto detectado")
      speed.linear.x, speed.angular.z = (self.velocidad(0.0), 0.0)
      self.cmd_vel_pub.publish(speed)
      return

    self.get_logger().info(f"Objeto detectado en x={pos_x_obj.data}")
    # Proportional
    p_actuation = self.kp*-pos_x_obj.data

    # Message sending
    speed.angular.z = p_actuation
    speed.linear.x = self.velocidad(0.2) if abs(pos_x_obj.data)<50 else self.velocidad(0.0)
    self.cmd_vel_pub.publish(speed)


def main():
  rclpy.init()
  p_ctrl = PIDController()
  rclpy.spin( p_ctrl )

if __name__ == '__main__':
  main()



