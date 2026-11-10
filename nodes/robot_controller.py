#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from rclpy.executors import MultiThreadedExecutor
from dead_reckoning_nav import INFTY
import math

#Le puse sleep pq se mew suscribia antes de tiempo y como q me lo bugeaba xd
import time

MAX_VEL_EFFORT = 0.2
VEL_KP = 1.0 #1.0
VEL_KI = 0.3 #(1.0, 0.3)
VEL_KD = 0.0

MAX_ANGLE_EFFORT = 1.0
ANGLE_KP = 0.3 #0.3
ANGLE_KI = 0.3 #(0.3, 0.2)
ANGLE_KD = 0.0


class PIDController( Node ):
  def __init__(self, name, setpoint_topic, measures_topic, effort_topic, max_effort, kp, ki = 0, kd = 0, angular=False):
    super().__init__(name)
    self.name = name
    self.kp = kp
    self.ki = ki
    self.kd = kd

    self.setpoint = None
    self.state = None
    self.max_effort=max_effort
    self.angular = angular
    
    self.accumulated_error = 0
    self.last_error = 0
    self.last_check = 0

    self.dist_set_point_sub = self.create_subscription( Float64, setpoint_topic, self.setpoint_cb, 1)
    self.dist_state_sub = self.create_subscription( Float64, measures_topic, self.state_cb, 1)
    self.actuation_pub = self.create_publisher( Float64, effort_topic, 1)

  def check_connections(self):
      #Loop para q no se bugeen los one shot
      while rclpy.ok():
        self.get_logger().info(f"Esperando subs de {self.name}")

        if (self.actuation_pub.get_subscription_count() > 0):
          break
        time.sleep(0.1)

      self.get_logger().info(f"[{self.name}] Inicializando Controller...")


  def setpoint_cb(self, msg):
      if self.setpoint == msg.data:
          return
      
      if(msg.data == INFTY):
        self.get_logger().info(f"[{self.name}] Quedandose quietito :p")
      else:
        self.get_logger().info(f"[{self.name}] Nuevo setpoint: {msg.data}")
      self.reset()
      self.setpoint = msg.data

  def state_cb(self, msg):
    if self.setpoint == None or self.setpoint == INFTY:
      return
    
    self.state = msg.data
    time_now = self.get_clock().now().nanoseconds / 1e9

    #Primer check
    if(self.last_check == 0):
        self.last_check = time_now
        return

    dt = time_now-self.last_check
    self.last_check = time_now
    if dt <=0:
       return

    error = self.setpoint - self.state
    if self.angular:
       error = math.atan2(math.sin(error), math.cos(error))

    deviation = (error-self.last_error)/dt
    self.accumulated_error += error*dt
    self.last_error = error

    #PID
    p_actuation = self.kp*error
    i_actuation = self.ki*self.accumulated_error
    d_actuation = self.kd*deviation

    #Actuacion
    actuation = p_actuation + i_actuation + d_actuation

    if(actuation < -self.max_effort):
       actuation = -self.max_effort
    elif(actuation > self.max_effort):
       actuation = self.max_effort

    #Enviar
    msg = Float64()
    msg.data = actuation
    self.actuation_pub.publish(msg)
    self.get_logger().info(f"[{self.name}] [Pose: {self.state} | Setpoint: {self.setpoint} | Error: {error}] | PID: {msg.data}")


  def reset(self):
    self.setpoint = None
    self.state = None
    self.accumulated_error = 0
    self.last_error = 0
    self.last_check = 0


def main():
    rclpy.init()
    velocity_pid = PIDController("Velocity_controller", "/velocity_setpoint", "/velocity_measure", "/velocity_effort", max_effort=MAX_VEL_EFFORT, kp=VEL_KP, ki=VEL_KI, kd=VEL_KD)
    angle_pid = PIDController("Angle_controller", "/angle_setpoint", "/angle_measure", "/angle_effort", max_effort=MAX_ANGLE_EFFORT, kp=ANGLE_KP, ki=ANGLE_KI, kd=ANGLE_KD, angular=True)

    #le agregamos el executor pq con spin normal se nos bugeaba :c
    executor = MultiThreadedExecutor()
    executor.add_node(velocity_pid)
    executor.add_node(angle_pid)

    velocity_pid.check_connections()
    angle_pid.check_connections()

    try:
       executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        velocity_pid.destroy_node()
        angle_pid.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
  main()



