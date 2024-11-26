import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np

testing = False
# 還沒測試正式版

class MovementControllerSim(Node):
    def __init__(self):
        super().__init__('movement_controller_sim')
        self.vx = 0
        # self.vy = 0 # 車子應該是不會有 vy
        self.w = 0

        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel_human',
            self.get_cmd_callback,
            10
        )

        self.move_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.move_ctrl)
        self.i = 0

    def get_cmd_callback(self, msg):
        self.vx = float(msg.linear.x)
        # self.vy = float(msg.linear.y)   # 車子應該是不會有 vy
        self.w = float(msg.angular.z)

    def move_ctrl(self):
        twist = Twist()

        if testing:
            self.vx = 1.0
            self.w = 0.01
            twist.linear.x = self.vx
            twist.angular.z = self.w
        else:
            twist.linear.x = float(self.vx)
            # twist.linear.y = float(self.vy)
            twist.angular.z = float(self.w)

        if self.i == 0:
            twist.angular.x = 1.0
        else:
            twist.angular.x = 0.0

        self.move_publisher.publish(twist)
        self.get_logger().info(f'{self.i} times,twist: {twist}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    movement_controller_sim = MovementControllerSim()

    rclpy.spin(movement_controller_sim)

    movement_controller_sim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()