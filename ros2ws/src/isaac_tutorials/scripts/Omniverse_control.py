#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import numpy as np
import math

class SwerveDriveController(Node):
    def __init__(self):
        super().__init__("swerve_drive_controller")

        # Create the publisher. This publisher will publish a JointState message to the /joint_command topic.
        self.publisher_ = self.create_publisher(JointState, "joint_command", 10)

        # Create the subscriber to listen to /cmd_vel topic for velocity commands
        self.subscription = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )

        # Create a JointState message for the swerve drive system
        self.joint_state = JointState()
        self.joint_state.name = [
            "right_front_connector_joint",  # Steering joint (Wheel 1)
            "left_front_connector_joint",   # Steering joint (Wheel 2)
            "left_back_connector_joint",    # Steering joint (Wheel 3)
            "right_back_connector_joint",   # Steering joint (Wheel 4)
            "right_front_wheel_joint",      # Wheel velocity joint (Wheel 1)
            "left_front_wheel_joint",       # Wheel velocity joint (Wheel 2)
            "left_back_wheel_joint",        # Wheel velocity joint (Wheel 3)
            "right_back_wheel_joint",       # Wheel velocity joint (Wheel 4)
        ]

        num_joints = len(self.joint_state.name)
        self.joint_state.position = np.array([0.0] * num_joints, dtype=np.float64).tolist()
        self.joint_state.velocity = np.array([0.0] * num_joints, dtype=np.float64).tolist()

        # Parameters for the swerve drive system
        self.wheel_base = 0.5  # Distance between front and back wheels
        self.track_width = 0.4  # Distance between left and right wheels

        self.wheel_radius = 0.1  # Radius of the wheels

        # Initial Twist message (linear and angular velocities)
        self.current_twist = Twist()

        # Create a timer for publishing joint states
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def cmd_vel_callback(self, msg):
        # Store the received Twist message (linear and angular velocities)
        self.current_twist = msg

    def calculate_swerve_kinematics(self):
        """Calculate the wheel velocities and steering angles based on the current twist command."""
        vx = self.current_twist.linear.x  # Forward velocity
        vy = self.current_twist.linear.y  # Lateral (sideways) velocity
        omega = self.current_twist.angular.z  # Angular velocity (rotation)

        # Calculate intermediate terms
        L = self.wheel_base
        W = self.track_width
        R = math.sqrt(L**2 + W**2)

        # Calculate the velocities for each wheel
        a = vx - omega * (L / 2)
        b = vx + omega * (L / 2)
        c = vy - omega * (W / 2)
        d = vy + omega * (W / 2)

        # Calculate wheel speeds
        wheel_speeds = [
            math.sqrt(b**2 + c**2),  # right_front wheel speed (Wheel 1)
            math.sqrt(b**2 + d**2),  # left_front wheel speed (Wheel 2)
            math.sqrt(a**2 + d**2),  # left_back wheel speed (Wheel 3)
            math.sqrt(a**2 + c**2),  # right_back wheel speed (Wheel 4)
        ]

        # Calculate wheel angles using atan2(y, x) and adjust for the y-axis reference
        wheel_angles = [
            math.atan2(c, b) - math.pi / 2,  # Adjusted angle for Wheel 1
            math.atan2(d, b) - math.pi / 2,  # Adjusted angle for Wheel 2
            math.atan2(d, a) - math.pi / 2,  # Adjusted angle for Wheel 3
            math.atan2(c, a) - math.pi / 2,  # Adjusted angle for Wheel 4
        ]

        # Normalize angles to be within -pi to pi
        wheel_angles = [(angle + math.pi) % (2 * math.pi) - math.pi for angle in wheel_angles]

        return wheel_speeds, wheel_angles

    def timer_callback(self):
        # Update the current timestamp for the joint states
        self.joint_state.header.stamp = self.get_clock().now().to_msg()

        # Calculate the swerve drive kinematics (wheel velocities and steering angles)
        wheel_speeds, wheel_angles = self.calculate_swerve_kinematics()

        # Update joint positions (for steering) and velocities (for wheel speed)
        # Assign wheel angles to the connector joints (steering) and wheel speeds to the wheel joints
        self.joint_state.position = [
            wheel_angles[0],  # right_front_connector_joint (Wheel 1)
            wheel_angles[1],  # left_front_connector_joint (Wheel 2)
            wheel_angles[2],  # left_back_connector_joint (Wheel 3)
            wheel_angles[3],  # right_back_connector_joint (Wheel 4)
            0.0,  # Placeholder for wheel rotation (not necessary for velocity control)
            0.0,  # Placeholder for wheel rotation
            0.0,  # Placeholder for wheel rotation
            0.0,  # Placeholder for wheel rotation
        ]

        self.joint_state.velocity = [
            0.0,  
            0.0,
            0.0,
            0.0,
            wheel_speeds[0],  # right_front_wheel_joint (Wheel 1)
            wheel_speeds[1],  # left_front_wheel_joint (Wheel 2)
            wheel_speeds[2],  # left_back_wheel_joint (Wheel 3)
            wheel_speeds[3],  # right_back_wheel_joint (Wheel 4)
        ]

        # Publish the joint state message with updated values
        self.publisher_.publish(self.joint_state)


def main(args=None):
    rclpy.init(args=args)

    swerve_drive_controller = SwerveDriveController()

    rclpy.spin(swerve_drive_controller)

    swerve_drive_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
