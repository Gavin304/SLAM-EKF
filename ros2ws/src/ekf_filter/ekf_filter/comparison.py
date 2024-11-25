#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class ErrorComparisonNode(Node):
    def __init__(self):
        super().__init__('error_comparison_node')

        # Subscribers
        self.ekf_subscription = self.create_subscription(
            Float32MultiArray,
            '/ekf_state',
            self.ekf_callback,
            10
        )
        self.pose_subscription = self.create_subscription(
            Pose,
            '/pose_data',
            self.pose_callback,
            10
        )

        # Variables to store the latest EKF and ground truth data
        self.ekf_state = None
        self.ground_truth_pose = None

        # Queues to store errors over time
        self.x_percentage_errors = deque(maxlen=1000)  # Store up to 1000 error values
        self.y_percentage_errors = deque(maxlen=1000)
        self.time_steps = deque(maxlen=1000)

        # Initialize a counter for time steps
        self.time_step = 0

        # Set up the plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 6))
        self.ax[0].set_title("X Percentage Error Over Time")
        self.ax[0].set_xlabel("Time Step")
        self.ax[0].set_ylabel("X Error (%)")
        self.ax[1].set_title("Y Percentage Error Over Time")
        self.ax[1].set_xlabel("Time Step")
        self.ax[1].set_ylabel("Y Error (%)")

    def ekf_callback(self, msg):
        # Store the EKF state data
        self.ekf_state = np.array(msg.data)
        self.compare_error()  # Call to compare error if both data are available

    def pose_callback(self, msg):
        # Store the ground truth pose data
        self.ground_truth_pose = np.array([msg.position.x, msg.position.y])
        self.compare_error()  # Call to compare error if both data are available

    def compare_error(self):
        # Ensure both EKF state and ground truth pose are available
        if self.ekf_state is not None and self.ground_truth_pose is not None:
            # Extract the EKF x and y positions
            ekf_x = self.ekf_state[0]  # x position
            ekf_y = self.ekf_state[1]  # y position

            # Extract the ground truth x and y positions
            ground_truth_x = self.ground_truth_pose[0] * 1000
            ground_truth_y = self.ground_truth_pose[1] * 1000

            # Compute the x and y errors
            x_error = abs(ekf_x - ground_truth_x)
            y_error = abs(ekf_y - ground_truth_y)

            # Compute the percentage errors
            x_percentage_error = (x_error / abs(ground_truth_x)) * 100 if ground_truth_x != 0 else 0
            y_percentage_error = (y_error / abs(ground_truth_y)) * 100 if ground_truth_y != 0 else 0

            # Append percentage errors and time step to the queues
            self.x_percentage_errors.append(x_percentage_error)
            self.y_percentage_errors.append(y_percentage_error)
            self.time_steps.append(self.time_step)
            self.time_step += 1

            # Update the plot
            self.update_plot()

            # Log the percentage errors
            self.get_logger().info(f'X Percentage Error: {x_percentage_error:.2f}%, Y Percentage Error: {y_percentage_error:.2f}%')

    def update_plot(self):
        # Clear the previous plots
        self.ax[0].cla()
        self.ax[1].cla()

        # Plot the x percentage error
        self.ax[0].plot(self.time_steps, self.x_percentage_errors, label="X Percentage Error")
        self.ax[0].set_title("X Percentage Error Over Time")
        self.ax[0].set_xlabel("Time Step")
        self.ax[0].set_ylabel("X Error (%)")
        self.ax[0].legend()

        # Plot the y percentage error
        self.ax[1].plot(self.time_steps, self.y_percentage_errors, label="Y Percentage Error", color='orange')
        self.ax[1].set_title("Y Percentage Error Over Time")
        self.ax[1].set_xlabel("Time Step")
        self.ax[1].set_ylabel("Y Error (%)")
        self.ax[1].legend()

        # Draw the updated plot
        plt.pause(0.001)  # Pause to update the plot

def main(args=None):
    rclpy.init(args=args)
    error_comparison_node = ErrorComparisonNode()
    rclpy.spin(error_comparison_node)
    error_comparison_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
