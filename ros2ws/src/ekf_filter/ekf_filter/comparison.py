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
        self.slam_ekf_subscription = self.create_subscription(
            Float32MultiArray,
            '/slam_ekf_state',
            self.slam_ekf_callback,
            10
        )
        self.pose_subscription = self.create_subscription(
            Pose,
            '/pose_data',
            self.pose_callback,
            10
        )

        # Variables to store the latest EKF, SLAM EKF, and ground truth data
        self.ekf_state = None
        self.slam_ekf_state = None
        self.ground_truth_pose = None

        # Queues to store errors over time
        self.x_ekf_percentage_errors = deque(maxlen=1000)
        self.y_ekf_percentage_errors = deque(maxlen=1000)
        self.x_slam_ekf_percentage_errors = deque(maxlen=1000)
        self.y_slam_ekf_percentage_errors = deque(maxlen=1000)
        self.time_steps = deque(maxlen=1000)

        # Initialize a counter for time steps
        self.time_step = 0

        # Set up the plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(4, 1, figsize=(10, 12))

        # Set titles and labels for each subplot
        self.ax[0].set_title("X Percentage Error (EKF vs Pose)")
        self.ax[0].set_xlabel("Time Step")
        self.ax[0].set_ylabel("X Error (%)")

        self.ax[1].set_title("Y Percentage Error (EKF vs Pose)")
        self.ax[1].set_xlabel("Time Step")
        self.ax[1].set_ylabel("Y Error (%)")

        self.ax[2].set_title("X Percentage Error (SLAM EKF vs Pose)")
        self.ax[2].set_xlabel("Time Step")
        self.ax[2].set_ylabel("X Error (%)")

        self.ax[3].set_title("Y Percentage Error (SLAM EKF vs Pose)")
        self.ax[3].set_xlabel("Time Step")
        self.ax[3].set_ylabel("Y Error (%)")

    def ekf_callback(self, msg):
        # Store the EKF state data
        self.ekf_state = np.array(msg.data)
        self.compare_error()  # Call to compare error if all data are available

    def slam_ekf_callback(self, msg):
        # Store the SLAM EKF state data
        self.slam_ekf_state = np.array(msg.data)
        self.compare_error()  # Call to compare error if all data are available

    def pose_callback(self, msg):
        # Store the ground truth pose data
        self.ground_truth_pose = np.array([msg.position.x, msg.position.y])
        self.compare_error()  # Call to compare error if all data are available

    def compare_error(self):
        # Ensure EKF state, SLAM EKF state, and ground truth pose are available
        if self.ekf_state is not None and self.slam_ekf_state is not None and self.ground_truth_pose is not None:
            # Extract positions
            ekf_x, ekf_y = self.ekf_state[0], self.ekf_state[1]
            slam_ekf_x, slam_ekf_y = self.slam_ekf_state[0], self.slam_ekf_state[1]
            ground_truth_x, ground_truth_y = self.ground_truth_pose[0] * 1000, self.ground_truth_pose[1] * 1000

            # Compute percentage errors for EKF
            x_ekf_error = abs(ekf_x - ground_truth_x)
            y_ekf_error = abs(ekf_y - ground_truth_y)
            x_ekf_percentage_error = (x_ekf_error / abs(ground_truth_x)) * 100 if ground_truth_x != 0 else 0
            y_ekf_percentage_error = (y_ekf_error / abs(ground_truth_y)) * 100 if ground_truth_y != 0 else 0

            # Compute percentage errors for SLAM EKF
            x_slam_ekf_error = abs(slam_ekf_x - ground_truth_x)
            y_slam_ekf_error = abs(slam_ekf_y - ground_truth_y)
            x_slam_ekf_percentage_error = (x_slam_ekf_error / abs(ground_truth_x)) * 100 if ground_truth_x != 0 else 0
            y_slam_ekf_percentage_error = (y_slam_ekf_error / abs(ground_truth_y)) * 100 if ground_truth_y != 0 else 0

            # Append errors and time step to queues
            self.x_ekf_percentage_errors.append(x_ekf_percentage_error)
            self.y_ekf_percentage_errors.append(y_ekf_percentage_error)
            self.x_slam_ekf_percentage_errors.append(x_slam_ekf_percentage_error)
            self.y_slam_ekf_percentage_errors.append(y_slam_ekf_percentage_error)
            self.time_steps.append(self.time_step)
            self.time_step += 1

            # Update the plot
            self.update_plot()

    def update_plot(self):
        # Clear previous plots
        for ax in self.ax:
            ax.cla()

        # Plot EKF errors
        self.ax[0].plot(self.time_steps, self.x_ekf_percentage_errors, label="EKF X Error", color='blue')
        self.ax[1].plot(self.time_steps, self.y_ekf_percentage_errors, label="EKF Y Error", color='green')

        # Plot SLAM EKF errors
        self.ax[2].plot(self.time_steps, self.x_slam_ekf_percentage_errors, label="SLAM EKF X Error", color='orange')
        self.ax[3].plot(self.time_steps, self.y_slam_ekf_percentage_errors, label="SLAM EKF Y Error", color='red')

        # Add titles and labels
        for i, ax in enumerate(self.ax):
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Error (%)")
            ax.legend()

        # Draw updated plots
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    error_comparison_node = ErrorComparisonNode()
    rclpy.spin(error_comparison_node)
    error_comparison_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
