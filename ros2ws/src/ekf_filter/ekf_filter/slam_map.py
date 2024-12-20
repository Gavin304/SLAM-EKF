#!/usr/bin/env python3
"""
2D SLAM Map Visualization Node
Subscribes to /ekf_state to visualize the robot state (first 6 vectors) and map landmarks (remaining vectors).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

class SlamMapNode(Node):    
    def __init__(self):
        super().__init__('slam_map_node')

        # Subscriber for /ekf_state
        self.state_subscription = self.create_subscription(
            Float32MultiArray,
            '/slam_ekf_state',
            self.state_callback,
            10
        )

        # Publisher for RViz2 visualization
        self.map_publisher = self.create_publisher(MarkerArray, '/slam_map', 10)

        # Robot state and landmarks
        self.robot_state = [0.0] * 6  # [x, y, theta, vx, vy, omega]
        self.landmarks = []  # [(x1, y1), (x2, y2), ...]

        self.get_logger().info("SLAM Map Node initialized.")

    def state_callback(self, msg):
        """
        Callback to process the /ekf_state message.
        Extracts robot state (first 6 values) and landmarks (remaining values).
        """
        state_data = msg.data

        # Ensure the state vector is valid
        if len(state_data) < 6:
            self.get_logger().error("Invalid state vector received!")
            return

        # Extract robot state
        self.robot_state = state_data[:6]

        # Extract landmarks (remaining values)
        num_landmarks = (len(state_data) - 6) // 2
        self.landmarks = [
            (state_data[6 + 2 * i], state_data[6 + 2 * i + 1])  # (x, y) for each landmark
            for i in range(num_landmarks)
        ]

        # Publish visualization
        self.publish_map()

    def publish_map(self):
        """
        Publishes the SLAM map as a MarkerArray for visualization in RViz2.
        """
        marker_array = MarkerArray()

        # Add robot marker
        robot_marker = Marker()
        robot_marker.header.frame_id = "map"
        robot_marker.header.stamp = self.get_clock().now().to_msg()
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.SPHERE
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = self.robot_state[0]  # x
        robot_marker.pose.position.y = self.robot_state[1]  # y
        robot_marker.pose.position.z = 0.0
        robot_marker.scale.x = 0.3
        robot_marker.scale.y = 0.3
        robot_marker.scale.z = 0.3
        robot_marker.color.r = 1.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 0.0
        robot_marker.color.a = 1.0
        marker_array.markers.append(robot_marker)

        # Add landmark markers
        for i, (lx, ly) in enumerate(self.landmarks):
            landmark_marker = Marker()
            landmark_marker.header.frame_id = "map"
            landmark_marker.header.stamp = self.get_clock().now().to_msg()
            landmark_marker.ns = "landmarks"
            landmark_marker.id = i + 1
            landmark_marker.type = Marker.CUBE
            landmark_marker.action = Marker.ADD
            landmark_marker.pose.position.x = lx
            landmark_marker.pose.position.y = ly
            landmark_marker.pose.position.z = 0.0
            landmark_marker.scale.x = 0.2
            landmark_marker.scale.y = 0.2
            landmark_marker.scale.z = 0.2
            landmark_marker.color.r = 0.0
            landmark_marker.color.g = 1.0
            landmark_marker.color.b = 0.0
            landmark_marker.color.a = 1.0
            marker_array.markers.append(landmark_marker)

        # Publish the MarkerArray
        self.map_publisher.publish(marker_array)
        self.get_logger().info(f"Published SLAM map with {len(self.landmarks)} landmarks.")

def main(args=None):
    rclpy.init(args=args)
    slam_map_node = SlamMapNode()
    rclpy.spin(slam_map_node)
    slam_map_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
