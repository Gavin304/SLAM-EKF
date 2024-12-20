#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

class PoseRepublisher(Node):
    def __init__(self):
        super().__init__('pose_republisher')
        self.pose_pub = self.create_publisher(PoseStamped, '/pose', 10)  # Publish pose on /pose topic
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.publish_pose)  # Publish at 10 Hz

    def publish_pose(self):
        try:
            # Lookup the transform between map and chassis
            transform = self.tf_buffer.lookup_transform('world', 'chassis', rclpy.time.Time())
            
            # Create and populate the PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'world'
            pose_msg.pose.position.x = transform.transform.translation.x
            pose_msg.pose.position.y = transform.transform.translation.y
            pose_msg.pose.position.z = transform.transform.translation.z
            pose_msg.pose.orientation = transform.transform.rotation
            
            # Publish the PoseStamped message
            self.pose_pub.publish(pose_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to get transform: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
