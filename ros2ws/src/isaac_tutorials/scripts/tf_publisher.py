#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Pose
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher_ = self.create_publisher(Pose, 'pose_data', 10)
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.publish_pose)  # 10 Hz

    def publish_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', 'collisions', now)
            
            pose = Pose()
            pose.position.x = trans.transform.translation.x
            pose.position.y = trans.transform.translation.y
            pose.position.z = trans.transform.translation.z
            pose.orientation.x = trans.transform.rotation.x
            pose.orientation.y = trans.transform.rotation.y
            pose.orientation.z = trans.transform.rotation.z
            pose.orientation.w = trans.transform.rotation.w

            self.publisher_.publish(pose)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn('Could not transform map to base_link: ' + str(e))

def main(args=None):
    rclpy.init(args=args)
    pose_publisher = PosePublisher()
    
    try:
        rclpy.spin(pose_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        pose_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
