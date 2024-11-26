#!/usr/bin/env python3
############################################################################################################
# SLAM - EXTENDED KALMAN FILTER for 4 wheel Independent Steer Independent Drive Robot
# Description:
# This script calculates the state of a 4 wheel Independent Steer Independent Drive Robot using an Extended Kalman Filter.
# Created by: Gavin Alexander Arpandy, Institut Teknologi Bandung, Indonesia, 2024
############################################################################################################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
import yaml
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from ekf_filter.functions_slam import StateTransitionFcn, Jacobian_Matrix, Measurement_Function, Jacobian_Landmark
import logging
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        
        # SUBSCRIBE TO THE TOPICS
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)
        self.joint_states_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)
        self.joint_command_subscription = self.create_subscription(
            JointState,
            '/joint_command',
            self.joint_command_callback,
            10)
        # Add Subscriber for Lidar data
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        #Add Publisher for Xnext
        self.state_publisher = self.create_publisher(
            Float32MultiArray,  
            '/ekf_state',      
            10                  
        )
        

        self.prev_time = self.get_clock().now()
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SensorFusionNode')

        # Internal storage for data
        self.imu_data = None
        self.joint_states_data = None
        self.joint_command_data = None
        self.control_input = np.zeros((8, 1)) 

        # EXTRACT DATA FROM PARAMETER CONFIG
        package_dir = get_package_share_directory('ekf_filter')
        yaml_file_path = os.path.join(package_dir, 'config', 'slam_ekf_config.yaml')
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        #DB SCAN CLUSTERING
        self.dbscan = DBSCAN(eps=config['eps'], min_samples=config['min_samples'])

        # INITIAL PARAMETERS
        self.state = np.array(config['initial_state']).reshape(-1, 1)
        P_init_state = np.array(config['P0_states'])
        self.Q_states = np.array(config['Q_states'])
        self.R_imu = np.array(config['R_imu'])
        self.R_encoder = np.array(config['R_encoder'])
        self.length = config['length']
        self.width = config['width']
        self.d = np.sqrt((self.width / 2)**2 + (self.length / 2)**2)
        self.wheel_radius = config['wheel_radius']
        self.landmark_treshold = config['landmark_treshold']
        self.landmarks = {}
        self.R_lidar = np.array(config['R_lidar'])
        self.landmark_accuracy = config['landmark_accuracy']
        self.Prm = np.zeros((len(self.state), 0)) #CHECKED [] Matrix
        self.Pmr = np.zeros((0, len(self.state)))   #CHECKED [] Matrix

        #Creating Full P matrix
        landmark_size = len(self.landmarks) * 2
        self.P = np.block([
                        [P_init_state, np.zeros((P_init_state.shape[0], landmark_size))],
                        [np.zeros((landmark_size, P_init_state.shape[1])), np.zeros((landmark_size, landmark_size))]
                    ])

        self.P_states = self.P #CHECKED 6x6 Matrix
    # EKF Predict Step without SLAM
    def predict_states(self):
        # Compute the state transition
        current_time = self.get_clock().now()
        self.delta_t = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time

        self.state[:6] = StateTransitionFcn(self.control_input, self.state[:6], self.delta_t).transition_function()
        self.state[2, 0] = (self.state[2, 0] + np.pi) % (2 * np.pi) - np.pi

        # Compute Jacobian of state transition
        F_states = Jacobian_Matrix(self.state[:6], self.delta_t).jacobian_transition_function()

        # Handle landmarks
        landmark_size = len(self.landmarks) * 2  # Total size of landmark states
        F = np.block([
            [F_states, np.zeros((len(self.state[:6]), landmark_size))],
            [np.zeros((landmark_size, len(self.state[:6]))), np.eye(landmark_size)]
        ])
        # Handle Q (noise covariance)
        self.Q = np.block([
            [self.Q_states, np.zeros((self.Q_states.shape[0], landmark_size))],
            [np.zeros((landmark_size, self.Q_states.shape[1])), np.zeros((landmark_size, landmark_size))]
        ])
        # Handle P (full covariance matrix)
        if landmark_size > 0:
            P = np.block([
                [self.P_states, self.Prm],
                [self.Pmr, np.zeros((landmark_size, landmark_size))]
            ])
        else:
            P = self.P_states  # No landmarks yet

        # Update full covariance matrix
        self.P = F @ P @ F.T + self.Q


        # Update submatrices
        self.P_states = self.P[:6, :6]


    # EKF Update Step for Robot State for IMU AND ENCODER
    def update_robot_state(self, z, R, H, z_pred):
        
        z = z.reshape(-1, 1)
        z_pred = z_pred.reshape(-1, 1)
        y = z - z_pred
        S = H @ self.P_states @ np.transpose(H) + R + np.eye(H.shape[0]) * 1e-6
        K = self.P_states @ np.transpose(H) @ np.linalg.inv(S)
        self.state[:6] = self.state[:6] + K @ y
        self.P_states = (np.eye(self.P_states.shape[0]) - K @ H) @ self.P_states
        self.logger.info(f'xnext: {self.state.flatten()}')


    # EKF Update Step for Landmark (LIDAR)
    def update_landmark_state(self, z, R, H, z_pred, landmark_index):
        # Compute the innovation (residual)
        z = z.reshape(-1, 1)
        z_pred = z_pred.reshape(-1, 1)
        y = z - z_pred
        # Innovation covariance
        S = H @ self.P @ H.T + R + np.eye(H.shape[0]) * 1e-6  

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)  # Full P for the Kalman gain
        # Update state estimate
        self.state = self.state + K @ y

        # Update the error covariance
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P
        self.P_states = self.P[:6, :6]
        self.Prm = self.P[:6, 6:]
        self.Pmr = self.P[6:, :6]
        self.logger.info(f'xnext: {self.state.flatten()}')
        
    # Callback Function To Handle Lidar Data
    def lidar_callback(self, msg):
        """
        Processes LiDAR data, clusters points, and updates SLAM EKF state.
        """

        # Parse LiDAR message
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Filter valid points within sensor range
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        ranges, angles = ranges[valid], angles[valid]

        # Convert polar coordinates (range, angle) to Cartesian (x, y) in robot's frame
        x_robot = ranges * np.cos(angles)
        y_robot = ranges * np.sin(angles)
        lidar_points = np.vstack((x_robot, y_robot)).T  # Shape: (N, 2)

        # Perform DBSCAN clustering
        if len(lidar_points) > 0:
            labels = self.dbscan.fit_predict(lidar_points)
            clusters = [lidar_points[labels == i] for i in set(labels) if i != -1]

            # Calculate cluster centers and radii
            centers = []
            for cluster in clusters:
                center = np.mean(cluster, axis=0)
                radius = np.mean(np.linalg.norm(cluster - center, axis=1))
                centers.append((center, radius))

            self.lidar_clusters = centers  # Store cluster centers and radii

        else:
            self.lidar_clusters = []

        # Use cluster centers as potential landmarks for SLAM
        self.process_landmarks_from_clusters()

    def process_landmarks_from_clusters(self):
        """
        Process clustered LiDAR data and update landmarks in SLAM.
        """
        if not self.lidar_clusters:
            return

        for center, radius in self.lidar_clusters:
            # Compute global position of the cluster center
            x_global = self.state[0, 0] + center[0] * np.cos(self.state[2, 0])
            y_global = self.state[1, 0] + center[1] * np.sin(self.state[2, 0])

            matched_landmark = False
            for landmark_id, (x, y) in self.landmarks.items():
                delta = np.array([[x_global - x], [y_global - y]])
                range_value = np.linalg.norm(delta)
                state_size = 6 + 2 * len(self.landmarks) # Total state vector size (robot + landmarks)
                H = Jacobian_Landmark(delta, range_value, landmark_id, state_size).jacobian_landmark_function()
                S = H @ self.P @ H.T + self.R_lidar
                mahalanobis_distance = np.sqrt(delta.T @ np.linalg.inv(S) @ delta)

                if mahalanobis_distance < self.landmark_treshold:
                    # Update existing landmark
                    matched_landmark = True
                    z = np.array([[x_global], [y_global]])
                    z_pred = np.array([[x], [y]])
                    self.update_landmark_state(z, self.R_lidar, H, z_pred, [6 + 2 * landmark_id, 6 + 2 * landmark_id + 1])
                    break

            if not matched_landmark:
                # Add a new landmark
                landmark_id = len(self.landmarks)
                self.landmarks[landmark_id] = (x_global, y_global)

                # Append the new landmark to the state vector
                self.state = np.vstack((self.state, [[x_global], [y_global]]))

                # Expand the covariance matrix
                new_size = self.state.shape[0]
                new_P = np.zeros((new_size, new_size))
                new_P[:self.P.shape[0], :self.P.shape[1]] = self.P
                new_P[-2:, -2:] = np.eye(2) * self.landmark_accuracy
                self.P = new_P

                # Update `self.Prm` and `self.Pmr` matrices
                self.Prm = self.P[:6, 6:]
                self.Pmr = self.P[6:, :6]
                

    # CALLBACK FUNCTION TO HANDLE ENCODER DATA
    def joint_states_callback(self, msg):
        self.joint_states_data = {
            'names': msg.name,
            'positions': msg.position,
            'velocities': msg.velocity,
            'efforts': msg.effort 
        }
        """
        Structure of message array: #Different from Joint_command
        name:
            - left_back_connector_joint
            - left_front_connector_joint
            - right_back_connector_joint
            - right_front_connector_joint
            - left_back_wheel_joint
            - left_front_wheel_joint
            - right_back_wheel_joint
            - right_front_wheel_joint

        """
        current_time = self.get_clock().now()
        self.delta_t = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time

        self.L = self.length / 2
        self.W = self.width / 2
        encoder_noise = np.random.normal(0, 0.05, (8, 1))
        # Extract wheel velocities and steering angles
        self.speedRF = self.joint_states_data['velocities'][7] + encoder_noise[0, 0] #ADDED NOISE TO THE ENCODER SINCE NO NOISE IN SIMULATION
        self.speedLF = self.joint_states_data['velocities'][5] + encoder_noise[1, 0] #ADDED NOISE
        self.speedLB = self.joint_states_data['velocities'][4] + encoder_noise[2, 0] #ADDED NOISE
        self.speedRB = self.joint_states_data['velocities'][6] + encoder_noise[3, 0] #ADDED NOISE
        self.angleRF = self.joint_states_data['positions'][3] + np.pi / 2 + encoder_noise[4, 0] #ADDED NOISE
        self.angleLF = self.joint_states_data['positions'][1] + np.pi / 2 + encoder_noise[5, 0] #ADDED NOISE
        self.angleLB = self.joint_states_data['positions'][0] + np.pi / 2 + encoder_noise[6, 0] #ADDED NOISE
        self.angleRB = self.joint_states_data['positions'][2] + np.pi / 2 + encoder_noise[7, 0] #ADDED NOISE

        # Compute velocity components for each wheel
        Vx_RF = self.speedRF * np.cos(self.angleRF)
        Vy_RF = self.speedRF * np.sin(self.angleRF)
        Vx_LF = self.speedLF * np.cos(self.angleLF)
        Vy_LF = self.speedLF * np.sin(self.angleLF)
        Vx_LB = self.speedLB * np.cos(self.angleLB)
        Vy_LB = self.speedLB * np.sin(self.angleLB)
        Vx_RB = self.speedRB * np.cos(self.angleRB)
        Vy_RB = self.speedRB * np.sin(self.angleRB)

        # Average translational velocities
        self.v_x_encoder = 0.25 * (Vx_RF + Vx_LF + Vx_LB + Vx_RB)
        self.v_y_encoder = 0.25 * (Vy_RF + Vy_LF + Vy_LB + Vy_RB)
        self.omega_encoder = 0.25 * ((-Vy_RF + Vy_LF + Vy_LB - Vy_RB) / (2 * self.L) +
                                     (Vx_RF + Vx_LF - Vx_LB - Vx_RB) / (2 * self.W))
        encoder_measurement = np.array([
            [self.v_x_encoder],
            [self.v_y_encoder],
            [self.omega_encoder]
        ])
        if np.isnan(encoder_measurement).any() or np.isinf(encoder_measurement).any():
            self.logger.error("Received NaN or Inf in encoder data. Skipping this update.")
            return
        self.logger.info(f'encoder_measurement: {encoder_measurement.flatten()}')

        # Compute H_encoder and z_pred for encoder
        H_encoder = Jacobian_Matrix(self.state, self.delta_t).jacobian_encoder_function()
        z_pred_encoder = Measurement_Function(self.state).encoder_measurement_function()

        # Perform EKF update using encoder data
        self.update_robot_state(encoder_measurement, self.R_encoder, H_encoder, z_pred_encoder)

        # Publish the updated state
        state_msg = Float32MultiArray()
        state_msg.data = self.state[:6].flatten().tolist()  # Flatten the state array and convert to a list

        # Publish the state message
        self.state_publisher.publish(state_msg)
    
    # CALLBACK FUNCTION TO HANDLE IMU DATA
    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        self.delta_t = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time
        self.imu_data = {
            'acceleration': {
                'x': msg.linear_acceleration.x,
                'y': msg.linear_acceleration.y,
                'z': msg.linear_acceleration.z
            },
            'angular_velocity': {
                'x': msg.angular_velocity.x,
                'y': msg.angular_velocity.y,
                'z': msg.angular_velocity.z
            } 
        }
        # IMU measurement vector 
        imu_noise = np.random.normal(0, 0.01, (3, 1))
        imu_measurement = np.array([
            [self.imu_data['acceleration']['x'] + imu_noise[0, 0]], #ADDED NOISE IN THE SIMULATION SINCE THE SIMULATION DOES NOT HAVE NOISE
            [self.imu_data['acceleration']['y'] + imu_noise[1, 0]], #ADDED NOISE 
            [self.imu_data['angular_velocity']['z'] + imu_noise[2, 0]]  #ADDED NOISE 
        ])


        if np.isnan(imu_measurement).any() or np.isinf(imu_measurement).any():
            self.logger.error("Received NaN or Inf in IMU data. Skipping this update.")
            return

        # Compute H_imu and z_pred for IMU
        H_imu = Jacobian_Matrix(self.state, self.delta_t).jacobian_imu_function()
        z_pred_imu = Measurement_Function(self.state).imu_measurement_function()

        # Perform EKF update using IMU data
        self.update_robot_state(imu_measurement, self.R_imu, H_imu, z_pred_imu)



    # CALLBACK FUNCTION TO HANDLE CONTROL INPUTS
    def joint_command_callback(self, msg):
        self.joint_command_data = {
            'names': msg.name,
            'positions': msg.position,
            'velocities': msg.velocity,
            'efforts': msg.effort
        }
        # Prepare control input vector
        self.control_input = np.array([
            self.joint_command_data['velocities'][4],
            self.joint_command_data['velocities'][5],
            self.joint_command_data['velocities'][6],
            self.joint_command_data['velocities'][7],
            self.joint_command_data['positions'][0] + np.pi / 2,
            self.joint_command_data['positions'][1] + np.pi / 2,
            self.joint_command_data['positions'][2] + np.pi / 2,
            self.joint_command_data['positions'][3] + np.pi / 2
        ]).reshape((8, 1))

        # Perform the EKF prediction step
        self.predict_states()

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()
    rclpy.spin(sensor_fusion_node)
    sensor_fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
