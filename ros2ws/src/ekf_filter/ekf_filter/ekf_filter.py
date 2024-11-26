#!/usr/bin/env python3
############################################################################################################
# EXTENDED KALMAN FILTER for 4 wheel Independent Steer Independent Drive Robot
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
from ekf_filter.functions import StateTransitionFcn, Jacobian_Matrix, Measurement_Function
import logging
from std_msgs.msg import Float32MultiArray


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
        yaml_file_path = os.path.join(package_dir, 'config', 'ekf_config.yaml')
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        # INITIAL PARAMETERS
        self.state = np.array(config['initial_state']).reshape(-1, 1)
        self.P = np.array(config['P0'])
        self.Q = np.array(config['Q'])
        self.R_imu = np.array(config['R_imu'])
        self.R_encoder = np.array(config['R_encoder'])
        self.length = config['length']
        self.width = config['width']
        self.d = np.sqrt((self.width / 2)**2 + (self.length / 2)**2)
        self.wheel_radius = config['wheel_radius']

    # EKF Predict Step
    def predict(self):
        # Compute the state transition
        current_time = self.get_clock().now()
        self.delta_t = (current_time - self.prev_time).nanoseconds / 1e9 # Convert to seconds
        self.prev_time = current_time
        

        self.state = StateTransitionFcn(self.control_input, self.state, self.delta_t).transition_function()
        self.state[2, 0] = (self.state[2, 0] + np.pi) % (2 * np.pi) - np.pi
        # Compute the Jacobian of the state transition function
        F = Jacobian_Matrix(self.state, self.delta_t).jacobian_transition_function()
        # Update the error covariance
        self.P = F @ self.P @ np.transpose(F) + self.Q
        # Log the xnext values (updated state)
        


    # EKF Update Step
    def update(self, z, R, H, z_pred):
        # Compute the innovation (residual)
        z = z.reshape(-1, 1)
        z_pred = z_pred.reshape(-1, 1)
        y = z - z_pred 
        # Innovation covariance
        S = H @ self.P @ np.transpose(H) + R + np.eye(H.shape[0]) * 1e-6
        # Kalman gain
        K = self.P @ np.transpose(H) @ np.linalg.inv(S)
        # Update state estimate
        self.state = self.state + K @ y
        # Update the error covariance
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P
        self.logger.info(f'xnext: {self.state.flatten()}')
        

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
        self.update(imu_measurement, self.R_imu, H_imu, z_pred_imu)
        

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
        #self.logger.info(f'encoder_measurement: {encoder_measurement.flatten()}')

        # Compute H_encoder and z_pred for encoder
        H_encoder = Jacobian_Matrix(self.state, self.delta_t).jacobian_encoder_function()
        z_pred_encoder = Measurement_Function(self.state).encoder_measurement_function()

        # Perform EKF update using encoder data
        self.update(encoder_measurement, self.R_encoder, H_encoder, z_pred_encoder)

        # Publish the updated state
        state_msg = Float32MultiArray()
        state_msg.data = self.state.flatten().tolist()  # Flatten the state array and convert to a list

        # Publish the state message
        self.state_publisher.publish(state_msg)

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
        self.predict()

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()
    rclpy.spin(sensor_fusion_node)
    sensor_fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
