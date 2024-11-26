import numpy as np
from ament_index_python.packages import get_package_share_directory
import yaml
import os
############################################################################################################
#STATE TRANSITION FUNCTION, MEASUREMENT FUNCTIONS, AND JACOBIAN FUNCTIONS
#Description:
#This script defines the state transition function, measurement functions and the Jacobian matrix of the transition function for 
#a 4 wheel Independent Steer Independent Drive Robot.
#Created by: Gavin Alexander Arpandy, Institut Teknologi Bandung, Indonesia, 2024
############################################################################################################

class StateTransitionFcn:
    def __init__(self,control_input,state_vector, delta_t):
        """
        STATE VECTOR state_vector[x,y,theta,vx,vy,omega]T
        Control Input: control_input[v1,v2,v3,v4,delta1,delta2,delta3,delta4]T
        """
        #EXTRACT DATA FROM PARAMETER CONFIG
        package_dir = get_package_share_directory('ekf_filter')
        yaml_file_path = os.path.join(package_dir, 'config', 'ekf_config.yaml')
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        #Extract State vector X
        self.x_k = float(state_vector[0,0])
        self.y_k = float(state_vector[1,0])
        self.theta_k = float(state_vector[2,0])

        #Extract Control Inputs
        self.v = np.array([control_input[0,0], control_input[1,0], control_input[2,0], control_input[3,0]]) #v1, v2, v3, v4
        self.delta = np.array([control_input[4,0], control_input[5,0], control_input[6,0], control_input[7,0]])  #delta1, delta2, delta3, delta4

        #Extract Parameters
        self.delta_t = delta_t
        self.L = config['length']/2
        self.W = config['width']/2

        #Compute velocity componenets for next state
        self.vx_rf = self.v[0] * np.cos(self.delta[0])  
        self.vy_rf = self.v[0] * np.sin(self.delta[0])  
        self.vx_lf = self.v[1] * np.cos(self.delta[1])  
        self.vy_lf = self.v[1] * np.sin(self.delta[1])  
        self.vx_lb = self.v[2] * np.cos(self.delta[2])  
        self.vy_lb = self.v[2] * np.sin(self.delta[2])  
        self.vx_rb = self.v[3] * np.cos(self.delta[3])  
        self.vy_rb = self.v[3] * np.sin(self.delta[3])


    #Function to compute the next state
    def transition_function(self):
        #Compute the velocity components for the next state
        self.v_x_next = 0.25 * (self.vx_rf + self.vx_lf + self.vx_lb + self.vx_rb)
        self.v_y_next = 0.25 * (self.vy_rf + self.vy_lf + self.vy_lb + self.vy_rb)
        self.omega_next = 0.25 * ((-self.vy_rf + self.vy_lf + self.vy_lb - self.vy_rb) / (2 * self.L) +
                             (self.vx_rf + self.vx_lf - self.vx_lb - self.vx_rb) / (2 * self.W))
        
        #Compute x_{k+1}, y_{k+1}, and theta_{k+1}
        self.x_next = self.x_k + self.v_x_next * self.delta_t * np.cos(self.theta_k) - self.v_y_next * self.delta_t * np.sin(self.theta_k)
        self.y_next = self.y_k + self.v_x_next * self.delta_t * np.sin(self.theta_k) + self.v_y_next * self.delta_t * np.cos(self.theta_k)
        self.theta_next = self.theta_k + self.omega_next * self.delta_t

        self.state_vector_next = np.array([[self.x_next],[self.y_next],[self.theta_next],[self.v_x_next],[self.v_y_next],[self.omega_next]])
        return self.state_vector_next

class Jacobian_Matrix:
    def __init__(self,state_vector, delta_t):
        """
        STATE VECTOR state_vector[x,y,theta,vx,vy,omega]T
        """
        #EXTRACT DATA FROM PARAMETER CONFIG
        package_dir = get_package_share_directory('sensor_fusion')
        yaml_file_path = os.path.join(package_dir, 'config', 'ekf_config.yaml')
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        #Extract State Vector
        self.theta_k = state_vector[2,0]
        self.v_x_k = state_vector[3,0]
        self.v_y_k = state_vector[4,0]
        self.omega_k = state_vector[5,0]
        self.delta_t = delta_t
    #Function to calculate the Jacobian matrix of the transition function
    def jacobian_transition_function(self):
        self.A_Jacobian_matrix = np.array([
            [1, 0, (-self.v_x_k * np.sin(self.theta_k) - self.v_y_k * np.cos(self.theta_k)) * self.delta_t, np.cos(self.theta_k) * self.delta_t, -np.sin(self.theta_k) * self.delta_t, 0],
            [0, 1, (self.v_x_k * np.cos(self.theta_k) - self.v_y_k * np.sin(self.theta_k)) * self.delta_t, np.sin(self.theta_k) * self.delta_t, np.cos(self.theta_k) * self.delta_t, 0],
            [0, 0, 1, 0, 0, self.delta_t],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        return self.A_Jacobian_matrix
    #Function to calculate the Jacobian matrix of the encoder measurement
    def jacobian_encoder_function(self):
        self.H_encoder_Jacobian_matrix = np.array([
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ])
        return self.H_encoder_Jacobian_matrix
    
    #Function to calculate the Jacobian matrix of the imu measurement
    def jacobian_imu_function(self):
        self.H_imu_Jacobian_matrix = np.array([
    [0, 0, 0, 0, self.omega_k, self.v_y_k],
    [0, 0, 0, -self.omega_k, 0, -self.v_x_k], 
    [0, 0, 0, 0, 0, 1],
])

        return self.H_imu_Jacobian_matrix
    
class Measurement_Function:
    def __init__(self,state_vector):
        """
        STATE VECTOR state_vector[x,y,theta,vx,vy,omega]T
        """
        self.theta_angle = 0.0 #Rotation angle of global coordinates to imu coordinates
        self.v_x_k = state_vector[3,0]
        self.v_y_k = state_vector[4,0]
        self.omega_k = state_vector[5,0]

    #This following function is the h(k) function for the imu measurement sensor
    def imu_measurement_function(self): 
        self.v_x_body_imu = np.cos(self.theta_angle) * self.v_x_k + np.sin(self.theta_angle) * self.v_y_k
        self.v_y_body_imu = -np.sin(self.theta_angle) * self.v_x_k + np.cos(self.theta_angle) * self.v_y_k

        # Compute the Imu acceleration in the body frame
        self.imu_ax = 0.0 + self.omega_k * self.v_y_body_imu   # Since the State Vectors only have velocity data, v_dotx is '0'
        self.imu_ay = 0.0 - self.omega_k * self.v_x_body_imu   # Since the State Vectors only have velocity data, v_doty is '0'
        self.imu_omega = self.omega_k
        
        # Output
        self.imu_measurement_zk = np.array([self.imu_ax, self.imu_ay, self.imu_omega])
        return self.imu_measurement_zk

    #This following function is the h(k) function for the encoder measurement sensor
    def encoder_measurement_function(self): 
        self.encoder_v_x_k = self.v_x_k  
        self.encoder_v_y_k = self.v_y_k  
        self.encoder_omega_k = self.omega_k  
        self.encoder_measurement_zk = np.array([self.encoder_v_x_k, self.encoder_v_y_k, self.encoder_omega_k]) 
        
        return self.encoder_measurement_zk


