# IMPLEMENTATION OF SLAM-EKF for Independent Steer & Independent Drive robot on Isaac SIM
How to run:
1. install all dependencies
```
pip install -r requirements.txt
```
2. build ROS2 workspace
```
cd ros2ws
colcon build
```
3. To run EKF-filter (in ros2ws)
```
cd
source /opt/ros/humble/setup.bash
cd ros2ws
source install/setup.bash
ros2 run ekf_filter ekf_filter 
```
4. To run Slam-EKF
```
cd
source /opt/ros/humble/setup.bash
cd ros2ws
source install/setup.bash
ros2 run ekf_filter slam_ekf
```
Reference for installation guide:
ROS2 Humble:
https://docs.ros.org/en/humble/Installation.html
ISAAC SIM 4.2.0:
https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html
ISAAC SIM .usd file not available due to copywrite
