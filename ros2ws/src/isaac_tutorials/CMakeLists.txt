cmake_minimum_required(VERSION 3.5)
project(isaac_tutorials)

set(Python3_EXECUTABLE "C:/Python38/python.exe")
set(Python3_INCLUDE_DIR "C:/Python38/include")
set(Python3_LIBRARY "C:/Python38/libs/python38.lib")
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter Development)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY
  rviz2
  scripts
  DESTINATION share/${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/ros2_publisher.py
  scripts/tf_publisher.py
  scripts/ros2_move_ctrl_sim.py
  scripts/ros2_fork_ctrl_sim.py
  scripts/Omniverse_control.py
  scripts/wheeltec_keyboard.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
