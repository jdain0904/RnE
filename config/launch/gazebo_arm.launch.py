from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('hpso_ik')
    urdf = os.path.join(pkg, 'urdf', 'fourdof_arm.urdf')
    ctrl = os.path.join(pkg, 'config', 'controllers.yaml')

    # Gazebo 실행 + 로봇 스폰
    return LaunchDescription([
        ExecuteProcess(cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'], output='screen'),
        Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
             arguments=['-entity', 'fourdof_arm', '-file', urdf, '-x', '0', '-y', '0', '-z', '0.05']),
        # controller manager spawner
        Node(package='controller_manager', executable='spawner', output='screen',
             arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager']),
        Node(package='controller_manager', executable='spawner', output='screen',
             arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager']),
    ])
