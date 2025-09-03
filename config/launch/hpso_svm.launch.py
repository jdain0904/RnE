from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hpso_ik',
            executable='hpso_svm_node',
            name='hpso_svm_node',
            output='screen',
            parameters=[{
                'max_iterations': 120,
                'master_pop_size': 12,
                'sub_pop_size': 6,
                'w': 0.5, 'c1': 1.5, 'c2': 1.5,
                'target_margin': 1.0, 'w_margin': 0.5, 'reduction': 'mean',
                'fixed_frame': 'base',
                'controller_ns': '/joint_trajectory_controller',
                'joint_names': ['joint1','joint2','joint3','joint4']
            }]
        ),
    ])
