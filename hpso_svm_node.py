import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

from hpso_ik.algos_svm import (
    DOF, joint_bounds, forward_kinematics_3d,
    train_default_judge, hpso_sliding_window_with_svm
)

class HpsoSvmNode(Node):
    def __init__(self):
        super().__init__('hpso_svm_node')

        # 파라미터
        self.declare_parameter('max_iterations', 120)
        self.declare_parameter('master_pop_size', 12)
        self.declare_parameter('sub_pop_size', 6)
        self.declare_parameter('w', 0.5)
        self.declare_parameter('c1', 1.5)
        self.declare_parameter('c2', 1.5)
        self.declare_parameter('target_margin', 1.0)
        self.declare_parameter('w_margin', 0.5)
        self.declare_parameter('reduction', 'mean')
        self.declare_parameter('fixed_frame', 'base')
        self.declare_parameter('controller_ns', '/joint_trajectory_controller')
        self.declare_parameter('joint_names', ['joint1','joint2','joint3','joint4'])

        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.controller_ns = self.get_parameter('controller_ns').value
        self.joint_names = list(self.get_parameter('joint_names').value)

        self.get_logger().info('Training/Loading SVM judge...')
        self.judge = train_default_judge()
        self.get_logger().info('SVM judge ready.')

        # I/O
        self.sub_target = self.create_subscription(Point, 'target', self.on_target, 10)
        self.pub_js     = self.create_publisher(JointState, 'joint_solution', 10)
        self.pub_marker = self.create_publisher(Marker, 'ee_marker', 10)

        # FollowJointTrajectory action client
        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            f'{self.controller_ns}/follow_joint_trajectory'
        )

        self.get_logger().info('Publish geometry_msgs/Point on /target to solve and move the arm.')

    def send_trajectory(self, q):
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Trajectory action server not available')
            return
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(a) for a in q]
        pt.time_from_start.sec = 2
        traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.get_logger().info(f'Sending trajectory to {self.controller_ns} for joints {self.joint_names}')
        self.action_client.send_goal_async(goal)  # fire-and-forget

    def on_target(self, msg: Point):
        target = np.array([msg.x, msg.y, msg.z], float)
        self.get_logger().info(f'Target: {target}')
        final_angles, final_error, g_hist, f_hist, t_hist, n_eval = \
            hpso_sliding_window_with_svm(
                target_function=lambda t: target,
                svm_judge=self.judge,
                master_pop_size=int(self.get_parameter('master_pop_size').value),
                sub_pop_size=int(self.get_parameter('sub_pop_size').value),
                max_iterations=int(self.get_parameter('max_iterations').value),
                w=float(self.get_parameter('w').value),
                c1=float(self.get_parameter('c1').value),
                c2=float(self.get_parameter('c2').value),
                target_margin=float(self.get_parameter('target_margin').value),
                w_margin=float(self.get_parameter('w_margin').value),
                reduction=str(self.get_parameter('reduction').value)
            )

        # publish JointState
        js = JointState()
        js.name = [f'joint_{i+1}' for i in range(DOF)]
        js.position = [float(a) for a in final_angles]
        self.pub_js.publish(js)

        # marker for EE
        ee = forward_kinematics_3d(final_angles)[-1]
        mk = Marker()
        mk.header.frame_id = self.fixed_frame
        mk.ns = 'hpso_ik_svm'
        mk.id = 1
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.pose.position.x = float(ee[0])
        mk.pose.position.y = float(ee[1])
        mk.pose.position.z = float(ee[2])
        mk.scale.x = mk.scale.y = mk.scale.z = 0.03
        mk.color.r = 0.0; mk.color.g = 0.6; mk.color.b = 1.0; mk.color.a = 1.0
        mk.lifetime = Duration()
        self.pub_marker.publish(mk)

        self.get_logger().info(f'Solved. err={final_error:.5f}, evals={n_eval}, q={np.array2string(final_angles, precision=3)}')

        # send to controller
        self.send_trajectory(final_angles)

def main():
    rclpy.init()
    node = HpsoSvmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
