#!/usr/bin/env python3
import rospy
from spacemouse import SpaceMouse
from input2action import input2action
import numpy as np

from sensor_msgs.msg import JointState

from xarm.wrapper import XArmAPI
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped, PoseStamped
from scipy.spatial.transform import Rotation

def euler_to_quaternion(roll, pitch, yaw):
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return rot.as_quat()

class Spacemouse2Xarm:
    def __init__(self):
        rospy.init_node('spacemouse2xarm')
        ip = rospy.get_param('~robot_ip', '192.168.1.233')

        self.use_native_xarm_spacemouse = rospy.get_param('~use_native_xarm_spacemouse', False)
        
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        self.device = SpaceMouse(pos_sensitivity=1.0, rot_sensitivity=1.0)
        self.device.start_control()
        self.locked = False
        self.last_grasp_press_time = rospy.Time.now().to_nsec()

        # publisher for pose and twist action to send to JParse controller
        self.action_pub = rospy.Publisher('robot_action', TwistStamped, queue_size=10)
        self.position_action_pub = rospy.Publisher('robot_position_action', PoseStamped, queue_size=10)
        self.reset_sub = rospy.Subscriber('reset_xarm', Bool, self.reset_callback)
        self.is_resetting = False
        self.lock_hand = rospy.Publisher('lock_hand', Bool, queue_size=10)
        
        self.timer = rospy.Timer(rospy.Duration(1/30.0), self.timer_callback)
        
        # reset 
        self.is_resetting = True
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_position(x=331, y=20.3, z=308, roll=173, pitch=0, yaw=0,
                                speed=100, is_radian=False, wait=True)
        self.arm.motion_enable(enable=True)
        
        self.arm.set_mode(5)
        self.arm.set_state(0)
        self.is_resetting = False

        if not self.use_native_xarm_spacemouse:
            self.arm.set_mode(4)
            self.arm.set_state(0)
        
        self.joint_states_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        
        

    def reset_callback(self, msg):
        """
        a reset callback function to reset the robot to a predefined position
        """
        if msg.data:
            self.is_resetting = True
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.set_position(x=331, y=20.3, z=308, roll=173, pitch=0, yaw=0,
                                  speed=100, is_radian=False, wait=True)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(1)
            self.arm.set_state(0)
            self.is_resetting = False

    def action_to_cartesian_velocity(self, actions):
        """
        converts space mouse actions to cartesian velocities
        """
        if actions is None:
            return 0, 0, 0, 0, 0, 0

        dt = 1000 / 30.0
        scale = 100
        ang_scale = 1
        vx = -scale * actions[0] / dt
        vy = -scale * actions[1] / dt
        vz = scale * actions[2] / dt
        wx = -ang_scale * actions[3]
        wy = -ang_scale * actions[4]
        wz = ang_scale * actions[5]
        return vx, vy, vz, wx, wy, wz

    def timer_callback(self, event):
        cur_pose = self.arm.get_position()[1]
        cur_pose = np.array(cur_pose)
        actions, grasp = input2action(device=self.device, robot="xArm")
        
        if grasp is None:
            grasp = 0
        
        grasp = 850 - 860 * grasp
        
        # get joint states 
        status, joint_states = self.arm.get_joint_states(is_radian=True)
        joint_pos = joint_states[0]
        joint_vel = joint_states[1]
        joint_torque = joint_states[2]
                
        
        vx, vy, vz, wx, wy, wz = self.action_to_cartesian_velocity(actions)
        cur_pose = cur_pose + np.array([vx, vy, vz, wx, wy, wz])
        action_pose_msg = PoseStamped()
        action_pose_msg.pose.position.x = cur_pose[0]
        action_pose_msg.pose.position.y = cur_pose[1]
        action_pose_msg.pose.position.z = cur_pose[2]
        roll, pitch, yaw = cur_pose[3], cur_pose[4], cur_pose[5]
        q = euler_to_quaternion(roll, pitch, yaw)
        action_pose_msg.pose.orientation.x = q[0]
        action_pose_msg.pose.orientation.y = q[1]
        action_pose_msg.pose.orientation.z = q[2]
        action_pose_msg.pose.orientation.w = q[3]
        action_pose_msg.header.stamp = rospy.Time.now()

        self.position_action_pub.publish(action_pose_msg)
        twist_msg = TwistStamped()
        twist_msg.twist.linear.x = vx
        twist_msg.twist.linear.y = vy
        twist_msg.twist.linear.z = vz
        twist_msg.twist.angular.x = wx
        twist_msg.twist.angular.y = wy
        twist_msg.twist.angular.z = wz
        twist_msg.header.stamp = rospy.Time.now()
        
        
        position_velocity = np.array([vx, vy, vz])
        position_velocity_norm = np.linalg.norm(position_velocity)
        if position_velocity_norm > 0.05:
            position_velocity = position_velocity / (position_velocity_norm * 0.05)
        
        vx, vy, vz = position_velocity

        self.action_pub.publish(twist_msg)
        current_time_ns = rospy.Time.now().to_nsec()


        # grasp control
        if grasp == 1:
            if (current_time_ns - self.last_grasp_press_time) > 5e8:
                self.last_grasp_press_time = current_time_ns
                bool_msg = Bool()
                bool_msg.data = not self.locked
                self.locked = bool_msg.data
                self.lock_hand.publish(bool_msg)
        if not self.is_resetting:
            v_scaling = 1
            vx = vx * v_scaling
            vy = vy * v_scaling
            vz = vz * v_scaling
            if self.use_native_xarm_spacemouse:
                # use the native xarm cartesian velocity controller.
                self.arm.vc_set_cartesian_velocity([vx, vy, vz, wx, wy, wz], is_radian=True)
            
            # set gripper position
            # ret_grip = self.arm.set_gripper_position(grasp)

    def spin(self):
        rospy.spin()
        self.arm.disconnect()

if __name__ == '__main__':
    node = Spacemouse2Xarm()
    node.spin()
